"""
Simple_Residual: A simplified residual model for semantic ID generation.

Standard autoregressive format: <BOS> sid_0 sid_1 sid_2 ... sid_{n-1}
No residual injection into decoder input — the decoder runs exactly like
the base TIGER model. The only difference is an auxiliary loss head:

At each position k, an output_adapter[k] maps the decoder hidden state to
the RQ-VAE latent space, then two auxiliary losses are computed:
  1. Codebook loss (soft or hard label): output_adapter[k](hidden_k) vs
     codebook[k], predicting which codebook entry was used.
  2. Cumulative residual loss: constrains output_adapter[k](hidden_k) to
     approximate sum(codebook[t][sid_t] for t in k+1..n-1), ensuring the
     representation captures the full remaining semantic content.

This is the simplest possible residual design — the model's decoder path
is completely unchanged, and residual information only appears as a
training-side auxiliary signal.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from .tiger import TIGER


class Simple_Residual(TIGER):
    """
    Simplified TIGER with auxiliary codebook/residual losses only.

    The decoder runs standard autoregressive: <BOS> sid_0 sid_1 ... sid_{n-1}.
    No residual tokens are interleaved, no residual is injected into the input.
    Auxiliary losses are computed from output_adapter projections of each
    position's hidden state.

    Args:
        config: T5Config
        n_semantic_codebook: number of RQ-VAE semantic codebook levels
        max_items_per_seq: max items per sequence
        flag_use_output_embedding: whether to output item embedding
        flag_use_learnable_text_embed: whether to use learnable text embedding
        embedding_head_dict: config for embedding head
        rqvae_codebook_weights: list of [codebook_size, latent_size] tensors
        codebook_size: size of each codebook
        latent_size: RQ-VAE latent dimension (codebook embedding dim)
        codebook_loss_weight: weight for codebook selection CE loss
        soft_label_K: number of nearest codebook entries for soft labels.
            0 = hard label (standard CE). Default: 0.
        soft_label_temperature: temperature for soft-label softmax.
            Higher = more uniform, lower = sharper. Default: 1.0.
        soft_label_temp_min: minimum temperature for soft-label temperature scheduling.
            As training progresses, the soft-label temperature linearly decays from
            soft_label_temperature to soft_label_temp_min over soft_label_temp_decay_steps.
            Only used when soft_label_temp_min is set. Default: None (no scheduling).
        soft_label_temp_decay_steps: total training steps over which the temperature
            decays from soft_label_temperature to soft_label_temp_min.
            Each forward_residual call counts as one step.
            Default: 10000.
        codebook_loss_weight_decay_steps: per-position codebook loss weight decay schedule.
            null = no decay (backward compatible, weight stays at codebook_loss_weight).
            list of (int or null) per codebook level, e.g. [5000, null, null, null]
            → position 0 decays over 5000 steps, others stay fixed.
            Uses cosine decay: alpha = 0.5 * (1 + cos(pi * step / decay_steps)),
            decaying from 1.0 to 0.0. The effective weight for position k is:
              codebook_loss_weight * alpha_k(step).
            This implements "training wheel removal" — auxiliary loss provides
            structure prior early on (accelerating convergence) but may become
            a ceiling later when soft-label nearest-neighbors are imperfect.
            Cosine decay keeps weight high early, then rapidly withdraws the
            signal in mid-training. Each forward_residual call counts as one step.
            Default: None.
        cumulative_residual_loss_weight: weight for cumulative residual
            alignment loss. At step k, constrains output_adapter[k](hidden_k)
            to approximate sum(codebook[t][sid_t] for t in k+1..n-1).
            Default: 0.0 (disabled).
        cumulative_residual_loss_type: "mse", "cosine", or "ce".
            Default: "mse".
        cumulative_residual_loss_temperature: temperature for "ce" type.
            Default: 1.0.
    """

    def __init__(
        self,
        config: T5Config,
        n_semantic_codebook: int,
        max_items_per_seq: int,
        flag_use_output_embedding: bool = False,
        flag_use_learnable_text_embed: bool = False,
        embedding_head_dict: dict = None,
        rqvae_codebook_weights: list = None,
        codebook_size: int = 256,
        latent_size: int = 128,
        codebook_loss_weight: float = 1.0,
        soft_label_K: int = 0,
        soft_label_temperature: float = 1.0,
        soft_label_temp_min: float = None,
        soft_label_temp_decay_steps: int = 10000,
        codebook_loss_weight_decay_steps: list = None,
        cumulative_residual_loss_weight: float = 0.0,
        cumulative_residual_loss_type: str = "mse",
        cumulative_residual_loss_temperature: float = 1.0,
    ):
        super().__init__(
            config=config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=max_items_per_seq,
            flag_use_output_embedding=flag_use_output_embedding,
            flag_use_learnable_text_embed=flag_use_learnable_text_embed,
            embedding_head_dict=embedding_head_dict,
        )

        self.latent_size = latent_size
        self.d_model = config.d_model
        self.codebook_size = codebook_size
        self.codebook_loss_weight = codebook_loss_weight
        self.n_semantic_codebook = n_semantic_codebook
        self.soft_label_K = soft_label_K
        self.soft_label_temperature = soft_label_temperature
        self.soft_label_temp_min = soft_label_temp_min
        self.soft_label_temp_decay_steps = soft_label_temp_decay_steps
        self._soft_label_temp_step = 0  # step counter for temperature scheduling
        self.codebook_loss_weight_decay_steps = codebook_loss_weight_decay_steps
        self._codebook_weight_decay_step = 0  # step counter for weight decay scheduling
        self.cumulative_residual_loss_weight = cumulative_residual_loss_weight
        self.cumulative_residual_loss_type = cumulative_residual_loss_type
        self.cumulative_residual_loss_temperature = cumulative_residual_loss_temperature

        # Codebook offsets for SID -> codebook index conversion
        self._codebook_offsets = None

        # Register codebook embeddings (frozen, from RQ-VAE)
        if rqvae_codebook_weights is not None:
            for i, cb_weight in enumerate(rqvae_codebook_weights):
                if cb_weight.shape[0] > codebook_size:
                    cb_weight = cb_weight[:codebook_size]
                self.register_buffer(
                    f"codebook_emb_{i}",
                    cb_weight.detach().clone(),
                )
        else:
            for i in range(n_semantic_codebook):
                self.register_buffer(
                    f"codebook_emb_{i}",
                    torch.randn(codebook_size, latent_size) * 0.01,
                )

        # Output adapters: map hidden state (d_model) -> latent space (latent_size)
        # One per codebook level. Used for:
        #   1. Codebook classification loss: output_adapter[k](hidden_k) vs codebook[k]
        #   2. Cumulative residual loss: output_adapter[k](hidden_k) vs sum of remaining codebooks
        self.output_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.latent_size, bias=False),
                nn.LayerNorm(self.latent_size, eps=config.layer_norm_epsilon),
            )
            for _ in range(self.n_semantic_codebook)
        ])

    @property
    def codebook_embs(self):
        """Return list of codebook embedding tensors."""
        return [
            getattr(self, f"codebook_emb_{i}")
            for i in range(self.n_semantic_codebook)
        ]

    def set_codebook_offsets(self, codebook_sizes):
        """
        Set SID token offsets for variable codebook sizes.

        Args:
            codebook_sizes: list of per-level codebook sizes, or int for uniform.
        """
        if isinstance(codebook_sizes, int):
            n_levels = self.n_semantic_codebook
            self._codebook_offsets = [codebook_sizes * i + 1 for i in range(n_levels)]
        else:
            offsets = [1]
            for sz in codebook_sizes[:-1]:
                offsets.append(offsets[-1] + sz)
            self._codebook_offsets = offsets

    def _get_codebook_idx(self, sid_tokens, level):
        """Convert SID token to codebook index (0 to codebook_size-1)."""
        if self._codebook_offsets is not None:
            cb_idx = sid_tokens - self._codebook_offsets[level]
        else:
            cb_idx = sid_tokens - 1 - level * self.codebook_size
        return cb_idx.clamp(0, self.codebook_size - 1)

    def _compute_code_loss(self, pred_latent, level, labels_sids, B, device, soft_label_temp=None):
        """
        Compute codebook cross-entropy loss for a given level.

        Args:
            pred_latent: [B, latent_size] — predicted latent from output adapter
            level: which codebook level this targets
            labels_sids: [B, n_codebook] — ground-truth sid tokens
            B: batch size
            device: torch device

        Returns:
            scalar loss (averaged over batch)
        """
        cb_weights = self.codebook_embs[level]  # [codebook_size, latent_size]
        logits_cb = pred_latent @ cb_weights.T.float()  # [B, codebook_size]
        cb_idx = self._get_codebook_idx(labels_sids[:, level], level)  # [B]

        if self.soft_label_K > 0:
            # Ground truth embedding
            gt_emb = cb_weights[cb_idx].float()  # [B, latent_size]

            # Squared L2 distance from each codebook entry to the GT
            dist_sq = (
                (cb_weights.float().unsqueeze(0) - gt_emb.unsqueeze(1)) ** 2
            ).sum(-1)  # [B, codebook_size]

            # Top-K nearest entries
            K = min(self.soft_label_K, self.codebook_size)
            _, topk_indices = torch.topk(dist_sq, k=K, dim=-1, largest=False)  # [B, K]

            # L2 distance (sqrt) for better softmax scaling
            topk_dist_sq = dist_sq.gather(1, topk_indices)  # [B, K]
            topk_dist = torch.sqrt(topk_dist_sq.clamp(min=1e-8))  # [B, K]

            # Soft labels: softmax on negative distances / temperature
            _temp = soft_label_temp if soft_label_temp is not None else self.soft_label_temperature
            neg_scaled_dist = -topk_dist / _temp  # [B, K]
            soft_probs_topk = F.softmax(neg_scaled_dist, dim=-1)  # [B, K]

            # Build full soft label distribution
            soft_labels = torch.zeros(
                B, self.codebook_size, device=device, dtype=torch.float32
            )
            soft_labels.scatter_(1, topk_indices, soft_probs_topk)

            # Cross-entropy with soft targets
            log_probs = F.log_softmax(logits_cb, dim=-1)  # [B, codebook_size]
            return -(soft_labels * log_probs).sum(dim=-1).mean()
        else:
            # Hard-label: standard CE
            return F.cross_entropy(logits_cb, cb_idx, reduction="sum") / B

    def forward_residual(
        self,
        input_ids=None,
        attention_mask=None,
        labels_sids=None,
        inputs_embeds=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Forward pass with auxiliary codebook/residual losses.

        Decodes in standard autoregressive format: <BOS> sid_0 sid_1 ... sid_{n-1}
        The decoder path is identical to base TIGER. After each step, the hidden
        state is projected to latent space via output_adapter[k], and auxiliary
        losses (codebook + cumulative residual) are computed.

        Args:
            input_ids: [B, enc_seq_len] encoder input token IDs
            attention_mask: [B, enc_seq_len] encoder attention mask
            labels_sids: [B, n_codebook] ground truth SID tokens
            inputs_embeds: optional pre-computed encoder input embeddings
            encoder_outputs: optional pre-computed encoder outputs

        Returns:
            dict with keys: 'loss', 'sid_loss', 'codebook_loss',
                'cumulative_residual_loss', 'logits', 'predicted_embedding'
        """
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        n_codebook = labels_sids.shape[1]
        n_sem = self.n_semantic_codebook

        # ---- Encode the item sequence ----
        if encoder_outputs is None:
            if inputs_embeds is not None:
                encoder_outputs = self.encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            else:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

        # ---- Optional: predicted embedding for dense retrieval ----
        if self.flag_use_output_embedding:
            item_seq_len = attention_mask.sum(-1)
            self.predicted_embedding = self.gather_indexes(
                encoder_outputs.last_hidden_state, item_seq_len - 1
            )
        else:
            self.predicted_embedding = None

        encoder_hidden = encoder_outputs.last_hidden_state

        # ── Compute scheduled soft-label temperature ──
        # Linear decay from soft_label_temperature (initial) to soft_label_temp_min
        # over soft_label_temp_decay_steps forward calls.
        soft_label_temp_progress = 0.0
        soft_label_temp_step = self._soft_label_temp_step
        if self.soft_label_temp_min is not None and self.soft_label_K > 0:
            progress = min(
                self._soft_label_temp_step / self.soft_label_temp_decay_steps, 1.0
            )
            soft_label_temp_progress = progress
            current_soft_label_temp = (
                self.soft_label_temperature
                - (self.soft_label_temperature - self.soft_label_temp_min) * progress
            )
            self._soft_label_temp_step += 1
        else:
            current_soft_label_temp = self.soft_label_temperature

        # ── Compute per-position codebook loss weight decay (cosine) ──
        # alpha_k = 0.5 * (1 + cos(pi * step / decay_steps_k))
        # Decays from 1.0 to 0.0 over decay_steps_k forward calls.
        # Positions without decay (None) keep alpha = 1.0.
        codebook_weight_decay_step = self._codebook_weight_decay_step
        per_position_alphas = []  # alpha_k for each codebook level k
        for k in range(n_sem):
            decay_steps_k = (
                self.codebook_loss_weight_decay_steps[k]
                if self.codebook_loss_weight_decay_steps is not None
                and k < len(self.codebook_loss_weight_decay_steps)
                and self.codebook_loss_weight_decay_steps[k] is not None
                else None
            )
            if decay_steps_k is not None and decay_steps_k > 0:
                progress_k = min(codebook_weight_decay_step / decay_steps_k, 1.0)
                alpha_k = 0.5 * (1.0 + math.cos(math.pi * progress_k))
            else:
                alpha_k = 1.0
            per_position_alphas.append(alpha_k)
        if any(a < 1.0 for a in per_position_alphas):
            self._codebook_weight_decay_step += 1

        # ---- Precompute cumulative codebook sums for residual loss ----
        cumulative_residual_loss = 0.0
        num_cumul_steps = 0
        if self.cumulative_residual_loss_weight > 0:
            gt_cb_embs = []
            for t in range(n_sem):
                cb_idx_t = self._get_codebook_idx(labels_sids[:, t], t)
                gt_cb_embs.append(
                    self.codebook_embs[t][cb_idx_t].float()
                )  # [B, latent_size]
            # Right-to-left cumulative sum:
            # cumsum_gt[k] = codebook[k][sid_k] + codebook[k+1][sid_{k+1}] + ...
            cumsum_gt = [gt_cb_embs[-1]]
            for t in range(n_sem - 2, -1, -1):
                cumsum_gt.insert(0, gt_cb_embs[t] + cumsum_gt[0])

        # ---- Decode step-by-step (standard autoregressive) ----
        past_key_values = None
        all_ntp_logits = []
        sid_loss = 0.0
        codebook_loss = 0.0
        num_sid_steps = 0
        num_codebook_steps = 0

        for k in range(n_codebook):
            # === Decoder input ===
            if k == 0:
                dec_input_ids = torch.full(
                    (B, 1),
                    self.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                decoder_out = self.decoder(
                    input_ids=dec_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Teacher forcing: use ground truth sid_{k-1}
                dec_input_ids = labels_sids[:, k - 1 : k]  # [B, 1]
                decoder_out = self.decoder(
                    input_ids=dec_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            hidden = decoder_out.last_hidden_state[:, -1, :]  # [B, d_model]
            past_key_values = decoder_out.past_key_values

            # === NTP logits for sid_k prediction ===
            logits = self.lm_head(hidden).float()  # [B, vocab_size]
            all_ntp_logits.append(logits)

            # NTP loss
            target_k = labels_sids[:, k]  # [B]
            sid_loss += F.cross_entropy(logits, target_k, reduction="sum") / B
            num_sid_steps += 1

            # === Auxiliary losses (only for positions within codebook range) ===
            if k < n_sem:
                # Map hidden to latent space via output_adapter[k]
                h_latent = self.output_adapters[k](hidden.float())  # [B, latent_size]

                # === Auxiliary codebook loss for level k ===
                alpha_k = per_position_alphas[k]
                loss_k = self._compute_code_loss(
                    h_latent, k, labels_sids, B, device,
                    soft_label_temp=current_soft_label_temp,
                )
                codebook_loss += loss_k * alpha_k
                num_codebook_steps += 1

                # === Auxiliary cumulative residual loss for level k ===
                if self.cumulative_residual_loss_weight > 0 and k < n_sem - 1:
                    # Target: sum of codebook entries from k+1 to n-1
                    target_cumul = cumsum_gt[k + 1]  # [B, latent_size]
                    if self.cumulative_residual_loss_type == "cosine":
                        cos_sim = F.cosine_similarity(h_latent, target_cumul, dim=-1)
                        cumulative_residual_loss += (1.0 - cos_sim).sum() / B
                    elif self.cumulative_residual_loss_type == "ce":
                        sim_matrix = h_latent @ target_cumul.T  # [B, B]
                        sim_matrix = sim_matrix / self.cumulative_residual_loss_temperature
                        labels_ce = torch.arange(B, device=device)
                        cumulative_residual_loss += F.cross_entropy(
                            sim_matrix, labels_ce, reduction="mean"
                        )
                    else:
                        # "mse"
                        cumulative_residual_loss += F.mse_loss(
                            h_latent, target_cumul, reduction="sum"
                        ) / B
                    num_cumul_steps += 1

        # ---- Combine losses ----
        sid_loss = sid_loss / num_sid_steps if num_sid_steps > 0 else 0.0
        codebook_loss = (
            codebook_loss / num_codebook_steps if num_codebook_steps > 0 else 0.0
        )
        cumulative_residual_loss = (
            cumulative_residual_loss / num_cumul_steps
            if num_cumul_steps > 0
            else 0.0
        )
        total_loss = (
            sid_loss
            + self.codebook_loss_weight * codebook_loss
            + self.cumulative_residual_loss_weight * cumulative_residual_loss
        )

        # Stack NTP logits: [B, n_codebook, vocab_size]
        stacked_logits = torch.stack(all_ntp_logits, dim=1)

        # ── Debug log: codebook weight decay ──
        if self.codebook_loss_weight_decay_steps is not None:
            if not hasattr(self, '_cb_weight_decay_log_step'):
                self._cb_weight_decay_log_step = 0
                self._cb_weight_decay_log_interval = 50
            if self._cb_weight_decay_log_step % self._cb_weight_decay_log_interval == 0:
                alpha_str = ", ".join(
                    f"k{k}:α={per_position_alphas[k]:.3f}" for k in range(n_sem)
                )
                print(
                    f"\n[CBWeightDecay] step={codebook_weight_decay_step} {alpha_str}"
                )
            self._cb_weight_decay_log_step += 1

        return {
            "loss": total_loss,
            "sid_loss": sid_loss,
            "codebook_loss": codebook_loss,
            "cumulative_residual_loss": cumulative_residual_loss,
            "logits": stacked_logits,
            "predicted_embedding": self.predicted_embedding,
            "current_soft_label_temp": current_soft_label_temp,
            "soft_label_temp_progress": soft_label_temp_progress,
            "soft_label_temp_step": soft_label_temp_step,
            "codebook_weight_decay_step": codebook_weight_decay_step,
            "per_position_alphas": per_position_alphas,
        }

    @torch.no_grad()
    def generate_residual(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        max_new_tokens=4,
        num_beams=1,
        num_return_sequences=1,
        **kwargs,
    ):
        """
        Greedy/beam-search generation (standard autoregressive, no residual).

        Since residual information is only used as auxiliary loss during
        training and not injected into the decoder input, generation is
        identical to base TIGER — standard beam search over SID tokens.

        Args:
            input_ids: [B, enc_seq_len]
            attention_mask: [B, enc_seq_len]
            inputs_embeds: optional pre-computed encoder inputs_embeds
            max_new_tokens: number of SID tokens to generate (= n_codebook)
            num_beams: number of beams for beam search
            num_return_sequences: number of returned sequences

        Returns:
            output_sids: [B * num_return_sequences, max_new_tokens]
        """
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        n_codebook_total = max_new_tokens
        n_beams = num_beams
        batch_beam_size = B * n_beams

        # ── Encode ──
        if inputs_embeds is not None:
            encoder_outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        encoder_hidden = encoder_outputs.last_hidden_state  # [B, enc_len, d_model]

        if self.flag_use_output_embedding:
            item_seq_len = attention_mask.sum(-1)
            self.predicted_embedding = self.gather_indexes(
                encoder_outputs.last_hidden_state, item_seq_len - 1
            )
        else:
            self.predicted_embedding = None

        # Expand for beam search
        encoder_hidden = encoder_hidden.repeat_interleave(n_beams, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(n_beams, dim=0)

        # ── Beam state ──
        beam_scores = torch.zeros((B, n_beams), device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.reshape(-1)  # [B * n_beams]

        past_key_values = None
        generated_sids = []

        # ── Generate step by step ──
        for k in range(n_codebook_total):
            if k == 0:
                dec_input_ids = torch.full(
                    (batch_beam_size, 1),
                    self.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                decoder_out = self.decoder(
                    input_ids=dec_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                dec_input_ids = generated_sids[-1]  # [B * n_beams, 1]
                decoder_out = self.decoder(
                    input_ids=dec_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            hidden = decoder_out.last_hidden_state[:, -1, :]  # [B*n_beams, d_model]
            kv_after_step = decoder_out.past_key_values

            # ── Beam search for sid_k ──
            logits = self.lm_head(hidden)  # [B*n_beams, V]
            next_scores = F.log_softmax(logits, dim=-1) + beam_scores[:, None]
            next_scores = next_scores.reshape(B, n_beams * logits.shape[-1])

            vocab_size = logits.shape[-1]
            next_scores, next_indices = torch.topk(
                next_scores, min(2 * n_beams, next_scores.shape[-1]), dim=1
            )

            next_beam_idx = torch.div(
                next_indices, vocab_size, rounding_mode="floor"
            )
            next_tokens = next_indices % vocab_size

            chosen_scores = next_scores[:, :n_beams]
            chosen_tokens = next_tokens[:, :n_beams]
            chosen_beam_idx = next_beam_idx[:, :n_beams]

            beam_scores = chosen_scores.reshape(-1)

            # Reorder indices
            chosen_beam_flat = chosen_beam_idx + torch.arange(
                B, device=device
            )[:, None] * n_beams
            reorder_indices = chosen_beam_flat.reshape(-1)

            # Reorder KV cache
            past_key_values = self._reorder_cache(kv_after_step, reorder_indices)

            # Store predicted sid_k
            sid_k = chosen_tokens.reshape(-1, 1)  # [B*n_beams, 1]
            generated_sids.append(sid_k)

            # Reorder previously generated sids
            for i in range(len(generated_sids) - 1):
                generated_sids[i] = generated_sids[i][reorder_indices]

        # ── Collect output ──
        output = torch.cat(generated_sids, dim=-1)  # [B*n_beams, n_codebook_total]
        output = output.reshape(B, n_beams, n_codebook_total)
        output = output[:, :num_return_sequences, :]
        output = output.reshape(B * num_return_sequences, n_codebook_total)

        return output

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder past_key_values for beam search."""
        reordered = ()
        for layer_past in past_key_values:
            reordered += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered