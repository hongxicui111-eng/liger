"""
TIGER_Residual: TIGER model with residual information from RQ-VAE codebooks
interleaved in the decoder for improved semantic ID generation.

The generation process:
  Standard: <BOS> -> resid_0 -> sid_0 -> resid_1 -> sid_1 -> ... -> sid_{n-2} -> sid_{n-1}
  With flag_separate_bos_representation: <BOS> -> <SID_START> -> sid_0 -> resid_0 -> sid_1 -> resid_1 -> ...

Where:
  - Standard positions (BOS, SID_START, sid_k):  input = token embedding, output = NTP logits -> NTP loss
  - Residual positions (resid_k):     input = residual vector,   output = hidden -> MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from .tiger import TIGER


class TIGER_Residual(TIGER):
    """
    TIGER with residual decoder. Adds codebook embeddings from RQ-VAE
    and interleaves residual tokens between standard SID tokens during decoding.

    Args:
        config: T5Config
        n_semantic_codebook: number of RQ-VAE semantic codebook levels
        max_items_per_seq: max items per sequence
        flag_use_output_embedding: whether to output item embedding
        flag_use_learnable_text_embed: whether to use learnable text embedding
        embedding_head_dict: config for embedding head
        rqvae_codebook_weights: list of [codebook_size, latent_size] tensors (from RQ-VAE)
        codebook_size: size of each codebook
        latent_size: RQ-VAE latent dimension (codebook embedding dim)
        codebook_loss_weight: weight for codebook selection CE loss term
        num_residual_levels: how many residual positions to use (default: n_semantic_codebook - 1)
        soft_label_K: number of nearest codebook entries used as positive samples for
            soft-label codebook loss. 0 = hard label (original CE). Default: 0.
        soft_label_temperature: temperature for softmax normalization of distance-based
            soft labels. Higher = more uniform across K entries, lower = sharper (ground
            truth dominates). Default: 1.0.
        cumulative_residual_loss_weight: weight for cumulative residual alignment loss.
            At each residual position k, constrains output_adapter[k+1](hidden_resid_k)
            to be close to sum(codebook_emb[t][sid_t] for t in [k+1, ..., n-1]).
            This enforces that the residual representation captures the *full remaining*
            codebook content, not just the next codebook index. Default: 0.0 (disabled).
        cumulative_residual_loss_type: "mse" for L2 regression or "cosine" for cosine
            similarity loss. Default: "mse".
        resid_nontf_ratio: probability of using the model's predicted sid_k (instead of
            ground truth sid_k) for codebook index selection at each residual step.
            Per-step, per-sample: at each residual step k, each sample independently
            has this probability of using predicted sid_k instead of GT sid_k to
            compute the residual. This addresses the train-test mismatch where
            training uses GT sid_k but inference uses predicted sid_k.
            0.0 = always use GT (full teacher forcing, original behavior).
            Typical values: 0.1~0.3 for gradual introduction.
            Default: 0.0.
        ntp_nontf_ratio: probability of using the model's predicted sid_{k-1} (instead
            of ground truth sid_{k-1}) as decoder input at each NTP step (k > 0).
            This is standard scheduled sampling for the NTP decoder path.
            More aggressive than resid_nontf_ratio — changes the decoder's context,
            potentially destabilizing training. Recommended to start with a small
            value (0.05~0.1) if used.
            0.0 = always use GT (full teacher forcing).
            Default: 0.0.
        flag_separate_bos_representation: if True, the <BOS> token only generates
            an overall object representation (no SID prediction), and a separate
            <SID_START> token (sid_start_token_id) triggers the first SID (sid_0)
            generation. This decouples "understanding the object" from "generating
            SIDs", potentially allowing the BOS hidden state to better capture
            holistic object information.
            Default: False (backward compatible).
        sid_start_token_id: the special token ID used as the "begin SID generation"
            delimiter. Must be set when flag_separate_bos_representation=True.
            This token is inserted right after <BOS> to trigger sid_0 generation.
            Default: None.
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
        num_residual_levels: int = None,
        soft_label_K: int = 0,
        soft_label_temperature: float = 1.0,
        cumulative_residual_loss_weight: float = 0.0,
        cumulative_residual_loss_type: str = "mse",
        resid_nontf_ratio: float = 0.0,
        ntp_nontf_ratio: float = 0.0,
        flag_separate_bos_representation: bool = False,
        sid_start_token_id: int = None,
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
        self.num_residual_levels = (
            num_residual_levels
            if num_residual_levels is not None
            else n_semantic_codebook - 1
        )
        self.soft_label_K = soft_label_K
        self.soft_label_temperature = soft_label_temperature
        self.cumulative_residual_loss_weight = cumulative_residual_loss_weight
        self.cumulative_residual_loss_type = cumulative_residual_loss_type
        self.resid_nontf_ratio = resid_nontf_ratio
        self.ntp_nontf_ratio = ntp_nontf_ratio
        self.flag_separate_bos_representation = flag_separate_bos_representation
        self.sid_start_token_id = sid_start_token_id
        # Debug counter: log soft-label distribution sharpness every N forward passes
        self._soft_label_log_step = 0
        self._soft_label_log_interval = 20  # print every 20 forward calls
        self._nontf_log_step = 0
        self._nontf_log_interval = 50  # log non-TF stats every 50 forward calls

        # Register codebook embeddings (frozen, from RQ-VAE)
        if rqvae_codebook_weights is not None:
            for i, cb_weight in enumerate(rqvae_codebook_weights):
                # cb_weight shape: [codebook_size (+1 for padding), latent_size]
                # Remove padding row if present, and keep only the first codebook_size entries
                if cb_weight.shape[0] > codebook_size:
                    cb_weight = cb_weight[:codebook_size]
                self.register_buffer(
                    f"codebook_emb_{i}",
                    cb_weight.detach().clone(),
                )
        else:
            # Initialize randomly if not provided (will be loaded later)
            for i in range(n_semantic_codebook):
                self.register_buffer(
                    f"codebook_emb_{i}",
                    torch.randn(codebook_size, latent_size) * 0.01,
                )

        # ── Per-position adapters (no parameter sharing across levels) ──
        # Different residual levels have different semantics, so each position
        # needs its own adapter to map between backbone space and codebook space.
        # Residuals are computed in codebook latent space (like RQ-VAE), so:
        #   - output_adapters[k]: d_model → latent  (project hidden to codebook space for residual & MSE)
        #   - input_adapters[k]:  latent → d_model  (project residual back to backbone space for decoder)

        # Input adapters (k-1): map residual vector from codebook latent space
        # to backbone's d_model space before feeding into the decoder.
        # Each residual level gets its own adapter.
        self.input_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_size, self.d_model, bias=False),
                nn.LayerNorm(self.d_model, eps=config.layer_norm_epsilon),
            )
            for _ in range(self.num_residual_levels)
        ])

        # Output adapters (k): map decoder hidden vector to codebook latent space.
        # Used for: (1) projecting hidden_k to latent for residual computation,
        #            (2) projecting hidden_resid to latent for MSE alignment.
        # All k adapters are utilized — adapter[k] at residual position k,
        # adapter[k+1] at MSE position k.
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
        return [getattr(self, f"codebook_emb_{i}") for i in range(self.n_semantic_codebook)]

    def load_codebooks_from_rqvae(self, rqvae_model):
        """Load codebook weights from a trained RQ-VAE model."""
        for i, codebook in enumerate(rqvae_model.quantizer.codebooks):
            if i >= self.n_semantic_codebook:
                break
            # VQEmbedding has weight of shape [n_embed+1, embed_dim] with padding
            weight = codebook.weight[:-1].detach().clone()  # [codebook_size, latent_size]
            buffer = getattr(self, f"codebook_emb_{i}")
            buffer.copy_(weight)

    def _get_codebook_idx(self, sid_tokens, level):
        """Convert SID token to codebook index (0 to codebook_size-1)."""
        # SID token = original_code + codebook_size * level + 1
        cb_idx = sid_tokens - 1 - level * self.codebook_size
        return cb_idx.clamp(0, self.codebook_size - 1)

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
        Forward pass with residual interleaving.

        Decodes step-by-step: BOS -> resid_0 -> sid_0 -> resid_1 -> sid_1 -> ...

        Args:
            input_ids: [B, enc_seq_len] encoder input token IDs
            attention_mask: [B, enc_seq_len] encoder attention mask
            labels_sids: [B, n_codebook] ground truth SID tokens for teacher forcing
            inputs_embeds: optional pre-computed encoder input embeddings
            encoder_outputs: optional pre-computed encoder outputs

        Returns:
            dict with keys: 'loss', 'sid_loss', 'codebook_loss', 'logits', 'predicted_embedding'
        """
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        n_codebook = labels_sids.shape[1]
        n_sem = self.n_semantic_codebook
        num_resid = self.num_residual_levels

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

        # ---- Pre-compute cumulative codebook sums for cumulative residual loss ----
        # cumsum_gt[t] = sum(codebook_emb[j][sid_j] for j in [t, ..., n_sem-1])
        # Only iterate over registered codebook levels (n_sem), since the cumulative
        # target is only used at residual positions k < num_resid (≤ n_sem-1).
        # At residual position k, the target is cumsum_gt[k+1] — the remaining
        # codebook content from level k+1 to n_sem-1 that needs to be captured.
        cumulative_residual_loss = 0.0
        num_cumul_steps = 0
        if self.cumulative_residual_loss_weight > 0:
            gt_cb_embs = []
            for t in range(n_sem):  # only registered codebooks, not all n_codebook levels
                cb_idx_t = self._get_codebook_idx(labels_sids[:, t], t)
                gt_cb_embs.append(self.codebook_embs[t][cb_idx_t].float())  # [B, latent_size]
            # Right-to-left cumulative sum
            cumsum_gt = [gt_cb_embs[-1]]
            for t in range(n_sem - 2, -1, -1):
                cumsum_gt.insert(0, gt_cb_embs[t] + cumsum_gt[0])
            # cumsum_gt[t]: [B, latent_size]

        # ---- Decode step-by-step with residual interleaving ----
        past_key_values = None
        all_ntp_logits = []  # [n_codebook] list of [B, V]
        predicted_sids = []  # [n_codebook] list of [B] — model's own predictions for non-TF
        sid_loss = 0.0
        codebook_loss = 0.0
        num_sid_steps = 0
        num_codebook_steps = 0

        # ── Non-teacher-forcing statistics ──
        nontf_resid_count = 0  # how many residual steps used predicted sid
        nontf_ntp_count = 0    # how many NTP steps used predicted input
        nontf_resid_total = 0  # total residual steps eligible for non-TF
        nontf_ntp_total = 0    # total NTP steps eligible for non-TF

        # ── BOS-only step: build holistic object representation (no SID decoding) ──
        if self.flag_separate_bos_representation:
            dec_input_bos = torch.full(
                (B, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_out_bos = self.decoder(
                input_ids=dec_input_bos,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out_bos.past_key_values
            # No NTP loss, no logits — BOS only builds the representation

        for k in range(n_codebook):
            # === Standard position: predict sid_k via NTP ===
            if k == 0:
                if self.flag_separate_bos_representation:
                    dec_input_ids = torch.full(
                        (B, 1),
                        self.sid_start_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    dec_input_ids = torch.full(
                        (B, 1),
                        self.config.decoder_start_token_id,
                        dtype=torch.long,
                        device=device,
                    )
            else:
                # ── NTP non-teacher-forcing (scheduled sampling) ──
                # At step k > 0, the decoder normally receives GT sid_{k-1} as input.
                # With ntp_nontf_ratio > 0, some samples use the model's predicted
                # sid_{k-1} instead. This is standard scheduled sampling — makes the
                # model robust to its own prediction errors during inference.
                if self.ntp_nontf_ratio > 0 and len(predicted_sids) > 0:
                    ntp_nontf_mask = torch.rand(B, device=device) < self.ntp_nontf_ratio  # [B]
                    predicted_sid_prev = predicted_sids[-1]  # [B], predicted sid_{k-1}
                    # Blend: masked samples use predicted, rest use GT
                    dec_input_ids = torch.where(
                        ntp_nontf_mask.unsqueeze(-1),
                        predicted_sid_prev.unsqueeze(-1),  # [B, 1]
                        labels_sids[:, k - 1 : k],          # [B, 1]
                    )  # [B, 1]
                    nontf_ntp_count += ntp_nontf_mask.sum().item()
                    nontf_ntp_total += B
                else:
                    # Teacher forcing: use ground truth sid_{k-1} as input
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

            # NTP logits for this step — cast to FP32 for numerical stability under AMP
            logits = self.lm_head(hidden).float()  # [B, vocab_size]
            all_ntp_logits.append(logits)

            # Store predicted sid_k for potential non-TF use in subsequent steps
            predicted_sid_k = logits.argmax(dim=-1)  # [B]
            predicted_sids.append(predicted_sid_k)

            # NTP loss for sid_k — compute in FP32
            # NOTE: NTP loss always uses GT labels as target, regardless of
            # whether the decoder input was TF or non-TF. The loss measures
            # "can the model predict the correct next token", which is the
            # fundamental objective. Non-TF inputs just change the context.
            target_k = labels_sids[:, k]  # [B]
            sid_loss += F.cross_entropy(logits, target_k, reduction="sum") / B
            num_sid_steps += 1
        
            # === Residual position (if we need one after this level) ===
            # We add a residual position for levels 0 through num_resid-1
            if k < num_resid and k + 1 < n_codebook:
                # ── Residual non-teacher-forcing ──
                # Normally, residual = adapter(hidden_k) - codebook_emb[k][GT_sid_k].
                # With resid_nontf_ratio > 0, some samples use the model's predicted
                # sid_k instead of GT sid_k for codebook index selection. This
                # directly addresses the train-test mismatch:
                #   - Training: residual uses GT sid_k (always correct)
                #   - Inference: residual uses predicted sid_k (may be wrong)
                # By mixing predicted/GT during training, the model learns to
                # handle both "correct" and "incorrect" residuals in the KV cache.
                if self.resid_nontf_ratio > 0:
                    resid_nontf_mask_k = torch.rand(B, device=device) < self.resid_nontf_ratio  # [B]
                    # Blend: masked samples use predicted sid_k, rest use GT sid_k
                    blend_sid_k = torch.where(
                        resid_nontf_mask_k,
                        predicted_sid_k,      # [B], model's own prediction
                        labels_sids[:, k],    # [B], ground truth
                    )  # [B]
                    cb_idx_k = self._get_codebook_idx(blend_sid_k, k)
                    nontf_resid_count += resid_nontf_mask_k.sum().item()
                    nontf_resid_total += B
                else:
                    # Original behavior: always use GT sid_k
                    cb_idx_k = self._get_codebook_idx(labels_sids[:, k], k)
                cb_emb = self.codebook_embs[k][cb_idx_k]  # [B, latent_size]

                # Compute residual in codebook latent space (like RQ-VAE):
                #   h_latent = output_adapter[k](h_k)  → latent space
                #   residual = h_latent - embed_k(sid_k)    → latent space
                # Cast to FP32 to prevent overflow/NaN under AMP FP16 autocast
                h_latent = self.output_adapters[k](hidden.float())  # [B, latent_size]
                residual = (h_latent - cb_emb.float())  # [B, latent_size]

                # Clamp residual magnitude to prevent extreme values destabilizing decoder
                residual_max = self.latent_size * 2.0  # heuristic bound in latent space
                residual = residual.clamp(-residual_max, residual_max)

                # Map residual from latent space to backbone d_model space via input adapter,
                # then feed into decoder. Each level has its own adapter (no sharing).
                residual_adapted = self.input_adapters[k](residual.to(hidden.dtype))  # [B, d_model]
                residual_input = residual_adapted.unsqueeze(1)  # [B, 1, d_model]

                decoder_out_resid = self.decoder(
                    inputs_embeds=residual_input,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_resid = decoder_out_resid.last_hidden_state[:, -1, :]  # [B, d_model]

                # NaN guard: if residual decoder produced NaN, DON'T contaminate KV cache
                # Fall back to the NTP step's past_key_values instead
                if not torch.isnan(hidden_resid).any():
                    past_key_values = decoder_out_resid.past_key_values
                # else: keep past_key_values from the NTP step (no contamination)

                # Codebook selection CE loss: project to latent space and
                # classify within codebook[k+1] — mirrors VQ argmin selection,
                # not regression to a specific vector.
                hidden_resid_latent = self.output_adapters[k + 1](hidden_resid.float())  # [B, latent_size]

                # Compute similarity logits against all codebook[k+1] entries
                cb_weights = self.codebook_embs[k + 1]  # [codebook_size, latent_size]
                logits_cb = hidden_resid_latent @ cb_weights.T.float()  # [B, codebook_size]

                # Target: correct codebook index for level k+1
                cb_idx_next = self._get_codebook_idx(labels_sids[:, k + 1], k + 1)  # [B]

                if self.soft_label_K > 0:
                    # ── Soft-label codebook loss ──────────────────────────────
                    # Instead of hard CE (ground truth = 1, rest = 0), we use
                    # distance-based soft labels: the K nearest codebook entries
                    # to the ground truth receive softmax(-dist/temp) as their
                    # label probability, providing smooth gradient signals to
                    # nearby entries while still emphasizing the ground truth.
                    #
                    # Algorithm:
                    #   1. Compute L2 distance from each codebook entry to ground truth
                    #   2. Select top-K nearest entries (including ground truth itself)
                    #   3. Apply softmax(-dist/temp) over those K entries → soft_labels
                    #   4. Compute cross-entropy: -sum(soft_labels * log_softmax(logits_cb))
                    # ────────────────────────────────────────────────────────────

                    # Ground truth embedding in codebook[k+1]
                    gt_emb = cb_weights[cb_idx_next].float()  # [B, latent_size]

                    # Squared L2 distance: ||cb_weights[j] - gt_emb[b]||^2
                    # cb_weights: [codebook_size, latent_size], gt_emb: [B, latent_size]
                    dist_sq = ((cb_weights.float().unsqueeze(0) - gt_emb.unsqueeze(1)) ** 2).sum(-1)  # [B, codebook_size]

                    # Select top-K nearest entries (smallest distance)
                    K = min(self.soft_label_K, self.codebook_size)
                    _, topk_indices = torch.topk(dist_sq, k=K, dim=-1, largest=False)  # [B, K]

                    # Extract distances for the top-K entries, use L2 distance (sqrt)
                    # for better softmax scaling (squared distances are too extreme)
                    topk_dist_sq = dist_sq.gather(1, topk_indices)  # [B, K]
                    topk_dist = torch.sqrt(topk_dist_sq.clamp(min=1e-8))  # [B, K]

                    # Softmax on negative distances / temperature:
                    #   closer entries → higher probability
                    #   ground truth (dist=0) → highest probability
                    neg_scaled_dist = -topk_dist / self.soft_label_temperature  # [B, K]
                    soft_probs_topk = F.softmax(neg_scaled_dist, dim=-1)  # [B, K], sums to 1

                    # Build full soft label distribution over entire codebook
                    soft_labels = torch.zeros(B, self.codebook_size, device=device, dtype=torch.float32)
                    soft_labels.scatter_(1, topk_indices, soft_probs_topk)  # [B, codebook_size]

                    # Cross-entropy with soft targets:
                    #   loss = -sum(soft_labels * log_softmax(logits_cb))
                    # This generalizes hard CE (when soft_labels is one-hot, it reduces to CE)
                    log_probs = F.log_softmax(logits_cb, dim=-1)  # [B, codebook_size]
                    codebook_loss += -(soft_labels * log_probs).sum(dim=-1).mean()

                    # ── Debug log: check soft-label sharpness ────────────────
                    if self._soft_label_log_step % self._soft_label_log_interval == 0:
                        n_show = min(10, B)
                        max_prob = soft_probs_topk[:n_show, 0].detach()   # rank-0 = GT (dist=0)
                        min_prob = soft_probs_topk[:n_show, -1].detach()  # rank-(K-1) = farthest
                        print(
                            f"\n[SoftLabel] step={self._soft_label_log_step} "
                            f"level k→k+1: {k}→{k+1}  temp={self.soft_label_temperature}  K={K}"
                        )
                        print(f"  {'sample':>6}  {'gt_idx':>6}  {'max_p':>7}  {'min_p':>7}  "
                              f"{'topK_dists (L2)':>40s}")
                        for i in range(n_show):
                            dists_str = "  ".join(f"{topk_dist[i, j].item():.2f}" for j in range(K))
                            print(
                                f"  {i:>6}  {topk_indices[i, 0].item():>6}  "
                                f"{max_prob[i].item():>7.4f}  {min_prob[i].item():>7.4f}  "
                                f"{dists_str:>40s}"
                            )
                        print()  # blank line after the block
                    self._soft_label_log_step += 1
                    # ───────────────────────────────────────────────────────────
                else:
                    # ── Hard-label: standard CE (original behavior) ──
                    codebook_loss += F.cross_entropy(logits_cb, cb_idx_next, reduction="sum") / B
                num_codebook_steps += 1

                # ---- Cumulative residual constraint (optional) ----
                # Constrain output_adapter[k+1](hidden_resid_k) to be close to the
                # cumulative sum of all remaining codebook embeddings. This enforces
                # that the residual representation encodes the *full* remaining
                # content, not just the next codebook index. Complementary to the
                # classification-based codebook loss above.
                if self.cumulative_residual_loss_weight > 0:
                    target_cumul = cumsum_gt[k + 1]  # [B, latent_size]: sum of codes k+1..n-1
                    if self.cumulative_residual_loss_type == "cosine":
                        cos_sim = F.cosine_similarity(
                            hidden_resid_latent, target_cumul, dim=-1
                        )
                        cumulative_residual_loss += (1.0 - cos_sim).sum() / B
                    else:
                        cumulative_residual_loss += F.mse_loss(
                            hidden_resid_latent, target_cumul, reduction="sum"
                        ) / B
                    num_cumul_steps += 1

        # ---- Combine losses (normalize to mean per step) ----
        sid_loss = sid_loss / num_sid_steps if num_sid_steps > 0 else 0.0
        codebook_loss = codebook_loss / num_codebook_steps if num_codebook_steps > 0 else 0.0
        cumulative_residual_loss = cumulative_residual_loss / num_cumul_steps if num_cumul_steps > 0 else 0.0
        total_loss = (
            sid_loss
            + self.codebook_loss_weight * codebook_loss
            + self.cumulative_residual_loss_weight * cumulative_residual_loss
        )

        # Stack NTP logits: [B, n_codebook, vocab_size]
        stacked_logits = torch.stack(all_ntp_logits, dim=1)

        # ── Debug log: non-TF statistics ──
        if (self.resid_nontf_ratio > 0 or self.ntp_nontf_ratio > 0):
            if self._nontf_log_step % self._nontf_log_interval == 0:
                resid_pct = (nontf_resid_count / nontf_resid_total * 100) if nontf_resid_total > 0 else 0.0
                ntp_pct = (nontf_ntp_count / nontf_ntp_total * 100) if nontf_ntp_total > 0 else 0.0
                print(
                    f"\n[NonTF] step={self._nontf_log_step} "
                    f"resid_nontf_ratio={self.resid_nontf_ratio:.2f} → "
                    f"{nontf_resid_count}/{nontf_resid_total} ({resid_pct:.1f}% used predicted) "
                    f"ntp_nontf_ratio={self.ntp_nontf_ratio:.2f} → "
                    f"{nontf_ntp_count}/{nontf_ntp_total} ({ntp_pct:.1f}% used predicted)"
                )
            self._nontf_log_step += 1

        return {
            "loss": total_loss,
            "sid_loss": sid_loss,
            "codebook_loss": codebook_loss,
            "cumulative_residual_loss": cumulative_residual_loss,
            "logits": stacked_logits,
            "predicted_embedding": self.predicted_embedding,
            "nontf_resid_pct": nontf_resid_count / nontf_resid_total if nontf_resid_total > 0 else 0.0,
            "nontf_ntp_pct": nontf_ntp_count / nontf_ntp_total if nontf_ntp_total > 0 else 0.0,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder past_key_values for beam search.

        T5's past_key_values is a tuple of tuples:
            ((key_0, value_0), (key_1, value_1), ..., (key_L, value_L))
        Each key/value has shape [batch, num_heads, seq_len, head_dim].
        """
        reordered = ()
        for layer_past in past_key_values:
            reordered += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered

    @torch.no_grad()
    def generate_residual(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        max_new_tokens=4,
        num_beams=1,
        num_return_sequences=1,
        resid_beams=None,
        resid_score_weight=1.0,
        resid_beam_mode="pruning",
        **kwargs,
    ):
        """
        Beam-search generation with residual interleaving.

        Uses beam search at SID prediction positions (NTP) and beam pruning
        at residual positions. At residual positions, the codebook[k+1]
        classification logits (trained during forward_residual but previously
        unused in generation) are used to prune beams — each beam picks its
        best internal codebook entry, then beams are re-ranked by combined
        score and the top n_beams survive.

        When resid_beams=1, residual positions use greedy pass-through
        (no pruning/expansion). When resid_beams>1, the behavior depends on
        resid_beam_mode:
          - "pruning" (default): each beam picks its best internal codebook
            entry, then beams compete for survival. At most one candidate per
            parent beam → guaranteed unique SID prefixes.
          - "expansion": expand all n_beams × codebook_size candidates,
            select top n_beams globally. Multiple candidates may come from
            the same parent beam (may produce duplicate SID prefixes).

        Args:
            input_ids: [B, enc_seq_len]
            attention_mask: [B, enc_seq_len]
            inputs_embeds: optional pre-computed encoder inputs_embeds
            max_new_tokens: number of SID tokens to generate (= n_codebook)
            num_beams: number of beams for beam search at SID positions
            num_return_sequences: number of returned sequences (≤ num_beams)
            resid_beams: controls beam pruning at residual positions.
                None or >1: prune beams using codebook[k+1] logits (each beam
                picks its best codebook entry, then beams compete for survival).
                1: greedy pass-through at residual positions (no pruning).
            resid_score_weight: weight for codebook logit scores at residual
                positions when combining with beam scores. Default: 1.0.

        Returns:
            output_sids: [B * num_return_sequences, max_new_tokens]
        """
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        n_codebook_total = max_new_tokens
        n_beams = num_beams
        batch_beam_size = B * n_beams
        n_resid_beams = resid_beams if resid_beams is not None else n_beams
        resid_weight = resid_score_weight
        if resid_beam_mode not in ("pruning", "expansion"):
            raise ValueError(
                f"resid_beam_mode must be 'pruning' or 'expansion', got '{resid_beam_mode}'"
            )

        # ── Encode ──
        if inputs_embeds is not None:
            enr_outputs = self.encoder(
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

        # Expand encoder outputs for beam search: [B, ...] → [B * n_beams, ...]
        encoder_hidden = encoder_hidden.repeat_interleave(n_beams, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(n_beams, dim=0)

        # ── Beam state ──
        beam_scores = torch.zeros((B, n_beams), device=device)
        beam_scores[:, 1:] = -1e9  # only first beam is active initially
        beam_scores = beam_scores.reshape(-1)  # [B * n_beams]

        past_key_values = None
        generated_sids = []  # list of [B * n_beams, 1] tensors

        # ── BOS-only step: build holistic object representation (no SID decoding) ──
        if self.flag_separate_bos_representation:
            dec_input_bos = torch.full(
                (batch_beam_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_out_bos = self.decoder(
                input_ids=dec_input_bos,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out_bos.past_key_values
            # No beam search on BOS step — only builds KV cache representation

        # ── Generate step by step ──
        for k in range(n_codebook_total):
            # ---- Decoder input for this NTP step ----
            if k == 0:
                if self.flag_separate_bos_representation:
                    dec_input_ids = torch.full(
                        (batch_beam_size, 1),
                        self.sid_start_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    dec_input_ids = torch.full(
                        (batch_beam_size, 1),
                        self.config.decoder_start_token_id,
                        dtype=torch.long,
                        device=device,
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
            hidden = decoder_out.last_hidden_state[:, -1, :]  # [B * n_beams, d_model]
            kv_after_ntp = decoder_out.past_key_values

            # ---- Beam search for sid_k (NTP) ----
            logits = self.lm_head(hidden)  # [B * n_beams, V]
            next_scores = F.log_softmax(logits, dim=-1) + beam_scores[:, None]  # [B * n_beams, V]
            next_scores = next_scores.reshape(B, n_beams * logits.shape[-1])  # [B, n_beams * V]

            # Select top 2*n_beams candidates (then prune to n_beams)
            next_scores, next_indices = torch.topk(
                next_scores, min(2 * n_beams, next_scores.shape[-1]), dim=1
            )  # [B, 2*n_beams]

            # Reconstruct beam index and token
            next_beam_idx = torch.div(next_indices, logits.shape[-1], rounding_mode="floor")  # [B, 2*n_beams]
            next_tokens = next_indices % logits.shape[-1]  # [B, 2*n_beams]

            # Prune to n_beams final beams per batch
            chosen_scores = next_scores[:, :n_beams]  # [B, n_beams]
            chosen_tokens = next_tokens[:, :n_beams]  # [B, n_beams]
            chosen_beam_idx = next_beam_idx[:, :n_beams]  # [B, n_beams]

            beam_scores = chosen_scores.reshape(-1)  # [B * n_beams]

            # Build reorder indices for KV cache and generated_sids
            chosen_beam_flat = chosen_beam_idx + torch.arange(
                B, device=device
            )[:, None] * n_beams  # [B, n_beams], values in [0, B*n_beams)
            reorder_indices = chosen_beam_flat.reshape(-1)  # [B * n_beams]

            # Reorder KV cache
            past_key_values = self._reorder_cache(kv_after_ntp, reorder_indices)

            # Reorder hidden states (needed for residual computation)
            hidden = hidden[reorder_indices]  # [B * n_beams, d_model]

            # Store new sid_k
            sid_k = chosen_tokens.reshape(-1, 1)  # [B * n_beams, 1]
            generated_sids.append(sid_k)

            # Reorder previously generated sids
            for i in range(len(generated_sids) - 1):
                generated_sids[i] = generated_sids[i][reorder_indices]

            # ---- Residual position: beam expansion using codebook logits ----
            if k < self.num_residual_levels and k + 1 < n_codebook_total:
                cb_idx_k = self._get_codebook_idx(
                    sid_k.squeeze(-1), k
                )  # [B * n_beams]
                cb_emb = self.codebook_embs[k][cb_idx_k]  # [B * n_beams, latent_size]

                # Compute residual in codebook latent space (like RQ-VAE)
                h_latent = self.output_adapters[k](hidden)  # [B * n_beams, latent_size]
                residual = h_latent - cb_emb  # [B * n_beams, latent_size]

                # Map residual from latent space to backbone d_model space via input adapter
                residual_adapted = self.input_adapters[k](residual)  # [B * n_beams, d_model]

                dec_out_resid = self.decoder(
                    inputs_embeds=residual_adapted.unsqueeze(1),
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_resid = dec_out_resid.last_hidden_state[:, -1, :]  # [B * n_beams, d_model]
                kv_after_resid = dec_out_resid.past_key_values

                # NaN guard: if residual decoder produced NaN, DON'T contaminate KV cache
                if torch.isnan(hidden_resid).any():
                    # Fall back to greedy: use NTP step's past_key_values
                    past_key_values = past_key_values  # keep from NTP step
                    continue  # skip beam expansion for this step

                if n_resid_beams > 1:
                    # Compute codebook[k+1] classification logits from hidden_resid
                    # (same computation as in forward_residual's codebook_loss)
                    hidden_resid_latent = self.output_adapters[k + 1](hidden_resid.float())  # [B*n_beams, latent_size]
                    cb_weights_next = self.codebook_embs[k + 1]  # [codebook_size, latent_size]
                    cb_logits = hidden_resid_latent @ cb_weights_next.T.float()  # [B*n_beams, codebook_size]
                    cb_log_probs = F.log_softmax(cb_logits, dim=-1)  # [B*n_beams, codebook_size]

                    if resid_beam_mode == "pruning":
                        # ── Beam pruning: at most one candidate per parent beam ──
                        # → guaranteed unique SID prefixes (residual steps produce
                        #   no visible tokens, so multiple candidates from the same
                        #   parent would create output-space duplicates).
                        best_cb_log_prob, _ = cb_log_probs.max(dim=-1)  # [B*n_beams]
                        best_combined_score = beam_scores + resid_weight * best_cb_log_prob
                        best_combined_score_b = best_combined_score.reshape(B, n_beams)

                        top_scores, top_indices = torch.topk(
                            best_combined_score_b, n_beams, dim=1
                        )
                        reorder_idx_resid = (
                            top_indices + torch.arange(B, device=device)[:, None] * n_beams
                        ).reshape(-1)
                        beam_scores = top_scores.reshape(-1)

                    elif resid_beam_mode == "expansion":
                        # ── Beam expansion: all n_beams × codebook_size compete ──
                        # Combined score for every (beam, codebook_entry) pair.
                        next_scores_resid = (
                            beam_scores[:, None] + resid_weight * cb_log_probs
                        )  # [B*n_beams, codebook_size]
                        next_scores_resid = next_scores_resid.reshape(
                            B, n_beams * cb_log_probs.shape[-1]
                        )  # [B, n_beams * codebook_size]

                        # Select top 2*n_beams, then prune to n_beams
                        top_scores, top_indices = torch.topk(
                            next_scores_resid,
                            min(2 * n_beams, next_scores_resid.shape[-1]),
                            dim=1,
                        )  # [B, 2*n_beams]
                        top_scores = top_scores[:, :n_beams]
                        top_indices = top_indices[:, :n_beams]

                        # Reconstruct which parent beam each candidate came from
                        resid_beam_idx = torch.div(
                            top_indices, cb_log_probs.shape[-1], rounding_mode="floor"
                        )  # [B, n_beams]

                        reorder_idx_resid = (
                            resid_beam_idx + torch.arange(B, device=device)[:, None] * n_beams
                        ).reshape(-1)
                        beam_scores = top_scores.reshape(-1)

                    # Reorder KV cache from residual step
                    past_key_values = self._reorder_cache(kv_after_resid, reorder_idx_resid)

                    # Reorder ALL generated sids to match new beam order
                    for i in range(len(generated_sids)):
                        generated_sids[i] = generated_sids[i][reorder_idx_resid]
                else:
                    # resid_beams=1: greedy pass-through (original behavior)
                    past_key_values = kv_after_resid

        # ── Collect output ──
        # generated_sids: list of n_codebook_total tensors, each [B * n_beams, 1]
        output = torch.cat(generated_sids, dim=-1)  # [B * n_beams, n_codebook_total]
        output = output.reshape(B, n_beams, n_codebook_total)

        # ── Deduplicate: safety net to ensure unique sequences per batch item ──
        # Although the beam pruning at residual positions guarantees unique SID
        # prefixes at each intermediate step, the final NTP steps could still
        # produce identical sequences if two parent beams happen to generate the
        # same token (rare but possible). Deduplicate by keeping the highest-
        # scored beam for each unique sequence.
        beam_scores_b = beam_scores.reshape(B, n_beams)  # [B, n_beams]
        for b in range(B):
            unique_seqs, inverse_idx = torch.unique(
                output[b], dim=0, return_inverse=True
            )
            if unique_seqs.shape[0] < n_beams:
                # For each unique sequence, find the beam with the highest score
                # and use it as the representative. Then sort unique sequences
                # by their best beam score (descending) and fill remaining slots
                # by repeating the best sequence if needed.
                best_beam_per_unique = []
                best_score_per_unique = []
                for u in range(unique_seqs.shape[0]):
                    mask = (inverse_idx == u)
                    candidate_scores = beam_scores_b[b, mask]
                    best_idx = candidate_scores.argmax()
                    beam_indices = mask.nonzero(as_tuple=True)[0]
                    best_beam_per_unique.append(beam_indices[best_idx])
                    best_score_per_unique.append(candidate_scores[best_idx])

                # Sort unique sequences by best score (descending)
                sorted_order = torch.tensor(best_score_per_unique).argsort(descending=True)
                unique_output = output[b, torch.tensor(best_beam_per_unique)[sorted_order]]

                # If fewer unique sequences than n_beams, pad by repeating
                # the top sequence to maintain the expected output shape.
                # These padded duplicates won't affect metrics since they
                # duplicate the best candidate already counted.
                if unique_output.shape[0] < n_beams:
                    pad_count = n_beams - unique_output.shape[0]
                    unique_output = torch.cat(
                        [unique_output, unique_output[:1].repeat(pad_count, 1)],
                        dim=0,
                    )
                output[b] = unique_output

        # Take top num_return_sequences
        output = output[:, :num_return_sequences, :]  # [B, nrs, n_codebook]
        output = output.reshape(B * num_return_sequences, n_codebook_total)

        return output