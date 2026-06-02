"""
TIGER_Residual: TIGER model with residual information from RQ-VAE codebooks
interleaved in the decoder for improved semantic ID generation.

The generation process:
  <BOS> -> resid_0 -> sid_0 -> resid_1 -> sid_1 -> ... -> sid_{n-2} -> sid_{n-1}

Where:
  - Standard positions (BOS, sid_k):  input = token embedding, output = NTP logits -> NTP loss
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
        mse_loss_weight: weight for MSE loss term
        num_residual_levels: how many residual positions to use (default: n_semantic_codebook - 1)
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
        mse_loss_weight: float = 1.0,
        num_residual_levels: int = None,
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
        self.mse_loss_weight = mse_loss_weight
        self.num_residual_levels = (
            num_residual_levels
            if num_residual_levels is not None
            else n_semantic_codebook - 1
        )

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

        # Build projection layers if dimensions differ
        if self.d_model != self.latent_size:
            self.embed_to_dmodel = nn.Sequential(
                nn.Linear(self.latent_size, self.d_model, bias=False),
                nn.LayerNorm(self.d_model, eps=config.layer_norm_epsilon),
            )
            self.dmodel_to_latent = nn.Sequential(
                nn.Linear(self.d_model, self.latent_size, bias=False),
                nn.LayerNorm(self.latent_size, eps=config.layer_norm_epsilon),
            )
        else:
            self.embed_to_dmodel = nn.Identity()
            self.dmodel_to_latent = nn.Identity()

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
            dict with keys: 'loss', 'sid_loss', 'mse_loss', 'logits', 'predicted_embedding'
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

        # ---- Decode step-by-step with residual interleaving ----
        past_key_values = None
        all_ntp_logits = []  # [n_codebook] list of [B, V]
        sid_loss = 0.0
        mse_loss = 0.0

        for k in range(n_codebook):
            # === Standard position: predict sid_k via NTP ===
            if k == 0:
                dec_input_ids = torch.full(
                    (B, 1),
                    self.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
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

            # NTP loss for sid_k — compute in FP32
            target_k = labels_sids[:, k]  # [B]
            sid_loss += F.cross_entropy(logits, target_k, reduction="sum") / B

            # === Residual position (if we need one after this level) ===
            # We add a residual position for levels 0 through num_resid-1
            if k < num_resid and k + 1 < n_codebook:
                # Get codebook embedding for ground truth sid_k at level k
                cb_idx_k = self._get_codebook_idx(labels_sids[:, k], k)
                cb_emb = self.codebook_embs[k][cb_idx_k]  # [B, latent_size]
                cb_emb_d = self.embed_to_dmodel(cb_emb)  # [B, d_model]

                # Compute residual: sg(h_k) - proj(embed_k(sid_k))
                # Cast to FP32 to prevent overflow/NaN under AMP FP16 autocast
                residual = (hidden.detach().float() - cb_emb_d.float())  # [B, d_model]

                # Clamp residual magnitude to prevent extreme values destabilizing decoder
                residual_max = self.d_model * 2.0  # heuristic bound
                residual = residual.clamp(-residual_max, residual_max)

                # Feed residual as input embedding to decoder
                # Cast back to original dtype after clamping
                residual_input = residual.to(hidden.dtype).unsqueeze(1)  # [B, 1, d_model]

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

                # MSE loss: project to latent space, compare with next level's codebook embedding
                # Cast to FP32 for MSE stability
                hidden_resid_latent = self.dmodel_to_latent(hidden_resid.float())  # [B, latent_size]

                # Target: codebook[k+1][sid_{k+1}]
                cb_idx_next = self._get_codebook_idx(labels_sids[:, k + 1], k + 1)
                target_emb = self.codebook_embs[k + 1][cb_idx_next]  # [B, latent_size]

                mse_loss += F.mse_loss(hidden_resid_latent, target_emb.float(), reduction="sum") / B

        # ---- Combine losses ----
        total_loss = sid_loss + self.mse_loss_weight * mse_loss

        # Stack NTP logits: [B, n_codebook, vocab_size]
        stacked_logits = torch.stack(all_ntp_logits, dim=1)

        return {
            "loss": total_loss,
            "sid_loss": sid_loss,
            "mse_loss": mse_loss,
            "logits": stacked_logits,
            "predicted_embedding": self.predicted_embedding,
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
        **kwargs,
    ):
        """
        Beam-search generation with residual interleaving.

        Uses beam search at SID prediction positions (NTP) and greedy pass-through
        at residual positions (just feed through decoder, no beam expansion).

        Args:
            input_ids: [B, enc_seq_len]
            attention_mask: [B, enc_seq_len]
            inputs_embeds: optional pre-computed encoder inputs_embeds
            max_new_tokens: number of SID tokens to generate (= n_codebook)
            num_beams: number of beams for beam search at SID positions
            num_return_sequences: number of returned sequences (≤ num_beams)

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

        # ── Generate step by step ──
        for k in range(n_codebook_total):
            # ---- Decoder input for this NTP step ----
            if k == 0:
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

            # ---- Residual pass-through (greedy, no beam expansion) ----
            if k < self.num_residual_levels and k + 1 < n_codebook_total:
                cb_idx_k = self._get_codebook_idx(
                    sid_k.squeeze(-1), k
                )  # [B * n_beams]
                cb_emb = self.codebook_embs[k][cb_idx_k]  # [B * n_beams, latent_size]
                cb_emb_d = self.embed_to_dmodel(cb_emb)  # [B * n_beams, d_model]
                residual = hidden - cb_emb_d  # [B * n_beams, d_model]

                dec_out_resid = self.decoder(
                    inputs_embeds=residual.unsqueeze(1),
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = dec_out_resid.past_key_values

        # ── Collect output ──
        # generated_sids: list of n_codebook_total tensors, each [B * n_beams, 1]
        output = torch.cat(generated_sids, dim=-1)  # [B * n_beams, n_codebook_total]
        output = output.reshape(B, n_beams, n_codebook_total)

        # Take top num_return_sequences
        output = output[:, :num_return_sequences, :]  # [B, nrs, n_codebook]
        output = output.reshape(B * num_return_sequences, n_codebook_total)

        return output