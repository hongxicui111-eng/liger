"""
TIGER_SoftLabel: TIGER model with distance-based soft labels for SID prediction,
using RQ-VAE codebook embeddings as distance references. No residual interleaving.

This model is designed to isolate the effect of soft labels from the effect of
residual information. By comparing:
  - TIGER (hard label, no residual)  → baseline
  - TIGER_SoftLabel (soft label, no residual)  → effect of soft labels alone
  - TIGER_Residual (soft label + residual)  → combined effect

We can verify whether performance gains come from soft labels or from residual info.

The generation process is identical to standard TIGER (beam search on NTP logits),
since no residual positions are interleaved. The only difference is in training:
  - Standard TIGER: NTP loss uses hard one-hot labels (CE)
  - TIGER_SoftLabel: NTP loss uses distance-based soft labels from codebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from .tiger import TIGER


class TIGER_SoftLabel(TIGER):
    """
    TIGER with soft-label SID loss based on codebook distance.

    Uses the same standard T5 decoder as TIGER (no residual interleaving),
    but replaces the hard one-hot CE loss with distance-based soft labels:
    at each SID level k, the K nearest codebook entries to the ground truth
    receive softmax(-dist/temp) as their label probability.

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
        soft_label_K: number of nearest codebook entries used as positive samples
            for soft-label SID loss. 0 = hard label (original CE). Default: 0.
        soft_label_temperature: temperature for softmax normalization of distance-based
            soft labels. Higher = more uniform across K entries, lower = sharper.
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
        soft_label_K: int = 0,
        soft_label_temperature: float = 1.0,
    ):
        super().__init__(
            config=config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=max_items_per_seq,
            flag_use_output_embedding=flag_use_output_embedding,
            flag_use_learnable_text_embed=flag_use_learnable_text_embed,
            embedding_head_dict=embedding_head_dict,
        )

        self.codebook_size = codebook_size
        self.latent_size = latent_size
        self.soft_label_K = soft_label_K
        self.soft_label_temperature = soft_label_temperature

        # Precompute SID token offsets for each codebook level.
        # With uniform codebook_size: offset[level] = codebook_size * level + 1
        # With variable codebook_sizes: offset[level] = 1 + sum(codebook_sizes[:level])
        # This is used by _get_codebook_idx and _compute_soft_label_for_level
        # to correctly map between codebook indices and SID token IDs.
        if not hasattr(self, '_codebook_offsets') or self._codebook_offsets is None:
            self._codebook_offsets = None  # will be set by set_codebook_offsets()

        # Debug counter for soft-label distribution logging
        self._soft_label_log_step = 0
        self._soft_label_log_interval = 2000  # print every 20 forward calls

        # Register codebook embeddings (frozen, from RQ-VAE)
        # Same as TIGER_Residual — we need them to compute distance-based soft labels
        if rqvae_codebook_weights is not None:
            for i, cb_weight in enumerate(rqvae_codebook_weights):
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

    @property
    def codebook_embs(self):
        """Return list of codebook embedding tensors."""
        return [getattr(self, f"codebook_emb_{i}") for i in range(self.n_semantic_codebook)]

    def load_codebooks_from_rqvae(self, rqvae_model):
        """Load codebook weights from a trained RQ-VAE model."""
        for i, codebook in enumerate(rqvae_model.quantizer.codebooks):
            if i >= self.n_semantic_codebook:
                break
            weight = codebook.weight[:-1].detach().clone()  # [codebook_size, latent_size]
            buffer = getattr(self, f"codebook_emb_{i}")
            buffer.copy_(weight)

    def _get_codebook_idx(self, sid_tokens, level):
        """Convert SID token to codebook index (0 to codebook_size-1)."""
        # SID token = original_code + offset[level]
        # where offset is either codebook_size * level + 1 (uniform)
        # or cumulative sum of codebook_sizes + 1 (variable, for LETTER)
        if self._codebook_offsets is not None:
            cb_idx = sid_tokens - self._codebook_offsets[level]
        else:
            cb_idx = sid_tokens - 1 - level * self.codebook_size
        return cb_idx.clamp(0, self.codebook_size - 1)

    def set_codebook_offsets(self, codebook_sizes):
        """
        Set SID token offsets for variable codebook sizes (e.g., LETTER tokenizer).

        Args:
            codebook_sizes: list of per-level codebook sizes (e.g., [256, 256, 256, 256]).
                           Can also be an int for uniform sizes (backward compatible).
        """
        if isinstance(codebook_sizes, int):
            # Uniform: offset[level] = codebook_sizes * level + 1
            n_levels = self.n_semantic_codebook
            self._codebook_offsets = [codebook_sizes * i + 1 for i in range(n_levels)]
        else:
            # Variable (LETTER): offset[level] = 1 + sum(codebook_sizes[:level])
            offsets = [1]
            for sz in codebook_sizes[:-1]:  # exclude last since we only need offsets for existing levels
                offsets.append(offsets[-1] + sz)
            self._codebook_offsets = offsets

    def _compute_soft_label_for_level(
        self, level, gt_codebook_idx, device, B, vocab_size
    ):
        """
        Compute distance-based soft label distribution for SID prediction at level k.

        For each sample in the batch, find the K nearest codebook entries to the
        ground truth codebook embedding, then assign softmax(-dist/temp) probabilities.

        The soft labels are placed at the corresponding SID token positions in the
        full vocabulary (offset by codebook_size * level + 1).

        Args:
            level: codebook level k (0 to n_semantic_codebook-1)
            gt_codebook_idx: [B] ground truth codebook indices at level k
            device: torch device
            B: batch size
            vocab_size: full vocabulary size

        Returns:
            soft_labels: [B, vocab_size] soft label distribution
        """
        cb_weights = self.codebook_embs[level]  # [codebook_size, latent_size]
        gt_emb = cb_weights[gt_codebook_idx].float()  # [B, latent_size]

        # Squared L2 distance from each codebook entry to ground truth
        dist_sq = ((cb_weights.float().unsqueeze(0) - gt_emb.unsqueeze(1)) ** 2).sum(-1)  # [B, codebook_size]

        K = min(self.soft_label_K, self.codebook_size)
        _, topk_indices = torch.topk(dist_sq, k=K, dim=-1, largest=False)  # [B, K]

        # L2 distance for top-K entries
        topk_dist_sq = dist_sq.gather(1, topk_indices)  # [B, K]
        topk_dist = torch.sqrt(topk_dist_sq.clamp(min=1e-8))  # [B, K]

        # Softmax on negative distances / temperature
        neg_scaled_dist = -topk_dist / self.soft_label_temperature  # [B, K]
        soft_probs_topk = F.softmax(neg_scaled_dist, dim=-1)  # [B, K]

        # Map codebook indices to SID token IDs in the full vocabulary
        # SID token = codebook_idx + offset[level]
        # where offset is either codebook_size * level + 1 (uniform)
        # or cumulative sum of codebook_sizes + 1 (variable, for LETTER)
        if self._codebook_offsets is not None:
            topk_sid_tokens = topk_indices + self._codebook_offsets[level]  # [B, K]
        else:
            topk_sid_tokens = topk_indices + self.codebook_size * level + 1  # [B, K]

        # Build full soft label distribution over the entire vocabulary
        soft_labels = torch.zeros(B, vocab_size, device=device, dtype=torch.float32)
        soft_labels.scatter_(1, topk_sid_tokens, soft_probs_topk)  # [B, vocab_size]

        return soft_labels, topk_indices, topk_dist, soft_probs_topk

    def forward_softlabel(
        self,
        input_ids=None,
        attention_mask=None,
        labels_sids=None,
        inputs_embeds=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Forward pass with soft-label SID loss.

        Uses the standard T5 decoder (no residual interleaving), but replaces
        the hard one-hot CE loss with distance-based soft labels from codebook
        embeddings.

        The decoder runs in parallel over all SID positions (standard T5 behavior),
        and the soft label loss is computed per-level using codebook distance.

        Args:
            input_ids: [B, enc_seq_len] encoder input token IDs
            attention_mask: [B, enc_seq_len] encoder attention mask
            labels_sids: [B, n_codebook] ground truth SID tokens
            inputs_embeds: optional pre-computed encoder input embeddings
            encoder_outputs: optional pre-computed encoder outputs

        Returns:
            dict with keys: 'loss', 'sid_loss', 'logits', 'predicted_embedding'
        """
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        n_codebook = labels_sids.shape[1]

        # ── Encode the item sequence (standard TIGER encoding) ──
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

        encoder_hidden = encoder_outputs.last_hidden_state

        # ── Optional: predicted embedding for dense retrieval ──
        if self.flag_use_output_embedding:
            item_seq_len = attention_mask.sum(-1)
            self.predicted_embedding = self.gather_indexes(
                encoder_hidden, item_seq_len - 1
            )
        else:
            self.predicted_embedding = None

        # ── Build decoder input IDs and labels for standard T5 autoregressive ──
        # Standard T5: decoder_input_ids = [<BOS>, sid_0, sid_1, ..., sid_{n-2}]
        # labels = [sid_0, sid_1, ..., sid_{n-1}]
        # (T5ForConditionalGeneration handles shifting internally)
        decoder_input_ids = torch.cat([
            torch.full((B, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device),
            labels_sids[:, :-1],  # shift right: remove last token
        ], dim=1)  # [B, n_codebook]

        # ── Run standard T5 decoder (parallel over all positions) ──
        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        decoder_hidden = decoder_output.last_hidden_state  # [B, n_codebook, d_model]

        # ── Compute logits ──
        logits = self.lm_head(decoder_hidden).float()  # [B, n_codebook, vocab_size]

        # ── Compute SID loss with soft labels ──
        vocab_size = logits.shape[-1]
        sid_loss = 0.0

        for k in range(n_codebook):
            target_k = labels_sids[:, k]  # [B] ground truth SID token

            if self.soft_label_K > 0 and k < self.n_semantic_codebook:
                # ── Soft-label SID loss ──
                # Use codebook distance to create soft label distribution
                gt_cb_idx_k = self._get_codebook_idx(target_k, k)  # [B]

                soft_labels, topk_indices, topk_dist, soft_probs_topk = \
                    self._compute_soft_label_for_level(
                        level=k,
                        gt_codebook_idx=gt_cb_idx_k,
                        device=device,
                        B=B,
                        vocab_size=vocab_size,
                    )

                # Cross-entropy with soft targets:
                #   loss = -sum(soft_labels * log_softmax(logits))
                log_probs = F.log_softmax(logits[:, k, :], dim=-1)  # [B, vocab_size]
                level_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
                sid_loss += level_loss

                # ── Debug log: check soft-label sharpness ──
                if self._soft_label_log_step % self._soft_label_log_interval == 0:
                    K = min(self.soft_label_K, self.codebook_size)
                    n_show = min(10, B)
                    max_prob = soft_probs_topk[:n_show, 0].detach()
                    min_prob = soft_probs_topk[:n_show, -1].detach()
                    print(
                        f"\n[SoftLabel-SID] step={self._soft_label_log_step} "
                        f"level k={k}  temp={self.soft_label_temperature}  K={K}"
                    )
                    print(f"  {'sample':>6}  {'gt_cb':>6}  {'max_p':>7}  {'min_p':>7}  "
                          f"{'topK_dists (L2)':>40s}")
                    for i in range(n_show):
                        dists_str = "  ".join(f"{topk_dist[i, j].item():.2f}" for j in range(K))
                        print(
                            f"  {i:>6}  {topk_indices[i, 0].item():>6}  "
                            f"{max_prob[i].item():>7.4f}  {min_prob[i].item():>7.4f}  "
                            f"{dists_str:>40s}"
                        )
                    print()

                self._soft_label_log_step += 1

            else:
                # ── Hard-label: standard CE (original TIGER behavior) ──
                # For levels beyond n_semantic_codebook, or when soft_label_K=0
                sid_loss += F.cross_entropy(logits[:, k, :], target_k, reduction="mean")

        # Average over levels
        sid_loss = sid_loss / n_codebook

        total_loss = sid_loss

        return {
            "loss": total_loss,
            "sid_loss": sid_loss,
            "logits": logits,
            "predicted_embedding": self.predicted_embedding,
        }