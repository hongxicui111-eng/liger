"""
TIGER_Residual: TIGER model with recursive residual information from RQ-VAE
codebooks interleaved in the decoder for improved semantic ID generation.

The residual computation follows the RQ-VAE recursive pattern:
  k=0: residual_0 = output_adapter[0](hidden_0) - codebook_emb[0][sid_0]
  k>0: residual_k = output_adapter[k](hidden_resid_{k-1}) - codebook_emb[k][sid_k]

At each step k>0, the output_adapter[k] projects the previous step's residual
decoder output (hidden_resid_{k-1}) to latent space, then subtracts the current
level's codebook entry. This mirrors how RQ-VAE quantizes: each level subtracts
its codebook entry from the *previous residual representation*, not from a fresh
projection of the original input.

The generation process:
  Standard: <BOS> -> resid_0 -> sid_0 -> resid_1 -> sid_1 -> ... -> sid_{n-2} -> sid_{n-1}
  With flag_separate_bos_representation: <BOS> -> <SID_START> -> sid_0 -> resid_0 -> sid_1 -> resid_1 -> ...

Where:
  - Standard positions (BOS, SID_START, sid_k):  input = token embedding, output = NTP logits -> NTP loss
  - Residual positions (resid_k):     input = residual vector,   output = hidden -> MSE loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from .tiger import TIGER


def _kv_to_dynamic_cache(past_key_values):
    """Convert tuple-based KV cache to DynamicCache for transformers >= 4.44.

    Newer transformers versions require DynamicCache instead of raw tuples.
    Returns None if input is None, DynamicCache if input is a tuple,
    or the input unchanged if already a DynamicCache.
    """
    if past_key_values is None:
        return None
    # Already a cache object (DynamicCache or compatible)
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values
    # Convert tuple: ((key_layer, value_layer), ...) -> DynamicCache
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, layer_kv in enumerate(past_key_values):
        if layer_kv is not None:
            key, value = layer_kv
            cache.update(key, value, layer_idx)
    return cache


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
        soft_label_temp_min: minimum temperature for soft-label temperature scheduling.
            As training progresses, the soft-label temperature linearly decays from
            soft_label_temperature (initial) to this value, making the distribution
            progressively sharper. Only used when soft_label_K > 0.
            Default: None (no scheduling, temperature stays fixed at soft_label_temperature).
        soft_label_temp_decay_steps: total training steps over which the temperature
            decays from soft_label_temperature to soft_label_temp_min.
            Each forward_residual call counts as one step.
            Only used when soft_label_temp_min is set.
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
        cumulative_residual_loss_weight: weight for cumulative residual alignment loss.
            At each residual position k, constrains output_adapter[k+1](hidden_resid_k)
            to be close to sum(codebook_emb[t][sid_t] for t in [k+1, ..., n-1]).
            This enforces that the residual representation captures the *full remaining*
            codebook content, not just the next codebook index. Default: 0.0 (disabled).
        cumulative_residual_loss_type: "mse" for L2 regression, "cosine" for cosine
            similarity loss, or "ce" for batch-wise cross-entropy loss based on
            inner product (InfoNCE-style contrastive loss). With "ce", each sample's
            predicted residual representation (hidden_resid_latent) is treated as a
            query, and all samples' cumulative codebook targets (target_cumul) in the
            batch serve as keys. The positive pair is the same sample index (diagonal),
            and all other samples are negatives. The cross-entropy loss encourages the
            residual representation to be more discriminative across items.
            Default: "mse".
        cumulative_residual_loss_temperature: temperature scaling for the "ce" loss
            type. Divides the inner-product logits before softmax. Higher temperature
            yields softer probability distributions (easier negatives), lower temperature
            makes the loss focus more on hard negatives. Only used when
            cumulative_residual_loss_type="ce". Default: 1.0.
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
        soft_label_temp_min: float = None,
        soft_label_temp_decay_steps: int = 10000,
        codebook_loss_weight_decay_steps: list = None,
        cumulative_residual_loss_weight: float = 0.0,
        cumulative_residual_loss_type: str = "mse",
        cumulative_residual_loss_temperature: float = 1.0,
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
        # Precompute SID token offsets for each codebook level.
        # With uniform codebook_size: offset[level] = codebook_size * level + 1
        # With variable codebook_sizes: offset[level] = 1 + sum(codebook_sizes[:level])
        self._codebook_offsets = None  # set by set_codebook_offsets()
        self.num_residual_levels = (
            num_residual_levels
            if num_residual_levels is not None
            else n_semantic_codebook - 1
        )
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




        self.input_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_size, self.d_model, bias=False),
                nn.LayerNorm(self.d_model, eps=config.layer_norm_epsilon),
            )
            for _ in range(self.num_residual_levels)
        ])

   

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
        prev_residual_output_latent = None  # recursive: carries hidden_resid_{k-1} in d_model space into step k
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

            past_key_values_kv = _kv_to_dynamic_cache(past_key_values)
            decoder_out = self.decoder(
                input_ids=dec_input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values_kv,
                use_cache=True,
                return_dict=True,
            )
            # sid的输出
            hidden = decoder_out.last_hidden_state[:, -1, :]  # [B, d_model]
            past_key_values = decoder_out.past_key_values

            # NTP logits for this step — cast to FP32 for numerical stability under AMP
            logits = self.lm_head(hidden).float()  # [B, vocab_size]
            all_ntp_logits.append(logits)

            # Store predicted sid_k for potential non-TF use in subsequent steps
            predicted_sid_k = logits.argmax(dim=-1)  # [B]
            predicted_sids.append(predicted_sid_k)

            target_k = labels_sids[:, k]  # [B]
            sid_loss += F.cross_entropy(logits, target_k, reduction="sum") / B
            num_sid_steps += 1

            if k < num_resid and k + 1 < n_codebook:
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


                if k == 0:
                    h_latent = self.output_adapters[k](hidden.float())  # [B, latent_size]
                else:
                    h_latent = prev_residual_output_latent  # [B, latent_size]


                residual = (h_latent - cb_emb.float())  # [B, latent_size]

                # Clamp residual magnitude to prevent extreme values destabilizing decoder
                residual_max = self.latent_size * 2.0  # heuristic bound in latent space
                residual = residual.clamp(-residual_max, residual_max)

                # Map residual from latent space to backbone d_model space via input adapter,
                # then feed into decoder. Each level has its own adapter (no sharing).
                residual_adapted = self.input_adapters[k](residual.to(hidden.dtype))  # [B, d_model]
                residual_input = residual_adapted.unsqueeze(1)  # [B, 1, d_model]

                past_key_values_kv = _kv_to_dynamic_cache(past_key_values)
                decoder_out_resid = self.decoder(
                    inputs_embeds=residual_input,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values_kv,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_resid = decoder_out_resid.last_hidden_state[:, -1, :]  # [B, d_model]

                # NaN guard: if residual decoder produced NaN, DON'T contaminate KV cache
                # Fall back to the NTP step's past_key_values instead
                if not torch.isnan(hidden_resid).any():
                    past_key_values = decoder_out_resid.past_key_values
                

                hidden_resid_latent = self.output_adapters[k + 1](hidden_resid.float())  # [B, latent_size]
                
                prev_residual_output_latent= hidden_resid_latent
                # ── Inline helper: compute codebook CE loss (soft or hard) ──
                def _compute_code_loss(pred_latent, level_k, labels_sids_batch, B_val, device_val):
                    """
                    Compute codebook cross-entropy loss for level (level_k + 1).

                    Args:
                        pred_latent: [B, latent_size] — predicted latent from output adapter
                        level_k:     current residual level (targets codebook[level_k+1])
                        labels_sids_batch: [B, n_codebook] — ground-truth sid tokens
                        B_val:       batch size
                        device_val:  torch device

                    Returns:
                        scalar loss (already averaged over batch for hard-label,
                        or mean for soft-label)
                    """
                    cb_weights = self.codebook_embs[level_k + 1]  # [codebook_size, latent_size]
                    logits_cb = pred_latent @ cb_weights.T.float()  # [B, codebook_size]
                    cb_idx_next = self._get_codebook_idx(labels_sids_batch[:, level_k + 1], level_k + 1)  # [B]

                    if self.soft_label_K > 0:
                        # Ground truth embedding in codebook[level_k+1]
                        gt_emb = cb_weights[cb_idx_next].float()  # [B, latent_size]

                        # Squared L2 distance: ||cb_weights[j] - gt_emb[b]||^2
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
                        # Uses scheduled temperature (decays over training) if enabled
                        neg_scaled_dist = -topk_dist / current_soft_label_temp  # [B, K]
                        soft_probs_topk = F.softmax(neg_scaled_dist, dim=-1)  # [B, K], sums to 1

                        # Build full soft label distribution over entire codebook
                        soft_labels = torch.zeros(B_val, self.codebook_size, device=device_val, dtype=torch.float32)
                        soft_labels.scatter_(1, topk_indices, soft_probs_topk)  # [B, codebook_size]

                        # Cross-entropy with soft targets:
                        #   loss = -sum(soft_labels * log_softmax(logits_cb))
                        # This generalizes hard CE (when soft_labels is one-hot, it reduces to CE)
                        log_probs = F.log_softmax(logits_cb, dim=-1)  # [B, codebook_size]
                        return -(soft_labels * log_probs).sum(dim=-1).mean()
                    else:
                        # ── Hard-label: standard CE (original behavior) ──
                        return F.cross_entropy(logits_cb, cb_idx_next, reduction="sum") / B_val

                codebook_loss += _compute_code_loss(hidden_resid_latent, k, labels_sids, B, device) * per_position_alphas[k + 1]
                num_codebook_steps += 1
                if k==0: #上面算的是第1级，现在需要算一下第第0级的loss
                    codebook_loss+=_compute_code_loss(h_latent,k-1,labels_sids,B,device) * per_position_alphas[0]
                    num_codebook_steps += 1

                if self.cumulative_residual_loss_weight > 0:
                    target_cumul = cumsum_gt[k + 1]  # [B, latent_size]: sum of codes k+1..n-1
                    if self.cumulative_residual_loss_type == "cosine":
                        cos_sim = F.cosine_similarity(
                            hidden_resid_latent, target_cumul, dim=-1
                        )
                        cumulative_residual_loss += (1.0 - cos_sim).sum() / B
                    elif self.cumulative_residual_loss_type == "ce":
                        sim_matrix = hidden_resid_latent @ target_cumul.T  # [B, B]
                        sim_matrix = sim_matrix / self.cumulative_residual_loss_temperature
                        labels_ce = torch.arange(B, device=device)
                        cumulative_residual_loss += F.cross_entropy(
                            sim_matrix, labels_ce, reduction="mean"
                        )
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

        # ── Debug log: soft-label temperature scheduling ──
        if self.soft_label_temp_min is not None and self.soft_label_K > 0:
            if self._soft_label_log_step % self._soft_label_log_interval == 0:
                progress_pct = min(
                    (self._soft_label_temp_step - 1) / self.soft_label_temp_decay_steps * 100, 100.0
                )
                print(
                    f"\n[TempSchedule] step={self._soft_label_temp_step} "
                    f"T={current_soft_label_temp:.4f} "
                    f"(init={self.soft_label_temperature:.2f} → min={self.soft_label_temp_min:.2f}, "
                    f"progress={progress_pct:.1f}%)"
                )
            self._soft_label_log_step += 1

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
            "nontf_resid_pct": nontf_resid_count / nontf_resid_total if nontf_resid_total > 0 else 0.0,
            "nontf_ntp_pct": nontf_ntp_count / nontf_ntp_total if nontf_ntp_total > 0 else 0.0,
            "current_soft_label_temp": current_soft_label_temp,
            "soft_label_temp_progress": soft_label_temp_progress,
            "soft_label_temp_step": soft_label_temp_step,
            "codebook_weight_decay_step": codebook_weight_decay_step,
            "per_position_alphas": per_position_alphas,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder past_key_values for beam search.

        Supports both tuple format and DynamicCache (transformers >= 4.44).
        """
        # DynamicCache path
        if hasattr(past_key_values, 'get_seq_length'):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        # Tuple path: ((key_0, value_0), (key_1, value_1), ...)
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
        prev_residual_output_latent = None  # recursive: carries output_adapter result in latent space

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

            past_key_values_kv = _kv_to_dynamic_cache(past_key_values)
            decoder_out = self.decoder(
                input_ids=dec_input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values_kv,
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

            # Reorder prev_residual_output_latent to match new beam order (needed for recursive residual)
            if prev_residual_output_latent is not None:
                prev_residual_output_latent = prev_residual_output_latent[reorder_indices]  # [B * n_beams, latent_size]

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

                # Compute residual in codebook latent space (recursive, like RQ-VAE):
                #   k=0: residual_0 = output_adapter[0](hidden_0) - embed_0(sid_0)
                #   k>0: residual_k = prev_residual_output_latent - embed_k(sid_k)
                #   (prev_residual_output_latent is output_adapter[k+1](hidden_resid) from step k-1)
                if k == 0:
                    h_latent = self.output_adapters[k](hidden.float())  # [B * n_beams, latent_size]
                else:
                    h_latent = prev_residual_output_latent  # [B * n_beams, latent_size]
                residual = h_latent - cb_emb  # [B * n_beams, latent_size]

                # Map residual from latent space to backbone d_model space via input adapter
                residual_adapted = self.input_adapters[k](residual)  # [B * n_beams, d_model]

                past_key_values_kv = _kv_to_dynamic_cache(past_key_values)
                dec_out_resid = self.decoder(
                    inputs_embeds=residual_adapted.unsqueeze(1),
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values_kv,
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

                # Store output_adapter result for recursive residual computation at next step
                hidden_resid_latent = self.output_adapters[k + 1](hidden_resid.float())  # [B * n_beams, latent_size]
                prev_residual_output_latent = hidden_resid_latent

                if n_resid_beams > 1:
                    # Compute codebook[k+1] classification logits from hidden_resid
                    # (same computation as in forward_residual's codebook_loss)
                    # hidden_resid_latent already computed above
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
                    # Reorder prev_residual_output_latent to match new beam order
                    if prev_residual_output_latent is not None:
                        prev_residual_output_latent = prev_residual_output_latent[reorder_idx_resid]
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