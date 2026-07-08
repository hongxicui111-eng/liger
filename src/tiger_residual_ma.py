"""
MA-LIGER: Memory-Augmented LIGER
==================================

TIGER_Residual_MA extends TIGER_Residual with VQ Prototype Memory.

Key change from TIGER_Residual:
  After computing encoder_hidden (and predicted_embedding for LIGER),
  the module performs VQ Prototype Lookup to retrieve sid_context,
  then uses Cross-Attention to fuse it into encoder_hidden before
  passing to the decoder.

Design decisions (from discussion):
  - LIGER mode: predicted_embedding is already computed at L329,
    reused directly as E_current for Prototype lookup (no redundant gather)
  - TIGER mode: fallback gather_indexes to compute E_current
  - VQ hard assignment with STE gradient propagation
  - Commitment Loss + Diversity Loss returned alongside main loss
  - EMA update for prototype_vectors (in PrototypeMemory.forward)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from .tiger_residual import TIGER_Residual
from .prototype_memory import PrototypeMemory, PrototypeFusion


class TIGER_Residual_MA(TIGER_Residual):
    """TIGER_Residual with Memory-Augmented Prototype (MA-LIGER model).

    Adds VQ Prototype Memory and Cross-Attention fusion on top of
    the standard TIGER_Residual architecture.

    Extra Args (beyond TIGER_Residual):
        use_prototype: whether to enable Prototype Memory
        prototype_config: dict with keys:
            K: number of prototypes (default 256)
            p: number of nearest prototypes to retrieve (default 3)
            M: number of SID embeddings per prototype (default 10)
            beta: commitment loss weight (default 0.25)
            delta: diversity loss weight (default 0.1)
            ema_decay: EMA decay rate for prototype updates (default 0.99)
            n_heads: number of attention heads in cross-attention (default 8)
            dropout: dropout in cross-attention (default 0.1)
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
        use_prototype: bool = False,
        prototype_config: dict = None,
    ):
        super().__init__(
            config=config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=max_items_per_seq,
            flag_use_output_embedding=flag_use_output_embedding,
            flag_use_learnable_text_embed=flag_use_learnable_text_embed,
            embedding_head_dict=embedding_head_dict,
            rqvae_codebook_weights=rqvae_codebook_weights,
            codebook_size=codebook_size,
            latent_size=latent_size,
            codebook_loss_weight=codebook_loss_weight,
            num_residual_levels=num_residual_levels,
            soft_label_K=soft_label_K,
            soft_label_temperature=soft_label_temperature,
            soft_label_temp_min=soft_label_temp_min,
            soft_label_temp_decay_steps=soft_label_temp_decay_steps,
            codebook_loss_weight_decay_steps=codebook_loss_weight_decay_steps,
            cumulative_residual_loss_weight=cumulative_residual_loss_weight,
            cumulative_residual_loss_type=cumulative_residual_loss_type,
            cumulative_residual_loss_temperature=cumulative_residual_loss_temperature,
            resid_nontf_ratio=resid_nontf_ratio,
            ntp_nontf_ratio=ntp_nontf_ratio,
            flag_separate_bos_representation=flag_separate_bos_representation,
            sid_start_token_id=sid_start_token_id,
        )

        self.use_prototype = use_prototype
        d_model = config.d_model

        if use_prototype:
            pconfig = prototype_config or {}
            K = pconfig.get("K", 256)
            p = pconfig.get("p", 3)
            L = pconfig.get("L", 8)
            beta = pconfig.get("beta", 0.25)
            delta = pconfig.get("delta", 0.1)
            ema_decay = pconfig.get("ema_decay", 0.99)
            n_heads = pconfig.get("n_heads", 8)
            gate_hidden = pconfig.get("gate_hidden", 64)
            dropout = pconfig.get("dropout", 0.1)

            self.prototype_memory = PrototypeMemory(
                d_model=d_model,
                K=K,
                p=p,
                L=L,
                beta=beta,
                delta=delta,
                ema_decay=ema_decay,
            )
            self.proto_fusion = PrototypeFusion(
                d_model=d_model,
                n_heads=n_heads,
                gate_hidden=gate_hidden,
                dropout=dropout,
            )
            print(
                f"[MA-LIGER] Prototype Memory initialized: "
                f"K={K}, p={p}, L={L}, beta={beta}, delta={delta}, "
                f"ema_decay={ema_decay}, n_heads={n_heads}, gate_hidden={gate_hidden}"
            )
        else:
            self.prototype_memory = None
            self.proto_fusion = None

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
        Forward pass with residual interleaving + Prototype Memory augmentation.

        The only difference from the parent TIGER_Residual.forward_residual is:
        after encoder_hidden is computed (L335 in parent), we:
          1. Retrieve E_current (predicted_embedding or fallback gather)
          2. VQ lookup → sid_context
          3. Cross-Attention fuse → encoder_hidden_aug
        Then the rest of the decoding proceeds identically.

        Returns:
            dict with same keys as parent, plus:
                'commitment_loss': scalar
                'diversity_loss': scalar
                'prototype_active_count': int
                'prototype_max_usage': int
        """
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # ============= Phase 1: Encode (identical to parent) =============
        n_codebook = labels_sids.shape[1]
        n_sem = self.n_semantic_codebook
        num_resid = self.num_residual_levels

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

        # predicted_embedding for dense retrieval (LIGER mode)
        if self.flag_use_output_embedding:
            item_seq_len = attention_mask.sum(-1)
            self.predicted_embedding = self.gather_indexes(
                encoder_outputs.last_hidden_state, item_seq_len - 1
            )
        else:
            self.predicted_embedding = None

        encoder_hidden = encoder_outputs.last_hidden_state

        # ============= Phase 2: Prototype Memory Augmentation (Dual-Path) =============
        proto_losses = {}
        gate_values = None
        if self.use_prototype and self.prototype_memory is not None:
            # Get E_current
            if self.predicted_embedding is not None:
                E_current = self.predicted_embedding  # [B, d_model]
            else:
                item_seq_len = attention_mask.sum(-1)
                E_current = self.gather_indexes(encoder_hidden, item_seq_len - 1)

            # VQ Prototype Lookup → interest tokens
            interest_context, proto_losses = self.prototype_memory(E_current)  # [B, p*L, d]

            # Dual-path fusion:
            #   Generation: concatenate interest_context to encoder_hidden
            #   Dense: augment predicted_embedding via Gate
            encoder_hidden, attention_mask, predicted_embedding_aug, gate_values = self.proto_fusion(
                encoder_hidden, interest_context, E_current,
                attention_mask=attention_mask,
            )

            # Override predicted_embedding with the augmented version
            if self.predicted_embedding is not None and self.flag_use_output_embedding:
                self.predicted_embedding = predicted_embedding_aug

        # ============= Phase 3: Scheduled parameters (identical to parent) =============
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

        codebook_weight_decay_step = self._codebook_weight_decay_step
        per_position_alphas = []
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
            for t in range(n_sem):
                cb_idx_t = self._get_codebook_idx(labels_sids[:, t], t)
                gt_cb_embs.append(self.codebook_embs[t][cb_idx_t].float())
            cumsum_gt = [gt_cb_embs[-1]]
            for t in range(n_sem - 2, -1, -1):
                cumsum_gt.insert(0, gt_cb_embs[t] + cumsum_gt[0])

        # ============= Phase 4: Decode (identical to parent) =============
        past_key_values = None
        all_ntp_logits = []
        predicted_sids = []
        prev_residual_output_latent = None
        sid_loss = 0.0
        codebook_loss = 0.0
        num_sid_steps = 0
        num_codebook_steps = 0

        nontf_resid_count = 0
        nontf_ntp_count = 0
        nontf_resid_total = 0
        nontf_ntp_total = 0

        # BOS-only step
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

        for k in range(n_codebook):
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
                    ntp_nontf_mask = torch.rand(B, device=device) < self.ntp_nontf_ratio
                    predicted_sid_prev = predicted_sids[-1]
                    dec_input_ids = torch.where(
                        ntp_nontf_mask.unsqueeze(-1),
                        predicted_sid_prev.unsqueeze(-1),
                        labels_sids[:, k - 1 : k],
                    )
                    nontf_ntp_count += ntp_nontf_mask.sum().item()
                    nontf_ntp_total += B
                else:
                    dec_input_ids = labels_sids[:, k - 1 : k]

            decoder_out = self.decoder(
                input_ids=dec_input_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            hidden = decoder_out.last_hidden_state[:, -1, :]
            past_key_values = decoder_out.past_key_values

            logits = self.lm_head(hidden).float()
            all_ntp_logits.append(logits)

            predicted_sid_k = logits.argmax(dim=-1)
            predicted_sids.append(predicted_sid_k)

            target_k = labels_sids[:, k]
            sid_loss += F.cross_entropy(logits, target_k, reduction="sum") / B
            num_sid_steps += 1

            if k < num_resid and k + 1 < n_codebook:
                if self.resid_nontf_ratio > 0:
                    resid_nontf_mask_k = torch.rand(B, device=device) < self.resid_nontf_ratio
                    blend_sid_k = torch.where(
                        resid_nontf_mask_k,
                        predicted_sid_k,
                        labels_sids[:, k],
                    )
                    cb_idx_k = self._get_codebook_idx(blend_sid_k, k)
                    nontf_resid_count += resid_nontf_mask_k.sum().item()
                    nontf_resid_total += B
                else:
                    cb_idx_k = self._get_codebook_idx(labels_sids[:, k], k)
                cb_emb = self.codebook_embs[k][cb_idx_k]

                if k == 0:
                    h_latent = self.output_adapters[k](hidden.float())
                else:
                    h_latent = prev_residual_output_latent

                residual = (h_latent - cb_emb.float())
                residual_max = self.latent_size * 2.0
                residual = residual.clamp(-residual_max, residual_max)

                residual_adapted = self.input_adapters[k](residual.to(hidden.dtype))
                residual_input = residual_adapted.unsqueeze(1)

                decoder_out_resid = self.decoder(
                    inputs_embeds=residual_input,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_resid = decoder_out_resid.last_hidden_state[:, -1, :]

                if not torch.isnan(hidden_resid).any():
                    past_key_values = decoder_out_resid.past_key_values

                hidden_resid_latent = self.output_adapters[k + 1](hidden_resid.float())
                prev_residual_output_latent = hidden_resid_latent

                # Codebook CE loss (inline helper, same as parent)
                def _compute_code_loss(pred_latent, level_k, labels_sids_batch, B_val, device_val):
                    cb_weights = self.codebook_embs[level_k + 1]
                    logits_cb = pred_latent @ cb_weights.T.float()
                    cb_idx_next = self._get_codebook_idx(labels_sids_batch[:, level_k + 1], level_k + 1)

                    if self.soft_label_K > 0:
                        gt_emb = cb_weights[cb_idx_next].float()
                        dist_sq = ((cb_weights.float().unsqueeze(0) - gt_emb.unsqueeze(1)) ** 2).sum(-1)
                        K_val = min(self.soft_label_K, self.codebook_size)
                        _, topk_indices = torch.topk(dist_sq, k=K_val, dim=-1, largest=False)
                        topk_dist_sq = dist_sq.gather(1, topk_indices)
                        topk_dist = torch.sqrt(topk_dist_sq.clamp(min=1e-8))
                        neg_scaled_dist = -topk_dist / current_soft_label_temp
                        soft_probs_topk = F.softmax(neg_scaled_dist, dim=-1)
                        soft_labels = torch.zeros(B_val, self.codebook_size, device=device_val, dtype=torch.float32)
                        soft_labels.scatter_(1, topk_indices, soft_probs_topk)
                        log_probs = F.log_softmax(logits_cb, dim=-1)
                        return -(soft_labels * log_probs).sum(dim=-1).mean()
                    else:
                        return F.cross_entropy(logits_cb, cb_idx_next, reduction="sum") / B_val

                codebook_loss += _compute_code_loss(hidden_resid_latent, k, labels_sids, B, device) * per_position_alphas[k + 1]
                num_codebook_steps += 1
                if k == 0:
                    codebook_loss += _compute_code_loss(h_latent, k - 1, labels_sids, B, device) * per_position_alphas[0]
                    num_codebook_steps += 1

                if self.cumulative_residual_loss_weight > 0:
                    target_cumul = cumsum_gt[k + 1]
                    if self.cumulative_residual_loss_type == "cosine":
                        cos_sim = F.cosine_similarity(hidden_resid_latent, target_cumul, dim=-1)
                        cumulative_residual_loss += (1.0 - cos_sim).sum() / B
                    elif self.cumulative_residual_loss_type == "ce":
                        sim_matrix = hidden_resid_latent @ target_cumul.T
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

        # ============= Phase 5: Combine losses =============
        sid_loss = sid_loss / num_sid_steps if num_sid_steps > 0 else 0.0
        codebook_loss = codebook_loss / num_codebook_steps if num_codebook_steps > 0 else 0.0
        cumulative_residual_loss = cumulative_residual_loss / num_cumul_steps if num_cumul_steps > 0 else 0.0

        total_loss = (
            sid_loss
            + self.codebook_loss_weight * codebook_loss
            + self.cumulative_residual_loss_weight * cumulative_residual_loss
        )

        # Add Prototype losses
        if self.use_prototype and proto_losses:
            commitment_loss = proto_losses["commitment_loss"]
            diversity_loss = proto_losses["diversity_loss"]
            total_loss = total_loss + self.prototype_memory.beta * commitment_loss
            total_loss = total_loss + self.prototype_memory.delta * diversity_loss

        stacked_logits = torch.stack(all_ntp_logits, dim=1)

        # Debug logging (same as parent)
        if self.soft_label_temp_min is not None and self.soft_label_K > 0:
            if self._soft_label_log_step % self._soft_label_log_interval == 0:
                progress_pct = min(
                    (self._soft_label_temp_step - 1) / self.soft_label_temp_decay_steps * 100, 100.0
                )
                print(
                    f"\n[TempSchedule] step={self._soft_label_temp_step} "
                    f"T={current_soft_label_temp:.4f} "
                    f"(init={self.soft_label_temperature:.2f} -> min={self.soft_label_temp_min:.2f}, "
                    f"progress={progress_pct:.1f}%)"
                )
            self._soft_label_log_step += 1

        if self.resid_nontf_ratio > 0 or self.ntp_nontf_ratio > 0:
            if self._nontf_log_step % self._nontf_log_interval == 0:
                resid_pct = (nontf_resid_count / nontf_resid_total * 100) if nontf_resid_total > 0 else 0.0
                ntp_pct = (nontf_ntp_count / nontf_ntp_total * 100) if nontf_ntp_total > 0 else 0.0
                print(
                    f"\n[NonTF] step={self._nontf_log_step} "
                    f"resid_nontf_ratio={self.resid_nontf_ratio:.2f} -> "
                    f"{nontf_resid_count}/{nontf_resid_total} ({resid_pct:.1f}% used predicted) "
                    f"ntp_nontf_ratio={self.ntp_nontf_ratio:.2f} -> "
                    f"{nontf_ntp_count}/{nontf_ntp_total} ({ntp_pct:.1f}% used predicted)"
                )
            self._nontf_log_step += 1

        if self.codebook_loss_weight_decay_steps is not None:
            if not hasattr(self, '_cb_weight_decay_log_step'):
                self._cb_weight_decay_log_step = 0
                self._cb_weight_decay_log_interval = 50
            if self._cb_weight_decay_log_step % self._cb_weight_decay_log_interval == 0:
                alpha_str = ", ".join(
                    f"k{k}:alpha={per_position_alphas[k]:.3f}" for k in range(n_sem)
                )
                print(
                    f"\n[CBWeightDecay] step={codebook_weight_decay_step} {alpha_str}"
                )
            self._cb_weight_decay_log_step += 1

        # Prototype logging
        if self.use_prototype and proto_losses:
            if not hasattr(self, '_proto_log_step'):
                self._proto_log_step = 0
                self._proto_log_interval = 50
            if self._proto_log_step % self._proto_log_interval == 0:
                print(
                    f"\n[Prototype] active={proto_losses['prototype_active_count']}/{self.prototype_memory.K} "
                    f"max_usage={proto_losses['prototype_max_usage']} "
                    f"commit_loss={proto_losses['commitment_loss'].item():.4f} "
                    f"divers_loss={proto_losses['diversity_loss'].item():.4f}"
                )
            self._proto_log_step += 1

        result = {
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

        # Add prototype-specific losses to result
        if self.use_prototype and proto_losses:
            result["commitment_loss"] = proto_losses["commitment_loss"]
            result["diversity_loss"] = proto_losses["diversity_loss"]
            result["prototype_active_count"] = proto_losses["prototype_active_count"]
            result["prototype_max_usage"] = proto_losses["prototype_max_usage"]
            if gate_values is not None:
                result["gate_values"] = gate_values  # [B] for logging

        return result
