"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from transformers import T5Config
from transformers.optimization import get_scheduler

from utils import CustomDataset, get_lr, setup_logging
import random
from .evaluation import (
    evaluate,
    evaluate_dense_ids,
    evaluate_dense_sids,
    generate_then_dense,
    get_target_embed,
    model_forward,
    model_forward_softlabel,
)
from .load_data import load_data
from .tiger import TIGER


# =============================================================================
# Continue Training Utilities
# Two-phase training: Phase 1 (adapter warm-up, frozen backbone)
#                      Phase 2 (joint fine-tuning, all params)
# =============================================================================


def _is_adapter_param(name: str) -> bool:
    """Check if a parameter belongs to adapter layers (not the T5 backbone).

    Adapter layers are the ones added by TIGER_Residual / Simple_Residual on
    top of the base TIGER model:
      - input_adapters.*  (TIGER_Residual only)
      - output_adapters.*

    Everything else (T5 encoder/decoder, shared embeddings, lm_head,
    semantic_pos, pos_embedding, emb_proj, etc.) is considered backbone.
    """
    return "input_adapters" in name or "output_adapters" in name


def _separate_params(model):
    """Separate model parameters into backbone and adapter groups.

    Returns:
        backbone_params: list of (name, param) for backbone params
        adapter_params:  list of (name, param) for adapter params
    """
    backbone_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if _is_adapter_param(name):
            adapter_params.append((name, param))
        else:
            backbone_params.append((name, param))
    return backbone_params, adapter_params


def _freeze_backbone(model):
    """Freeze all backbone parameters (disable gradients).

    Only adapter layers remain trainable. Saves GPU memory and compute
    during adapter warm-up phase.
    """
    for name, param in model.named_parameters():
        if not _is_adapter_param(name):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Phase 1 — Adapter Warm-up] Backbone FROZEN. "
          f"Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}%)")


def _unfreeze_all(model):
    """Unfreeze all parameters (enable gradients for joint fine-tuning)."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Phase 2 — Joint Fine-tuning] ALL params trainable. "
          f"Trainable: {trainable:,} / {total:,} params")


def _load_continue_checkpoint(model, checkpoint_path, device):
    """Load a TIGER or TIGER_Residual checkpoint as initialization for continue training.

    Loads matching weights with strict=False, allowing missing keys
    (adapter layers that don't exist in the base TIGER checkpoint).
    Reports loaded / missing / unexpected keys for transparency.

    Args:
        model: TIGER_Residual or Simple_Residual model
        checkpoint_path: path to .pt checkpoint file
        device: torch device

    Returns:
        The set of missing keys (adapter params initialized randomly).
    """
    print(f"\n{'='*70}")
    print(f"🔄 Continue Training: loading checkpoint from")
    print(f"   {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped training state
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    load_result = model.load_state_dict(state_dict, strict=False)

    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)

    # Classify missing keys
    missing_adapter = [k for k in missing if _is_adapter_param(k)]
    missing_backbone = [k for k in missing if not _is_adapter_param(k)]

    print(f"   ✓ Loaded {len(state_dict)} keys from checkpoint")
    if missing_adapter:
        print(f"   ℹ️  Missing adapter keys ({len(missing_adapter)}): "
              f"initialized randomly — expected for continue training")
        for k in sorted(missing_adapter)[:10]:
            print(f"      - {k}")
        if len(missing_adapter) > 10:
            print(f"      ... and {len(missing_adapter) - 10} more")
    if missing_backbone:
        print(f"   ⚠️  Missing backbone keys ({len(missing_backbone)}): "
              f"may indicate config mismatch")
        for k in sorted(missing_backbone)[:10]:
            print(f"      - {k}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys ({len(unexpected)}): "
              f"ignored (not in current model)")
        for k in sorted(unexpected)[:10]:
            print(f"      - {k}")

    print(f"{'='*70}\n")
    return missing


def _build_phase2_optimizer(model, base_lr, backbone_lr_factor, adapter_lr_factor,
                            weight_decay, backbone_params=None, adapter_params=None):
    """Build optimizer for Phase 2 (joint fine-tuning) with differential LRs.

    Backbone params use base_lr × backbone_lr_factor (typically lower).
    Adapter params use base_lr × adapter_lr_factor (typically 1.0 = same as base).

    Args:
        model: the model (or we can use pre-separated param lists)
        base_lr: base learning rate from trainer config
        backbone_lr_factor: multiplier for backbone lr
        adapter_lr_factor: multiplier for adapter lr
        weight_decay: weight decay for AdamW
        backbone_params: pre-separated backbone params (optional, auto-detect if None)
        adapter_params: pre-separated adapter params (optional, auto-detect if None)

    Returns:
        optimizer with two param groups
    """
    if backbone_params is None or adapter_params is None:
        bp, ap = _separate_params(model)
        backbone_params = [p for _, p in bp]
        adapter_params = [p for _, p in ap]

    optimizer = AdamW([
        {
            "params": backbone_params,
            "lr": base_lr * backbone_lr_factor,
        },
        {
            "params": adapter_params,
            "lr": base_lr * adapter_lr_factor,
        },
    ], weight_decay=weight_decay)

    print(f"[Phase 2 Optimizer] backbone_lr={base_lr * backbone_lr_factor:.6f}, "
          f"adapter_lr={base_lr * adapter_lr_factor:.6f}")
    return optimizer


# def first_failure_weighted_loss(
#     logits,          # [B, T, V]
#     labels,          # [B, T]  target token ids, -100 for padding
#     base_weight: float = 1.0,   # 成功分离时的最小权重
#     margin_temp: float = 1.0,   # 控制weight对margin的敏感度
# ):
#     """
#     Weight = softplus(-margin / temp) + base_weight
    
#     margin[i,t] = pos_prob[i,t] - max_{j!=i} P_i(labels[j,t])
#       margin >> 0 → 完全分开 → weight ≈ base_weight (小)
#       margin ≈ 0  → 刚好边界 → weight = softplus(0) + base ≈ 0.69 + base
#       margin << 0 → 完全未分开 → weight 很大
    
#     同时保留"第一个失败点之后截断"逻辑：
#       只有 t <= first_fail[i] 时才赋予动态weight，其余 t > first_fail 给 base_weight
#     """
#     B, T, V = logits.shape
#     device = logits.device

#     shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
#     shift_labels = labels[:, 1:].contiguous()        # [B, T-1]
#     T_shift = T - 1

#     probs = torch.softmax(shift_logits, dim=-1)      # [B, T-1, V]
#     valid_mask = (shift_labels != -100)              # [B, T-1]

#     safe_labels = shift_labels.clone()
#     safe_labels[~valid_mask] = 0

#     # pos_prob[i, t] = P_i(correct token at t)
#     pos_prob = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
#     pos_prob = pos_prob * valid_mask.float()

#     # neg_prob[i, t] = max_{j!=i} P_i(labels[j, t])
#     neg_prob = torch.zeros(B, T_shift, device=device)
#     for t in range(T_shift):
#         valid_at_t = valid_mask[:, t]
#         if valid_at_t.sum() == 0:
#             continue
#         labels_t = safe_labels[:, t]                 # [B]
#         prob_mat = probs[:, t, :][:, labels_t]       # [B, B]: prob_mat[i,j] = P_i(token_j)
#         prob_mat = prob_mat.masked_fill(
#             torch.eye(B, dtype=torch.bool, device=device), -1.0
#         )
#         neg_prob[:, t] = prob_mat.max(dim=1).values * valid_at_t.float()

#     # ── margin ───────────────────────────────────────────────────────────────
#     # margin[i,t]: 正例 vs 最难负例的概率差
#     # margin > 0 → 分开了; margin <= 0 → 没分开
#     margin = pos_prob - neg_prob                     # [B, T-1]

#     # ── 动态权重 ──────────────────────────────────────────────────────────────
#     # weight = softplus(-margin / temp) + base_weight
#     # softplus(x) = log(1 + exp(x))，平滑的 max(0, x)
#     # -margin越大（越分不开），weight越大
#     dynamic_weight = F.softplus(-margin / margin_temp) + base_weight  # [B, T-1]

#     # ── 截断：t > first_fail 的位置回退到 base_weight ────────────────────────
#     # 逻辑：第一个失败点之后的token已经在错误路径上，不需要再放大惩罚
#     failed = (margin <= 0) & valid_mask              # [B, T-1]
#     has_failure = failed.any(dim=1)                  # [B]
#     first_fail_idx = failed.float().argmax(dim=1)    # [B], 无failure时=0但has_failure=False

#     # 构造mask: position_mask[i,t] = True 当 t > first_fail[i]
#     positions = torch.arange(T_shift, device=device).unsqueeze(0).expand(B, -1)  # [B, T-1]
#     first_fail_2d = first_fail_idx.unsqueeze(1).expand(-1, T_shift)              # [B, T-1]
#     after_first_fail = (positions > first_fail_2d) & has_failure.unsqueeze(1)    # [B, T-1]

#     # 截断：first_fail之后的位置用base_weight
#     weight = torch.where(after_first_fail, 
#                          torch.full_like(dynamic_weight, base_weight), 
#                          dynamic_weight)

#     weight = weight * valid_mask.float()             # padding归零

#     # ── 加权 NTP loss ─────────────────────────────────────────────────────────
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
#     per_token_loss = loss_fct(
#         shift_logits.view(-1, V),
#         shift_labels.view(-1)
#     ).view(B, T_shift)

#     denom = weight[valid_mask].sum().clamp(min=1e-8)
#     total_loss = (per_token_loss * weight).sum() / denom

#     return


def train_epoch_simple_residual(
    epoch,
    train_dataloader,
    model,
    optimizer,
    device,
    scaler,
    scheduler,
    writer,
    seen_ids,
    n_semantic_codebook,
    n_codebook,
    method_config,
    item2sid,
    item_embedding,
):
    """Training epoch for Simple_Residual model (auxiliary codebook/residual losses)."""
    from .evaluation import model_forward_simple_residual, get_target_embed

    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)

    # Initialize accumulators
    loss = torch.tensor(0.0, device=device)
    hard_loss = 0.0
    codebook_loss_val = torch.tensor(0.0, device=device)
    grad_norm = 0.0
    embedding_loss = 0.0
    num_valid_batches = 0
    # Soft-label temperature tracking
    current_soft_label_temp = None
    soft_label_temp_progress = 0.0
    soft_label_temp_step = 0

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        outputs, _ = model_forward_simple_residual(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Extract losses from forward_residual
        hard_loss = outputs["sid_loss"]
        codebook_loss_val = outputs["codebook_loss"]
        cumulative_residual_loss_val = outputs.get("cumulative_residual_loss", 0.0)

        # Track current soft-label temperature for logging
        if "current_soft_label_temp" in outputs:
            current_soft_label_temp = outputs["current_soft_label_temp"]
        soft_label_temp_progress = outputs.get("soft_label_temp_progress", 0.0)
        soft_label_temp_step = outputs.get("soft_label_temp_step", 0)

        logits = outputs["logits"]  # [B, n_codebook, V]

        # NaN guard
        logits_has_nan = torch.isnan(logits).any().item()
        logits_has_inf = torch.isinf(logits).any().item()
        loss_is_nan = torch.isnan(outputs["loss"]).item() if isinstance(outputs["loss"], torch.Tensor) else False

        if logits_has_nan or logits_has_inf or loss_is_nan:
            print(f"\n[NaN 诊断] epoch={epoch}")
            print(f"  loss NaN? {loss_is_nan}")
            print(f"  logits NaN? {logits_has_nan}  logits inf? {logits_has_inf}")
            print(f"  → 跳过此 batch")
            continue

        loss = hard_loss * method_config["sid_loss_weight"]
        loss += codebook_loss_val * method_config.get("codebook_loss_weight", 1.0)
        loss += cumulative_residual_loss_val * method_config.get("cumulative_residual_loss_weight", 0.0)
        num_valid_batches += 1

        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = model.predicted_embedding
            _, logits_dense = get_target_embed(
                predicted_embedding, model, method_config, item_embedding
            )
            logits_label = batch["labels_ids"][:, 0].to(device) - 1
            supposed_sid_label = item2sid[logits_label.cpu()]
            assert (
                supposed_sid_label == batch["labels_sids"][:, :n_codebook].numpy()
            ).all()
            logits_dense[:, unseen_ids - 1] = -100
            embedding_loss = F.cross_entropy(logits_dense, logits_label)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    progress_bar.close()

    if num_valid_batches == 0:
        print(f"⚠️ No valid batches in epoch {epoch + 1} — all batches were skipped due to NaN/inf")

    logs = {
        "train/loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
        "train/sid_loss": hard_loss if isinstance(hard_loss, float) else hard_loss.item(),
        "train/codebook_loss": codebook_loss_val.item() if isinstance(codebook_loss_val, torch.Tensor) else codebook_loss_val,
        "train/cumulative_residual_loss": cumulative_residual_loss_val if isinstance(cumulative_residual_loss_val, float) else cumulative_residual_loss_val,
        "train/embedding_loss": embedding_loss,
    }
    if current_soft_label_temp is not None:
        logs["train/soft_label_temp"] = current_soft_label_temp
        logs["train/soft_label_temp_progress"] = soft_label_temp_progress
        logs["train/soft_label_temp_step"] = soft_label_temp_step
    # Log per-position codebook weight decay alphas
    per_position_alphas = outputs.get("per_position_alphas", None)
    if per_position_alphas is not None:
        for k, alpha_k in enumerate(per_position_alphas):
            logs[f"train/cb_weight_alpha_k{k}"] = alpha_k
        logs["train/cb_weight_decay_step"] = outputs.get("codebook_weight_decay_step", 0)
    writer.log(logs)

    return model


def evaluate_helper_simple_residual(
    model,
    device,
    val_dataloader_dict,
    val_unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    keyword="eval",
    KEYS=[10],
    RETRIEVE_KEY=[20, 40, 60, 80, 100],
):
    """
    Evaluation helper for Simple_Residual model.
    Uses standard evaluate (model.generate) instead of evaluate_residual,
    since generation is identical to base TIGER.
    """
    from .evaluation import (
        evaluate_dense_ids,
        evaluate_dense_sids,
        evaluate_simple_residual,
        generate_then_dense,
        get_target_embed,
    )

    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
            recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate_simple_residual(
                model,
                dataloader,
                all_semantic_ids,
                device,
                method_config=method_config,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        else:
            returned_cand = None
            returned_embd = None
        return logs, returned_cand, returned_embd

    def _dense_evaluate(logs, dataloader, name):
        if method_config["embedding_loss_weight"] > 0:
            if method_config["use_id"] == "item_id":
                recall_dict, ndcg_dict = evaluate_dense_ids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _unified_evaluate(logs, dataloader, returned_cand, returned_embd, name):
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            recall_dict, ndcg_dict = generate_then_dense(
                model,
                dataloader,
                val_unseen_semantic_ids,
                device,
                method_config=method_config,
                returned_cand=returned_cand,
                returned_embd=returned_embd,
                item2sid=item2sid,
                item_embedding=item_embedding,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for _retrieve_key in recall_dict.keys():
                for key in recall_dict[_retrieve_key].keys():
                    logs = add_log(
                        logs,
                        recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}",
                        name,
                    )
                    logs = add_log(
                        logs,
                        ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}",
                        name,
                    )
        return logs

    if "test" in keyword:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        logs, returned_cand_cold, returned_embd_cold = _evaluate(
            logs, val_dataloader_dict["cold_start"], f"genret_cold_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )
    else:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )

    if method_config["flag_use_output_embedding"] and "test" in keyword:
        logs = _dense_evaluate(
            logs, val_dataloader_dict["cold_start_embd"], f"dense_cold_{keyword}"
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["in_set"],
            returned_cand_in,
            returned_embd_in,
            f"uni_in_{keyword}",
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["cold_start"],
            returned_cand_cold,
            returned_embd_cold,
            f"uni_cold_{keyword}",
        )

    if (
        method_config["evaluation_method"] == "dense"
        and method_config["embedding_loss_weight"] > 0
    ):
        ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
    else:
        if method_config["sid_loss_weight"] == 0:
            ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
        else:
            ndcg_at_10 = logs[f"genret_in_{keyword}/NDCG@10"]

    return logs, ndcg_at_10


def train_simple_residual(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
    rqvae_codebook_weights=None,
    codebook_sizes=None,
):
    """
    Main training function for Simple_Residual.

    Same structure as train_tiger_residual but uses Simple_Residual model
    (no residual interleaving, only auxiliary codebook/residual losses).

    Args:
        rqvae_codebook_weights: list of [codebook_size, latent_size] tensors
                                 from RQ-VAE codebooks.
        codebook_sizes: list of per-level codebook sizes (for LETTER with variable sizes).
                        Falls back to config["RQ-VAE"]["code_book_size"] (int) for TIGER/LIGER.
    """
    from .simple_residual import Simple_Residual

    output_path = config["output_path"]
    if codebook_sizes is None:
        codebook_sizes = config["RQ-VAE"]["code_book_size"]
    if isinstance(codebook_sizes, int):
        codebook_size = codebook_sizes
    else:
        codebook_size = codebook_sizes[0]
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    config = config["TIGER"]
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    (
        training_data,
        val_data,
        test_data,
        unseen_val_data,
        unseen_test_data,
        seen_semantic_ids,
        val_unseen_semantic_ids,
        test_unseen_semantic_ids,
        max_last_semantic_ids,
        n_semantic_codebook,
        n_codebook,
        item2sid,
    ) = load_data(
        id_save_location,
        user_sequence,
        unseen_val,
        unseen_test,
        seen,
        item_embedding,
        method_config,
        max_length=config["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=max_items_per_seq,
    )
    all_semantic_ids = np.unique(
        np.concatenate(
            [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
            axis=0,
        ),
        axis=0,
    )
    unseen_semantic_ids = np.unique(
        np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
        axis=0,
    )

    if method_config["flag_use_output_embedding"]:
        item_embedding = item_embedding.to(device)

    unseen_val_dataset = CustomDataset(unseen_val_data)
    unseen_test_dataset = CustomDataset(unseen_test_data)
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    seen_semantic_ids = torch.from_numpy(seen_semantic_ids)
    val_unseen_semantic_ids = torch.from_numpy(val_unseen_semantic_ids)
    test_unseen_semantic_ids = torch.from_numpy(test_unseen_semantic_ids)
    all_semantic_ids = torch.from_numpy(all_semantic_ids)
    unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size

    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + sid_vocab_size + 2
        )
    else:
        this_vocab_size = sid_vocab_size + 2

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    t5_config = config["T5"]
    trainer_config = config["trainer"]
    model_config = T5Config(
        num_layers=t5_config["encoder_layers"],
        num_decoder_layers=t5_config["decoder_layers"],
        d_model=t5_config["d_model"],
        d_ff=t5_config["d_ff"],
        num_heads=t5_config["num_heads"],
        d_kv=t5_config["d_kv"],
        dropout_rate=t5_config["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config["feed_forward_proj"],
        n_positions=config["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    # Determine latent_size from codebook weights
    latent_size = t5_config["d_model"]  # default fallback
    if rqvae_codebook_weights is not None:
        latent_size = rqvae_codebook_weights[0].shape[-1]
    else:
        print(
            "⚠️ No codebook weights provided — using random initialization. "
            "Codebook loss may be less meaningful. "
            "Please provide rqvae_codebook_weights from a trained RQ-VAE."
        )

    # Simple_Residual-specific config
    codebook_loss_weight = method_config.get("codebook_loss_weight", 1.0)
    soft_label_K = method_config.get("soft_label_K", 0)
    soft_label_temperature = method_config.get("soft_label_temperature", 1.0)
    soft_label_temp_min = method_config.get("soft_label_temp_min", None)
    soft_label_temp_decay_steps = method_config.get("soft_label_temp_decay_steps", 10000)
    codebook_loss_weight_decay_steps = method_config.get("codebook_loss_weight_decay_steps", None)
    cumulative_residual_loss_weight = method_config.get("cumulative_residual_loss_weight", 0.0)
    cumulative_residual_loss_type = method_config.get("cumulative_residual_loss_type", "mse")
    cumulative_residual_loss_temperature = method_config.get("cumulative_residual_loss_temperature", 1.0)

    model = Simple_Residual(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
        rqvae_codebook_weights=rqvae_codebook_weights,
        codebook_size=codebook_size,
        latent_size=latent_size,
        codebook_loss_weight=codebook_loss_weight,
        soft_label_K=soft_label_K,
        soft_label_temperature=soft_label_temperature,
        soft_label_temp_min=soft_label_temp_min,
        soft_label_temp_decay_steps=soft_label_temp_decay_steps,
        codebook_loss_weight_decay_steps=codebook_loss_weight_decay_steps,
        cumulative_residual_loss_weight=cumulative_residual_loss_weight,
        cumulative_residual_loss_type=cumulative_residual_loss_type,
        cumulative_residual_loss_temperature=cumulative_residual_loss_temperature,
    ).to(device)

    # Set codebook offsets for LETTER tokenizer
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]

    # ── Continue Training Setup ──────────────────────────────────────────────
    # Two-phase schedule when continue_from_tiger_checkpoint is set:
    #   Phase 1 (adapter warm-up): backbone frozen, only adapters train
    #   Phase 2 (joint fine-tuning): all params trainable, differential LR
    continue_checkpoint = method_config.get("continue_from_tiger_checkpoint", None)
    adapter_warmup_steps = method_config.get("adapter_warmup_steps", 0)
    backbone_lr_factor = method_config.get("backbone_lr_factor", 0.1)
    adapter_lr_factor = method_config.get("adapter_lr_factor", 1.0)
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"
    backbone_is_frozen = False  # track current phase

    if continue_checkpoint and not os.path.exists(state_path):
        _load_continue_checkpoint(model, continue_checkpoint, device)
        if adapter_warmup_steps > 0:
            _freeze_backbone(model)
            backbone_is_frozen = True
            print(f"Phase 1 (adapter warm-up): {adapter_warmup_steps} steps, "
                  f"then Phase 2 (joint fine-tuning) for remaining steps.")
        else:
            print(f"Skipping Phase 1 (adapter_warmup_steps=0). "
                  f"Starting directly in Phase 2 (joint fine-tuning).")
            _unfreeze_all(model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader_embedding = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader_embedding = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader_embedding = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    val_dataloader_dict = {
        "in_set": val_dataloader,
        "in_set_embd": val_dataloader_embedding,
        "cold_start": unseen_val_dataloader,
        "cold_start_embd": unseen_val_dataloader_embedding,
    }
    test_dataloader_dict = {
        "in_set": test_dataloader,
        "in_set_embd": test_dataloader_embedding,
        "cold_start": unseen_test_dataloader,
        "cold_start_embd": unseen_test_dataloader_embedding,
    }

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})
    print(f"Total number of parameters: {total_params}")
    print(
        f"Total number of epochs: {int(np.ceil(total_steps / len(train_dataloader)))}"
    )

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        RETRIEVE_KEY = [10]

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"

    # ── Optimizer & Scheduler (phase-aware) ──────────────────────────────────
    if backbone_is_frozen:
        # Phase 1: Only adapter params in optimizer (backbone is frozen)
        adapter_only_params = [p for n, p in model.named_parameters()
                              if _is_adapter_param(n) and p.requires_grad]
        optimizer = AdamW(
            adapter_only_params,
            lr=trainer_config["lr"],
            weight_decay=trainer_config["weight_decay"],
        )
        # Phase 1 scheduler: warmup over adapter_warmup_steps
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=min(trainer_config["warmup_steps"], adapter_warmup_steps // 4),
                num_training_steps=adapter_warmup_steps,
            )
    elif continue_checkpoint and adapter_warmup_steps == 0:
        # Skip Phase 1, go directly to Phase 2 with differential LR
        backbone_params, adapter_params = _separate_params(model)
        optimizer = _build_phase2_optimizer(
            model,
            base_lr=trainer_config["lr"],
            backbone_lr_factor=backbone_lr_factor,
            adapter_lr_factor=adapter_lr_factor,
            weight_decay=trainer_config["weight_decay"],
            backbone_params=[p for _, p in backbone_params],
            adapter_params=[p for _, p in adapter_params],
        )
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=trainer_config["warmup_steps"],
                num_training_steps=total_steps,
            )
    else:
        # Standard training (no continue checkpoint)
        optimizer = AdamW(
            model.parameters(),
            lr=trainer_config["lr"],
            weight_decay=trainer_config["weight_decay"],
        )
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=trainer_config["warmup_steps"],
                num_training_steps=total_steps,
            )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # ── Resume from interrupted training (overrides continue checkpoint) ──
    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )
        state_dict = training_state["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg_10 = training_state["best_ndcg_10"]
        global_step = training_state["global_step"]
        best_epoch = training_state["best_epoch"]
        start_epoch = training_state["train_step"]
        # Restore phase state from checkpoint
        saved_phase = training_state.get("training_phase", 2)
        if saved_phase == 1 and global_step < adapter_warmup_steps:
            backbone_is_frozen = True
            _freeze_backbone(model)
        else:
            backbone_is_frozen = False
            _unfreeze_all(model)
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Load the model from: ", state_path)

    # ── Training Loop (with phase transition) ───────────────────────────────
    for epoch in range(
        start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))
    ):
        # ── Phase 1 → Phase 2 transition ─────────────────────────────────
        if backbone_is_frozen and global_step >= adapter_warmup_steps:
            print(f"\n{'='*70}")
            print(f"🔄 Phase transition: Adapter warm-up → Joint fine-tuning")
            print(f"   Completed {global_step} warm-up steps (target: {adapter_warmup_steps})")
            print(f"{'='*70}")

            _unfreeze_all(model)
            backbone_is_frozen = False

            bp, ap = _separate_params(model)
            optimizer = _build_phase2_optimizer(
                model,
                base_lr=trainer_config["lr"],
                backbone_lr_factor=backbone_lr_factor,
                adapter_lr_factor=adapter_lr_factor,
                weight_decay=trainer_config["weight_decay"],
                backbone_params=[p for _, p in bp],
                adapter_params=[p for _, p in ap],
            )

            remaining_steps = total_steps - global_step
            scheduler = None
            if trainer_config["scheduler"] != "none":
                scheduler = get_scheduler(
                    name=trainer_config["scheduler"],
                    optimizer=optimizer,
                    num_warmup_steps=min(trainer_config["warmup_steps"], remaining_steps // 10),
                    num_training_steps=remaining_steps,
                )
            print(f"   Remaining joint training steps: {remaining_steps}")
            print(f"   Backbone LR: {trainer_config['lr'] * backbone_lr_factor:.6f}")
            print(f"   Adapter  LR: {trainer_config['lr'] * adapter_lr_factor:.6f}")
            print(f"{'='*70}\n")

        model = train_epoch_simple_residual(
            epoch,
            train_dataloader,
            model,
            optimizer,
            device,
            scaler,
            scheduler,
            writer,
            seen,
            n_semantic_codebook,
            n_codebook,
            method_config,
            item2sid,
            item_embedding,
        )
        global_step += len(train_dataloader)

        # Log current training phase
        current_phase = 1 if backbone_is_frozen else 2

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:
            logs, ndcg_at_10 = evaluate_helper_simple_residual(
                model,
                device,
                val_dataloader_dict,
                unseen_semantic_ids,
                all_semantic_ids,
                item2sid,
                item_embedding,
                method_config,
                keyword="val",
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            logs["train/step"] = global_step
            logs["train/phase"] = current_phase

            if ndcg_at_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg_10,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "training_phase": current_phase,
            }
            torch.save(training_state, state_path)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            print("Finish because the patience run out.")
            break

    print("Testing...")

    model = Simple_Residual(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
        rqvae_codebook_weights=rqvae_codebook_weights,
        codebook_size=codebook_size,
        latent_size=latent_size,
        codebook_loss_weight=codebook_loss_weight,
        soft_label_K=soft_label_K,
        soft_label_temperature=soft_label_temperature,
        soft_label_temp_min=soft_label_temp_min,
        soft_label_temp_decay_steps=soft_label_temp_decay_steps,
        codebook_loss_weight_decay_steps=codebook_loss_weight_decay_steps,
        cumulative_residual_loss_weight=cumulative_residual_loss_weight,
        cumulative_residual_loss_type=cumulative_residual_loss_type,
        cumulative_residual_loss_temperature=cumulative_residual_loss_temperature,
    ).to(device)
    # Set codebook offsets for LETTER tokenizer
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    model.load_state_dict(torch.load(best_state_path), strict=False)

    logs, _ = evaluate_helper_simple_residual(
        model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
    )

    writer.log(logs)
    writer.finish()

    return total_loss, {}


# def first_failure_weighted_loss(
#     logits,
#     labels,
#     base_weight: float = 1.0,
#     margin_temp: float = 1.0,
#     gamma: float = 0,
# ):
#     B, T, V = logits.shape
#     device = logits.device

#     if T == 0:
#         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         return loss_fct(logits.view(-1, V), labels.view(-1)), {"failure_rate": 0.0, "first_fail_pos": -1}

#     probs = torch.softmax(logits, dim=-1)            # [B, T, V]
#     valid_mask = (labels != -100)                    # [B, T]

#     safe_labels = labels.clone()
#     safe_labels[~valid_mask] = 0  #[B,T]

#     pos_prob = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
#     pos_prob = pos_prob * valid_mask.float()

#     neg_prob = torch.zeros(B, T, device=device)
#     if B > 1:
#         for t in range(T):
#             valid_at_t = valid_mask[:, t]
#             if valid_at_t.sum() == 0:
#                 continue
#             labels_t = safe_labels[:, t]#[B]
#             prob_mat = probs[:, t, :][:, labels_t]   # [B, B]
#             # prob_mat = prob_mat.masked_fill(
#             #     torch.eye(B, dtype=torch.bool, device=device), 0.0
#             # )
#             same_token_mask = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1))  # [B, B]
#             prob_mat = prob_mat.masked_fill(same_token_mask, 0.0)
#             neg_prob[:, t] = prob_mat.max(dim=1).values * valid_at_t.float()

#     margin = pos_prob - neg_prob - gamma             # [B, T]

#     dynamic_weight = F.softplus(-margin / margin_temp) + base_weight  #[B,T]

#     failed = (margin <= 0) & valid_mask
#     has_failure = failed.any(dim=1)
#     first_fail_idx = failed.float().argmax(dim=1)

#     positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
#     first_fail_2d = first_fail_idx.unsqueeze(1).expand(-1, T)
#     after_first_fail = (positions > first_fail_2d) & has_failure.unsqueeze(1)

#     weight = torch.where(after_first_fail,
#                          torch.ones_like(dynamic_weight) * base_weight,
#                          dynamic_weight)
#     weight = weight * valid_mask.float()

#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
#     per_token_loss = loss_fct(
#         logits.view(-1, V),
#         labels.view(-1)
#     ).view(B, T)

#     denom = weight[valid_mask].sum().clamp(min=1e-8)
#     total_loss = (per_token_loss * weight).sum() / denom

#     return total_loss, {
#         "margin_mean": margin[valid_mask].mean().item(),
#         "weight_mean": weight[valid_mask].mean().item(),
#         "failure_rate": has_failure.float().mean().item(),
#         "first_fail_pos": first_fail_idx[has_failure].float().mean().item() if has_failure.any() else -1,
#     }


# def first_failure_weighted_loss(
#     logits,
#     labels,
#     base_weight: float = 1.0,
#     margin_temp: float = 1.0,
#     gamma: float = 0,
# ):
#     B, T, V = logits.shape
#     device = logits.device

#     if T == 0:
#         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         return loss_fct(logits.view(-1, V), labels.view(-1)), {"failure_rate": 0.0, "first_fail_pos": -1}

#     probs = torch.softmax(logits, dim=-1)            # [B, T, V]
#     valid_mask = (labels != -100)                    # [B, T]

#     safe_labels = labels.clone()
#     safe_labels[~valid_mask] = 0                     # [B, T]

#     pos_prob = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
#     pos_prob = pos_prob * valid_mask.float()

#     neg_prob = torch.zeros(B, T, device=device)
#     if B > 1:
#         for t in range(T):
#             valid_at_t = valid_mask[:, t]
#             if valid_at_t.sum() == 0:
#                 continue
#             labels_t = safe_labels[:, t]                         # [B]
#             prob_mat = probs[:, t, :][:, labels_t]               # [B, B]
#             same_token_mask = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1))  # [B, B]
#             prob_mat = prob_mat.masked_fill(same_token_mask, 0.0)
#             neg_prob[:, t] = prob_mat.max(dim=1).values * valid_at_t.float()

#     margin = pos_prob - neg_prob - gamma             # [B, T]

#     dynamic_weight = F.softplus(-margin / margin_temp) + base_weight  # [B, T]

#     # 找第一个失败点
#     failed = (margin <= 0) & valid_mask              # [B, T]
#     has_failure = failed.any(dim=1)                  # [B]
#     first_fail_idx = failed.float().argmax(dim=1)    # [B]  #返回第一个值 
    
#     # 构造 one-hot mask：只有 first_fail 位置为 True
#     positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)   # [B, T]
#     first_fail_2d = first_fail_idx.unsqueeze(1).expand(-1, T)                # [B, T]
#     at_first_fail = (positions == first_fail_2d) & has_failure.unsqueeze(1)  # [B, T]

#     # # 只有 first_fail 位置用 dynamic_weight，其余全用 base_weight
#     # weight = torch.where(at_first_fail, dynamic_weight,
#     #                      torch.ones_like(dynamic_weight) * base_weight)
#     # weight = weight * valid_mask.float()

#     weight = torch.where(failed, dynamic_weight,
#                         torch.ones_like(dynamic_weight) * base_weight)
#     weight = weight * valid_mask.float()

#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

#     per_token_loss = loss_fct(
#         logits.view(-1, V),
#         labels.view(-1)
#     ).view(B, T)

#     denom = weight[valid_mask].sum().clamp(min=1e-8)
#     total_loss = (per_token_loss * weight).sum() / denom
#     if random.random() < 0.01:
#         print("weights":weight[0])
#         sample_0_true_probs = probs[0, torch.arange(T), safe_labels[0]]
#         print("probs":sample_0_true_probs)
#         print("margin":margin[0])
#     return total_loss, {
#         "margin_mean": margin[valid_mask].mean().item(),
#         "weight_mean": weight[valid_mask].mean().item(),
#         "failure_rate": has_failure.float().mean().item(),
#         "first_fail_pos": first_fail_idx[has_failure].float().mean().item() if has_failure.any() else -1,
#     }

# def first_failure_weighted_loss(
#     logits,
#     labels,
#     base_weight: float = 1.0,
#     margin_temp: float = 1.0,
#     gamma: float = 0,
#     topk: int = 10,  # 本级batch内topk即算正确
# ):
#     B, T, V = logits.shape
#     device = logits.device

#     if T == 0:
#         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         return loss_fct(logits.view(-1, V), labels.view(-1)), {"failure_rate": 0.0, "first_fail_pos": -1}

#     probs = torch.softmax(logits, dim=-1)            # [B, T, V]
#     valid_mask = (labels != -100)                    # [B, T]

#     safe_labels = labels.clone()
#     safe_labels[~valid_mask] = 0                     # [B, T]

#     # -------------------------------------------------------------------------
#     # 🔥 核心：逐时间步t（逐级），计算【本级batch内】的相对排名
#     # 规则：每一级t，只和当前batch里这一步的token比，不跨级、不看全库
#     # -------------------------------------------------------------------------
#     batch_rel_rank_score = torch.zeros(B, T, device=device)  # 最终相对得分 [0~1]
#     batch_in_step_ranks = torch.zeros(B, T, device=device)   # 本级batch内排名

#     for t in range(T):
#         # 1. 取出当前步 t 的数据（逐级处理）
#         prob_t = probs[:, t, :]          # [B, V] 当前步概率
#         label_t = safe_labels[:, t]      # [B] 当前步标签
#         valid_t = valid_mask[:, t]       # [B] 当前步有效掩码

#         if not valid_t.any():
#             continue

#         # 2. 🔥 关键：统计【当前步 t + 当前batch内】出现的所有唯一token
#         batch_tokens_t = label_t[valid_t].unique()   # 本级batch内有效token [C_t]
#         C_t = batch_tokens_t.size(0)                 # 本级batch内token总数（每级都不一样）
#         if C_t <= 1:
#             batch_rel_rank_score[:, t] = 0.0
#             continue

#         # 3. 只在【本级batch内token】上计算概率分布 [B, C_t]
#         batch_prob_t = prob_t[:, batch_tokens_t]

#         # 4. 找到当前标签在【本级batch子集】中的索引
#         token_to_idx = {tok.item(): i for i, tok in enumerate(batch_tokens_t)}
#         label_idx = torch.tensor([token_to_idx[lb.item()] for lb in label_t], device=device)

#         # 5. 正例在本级batch内的概率
#         pos_prob_t = batch_prob_t[torch.arange(B), label_idx]

#         # 6. 🔥 计算：本级batch内的排名（只比batch里的token）
#         rank_t = (batch_prob_t > pos_prob_t.unsqueeze(-1)).sum(dim=-1) + 1  # [B]
#         batch_in_step_ranks[:, t] = rank_t

#         # 7. 🔥 相对得分 = (rank-1) / (本级batch内总token数 -1) → 0~1
#         batch_rel_rank_score[:, t] = (rank_t - 1) / (C_t - 1)

#     # 掩码无效位置
#     batch_rel_rank_score = batch_rel_rank_score * valid_mask.float()
#     batch_rel_rank_score = torch.clamp(batch_rel_rank_score, 0.0, 1.0)

#     # -------------------------------------------------------------------------
#     # 用【逐级batch内相对排名】计算动态权重（替换原来的margin逻辑）
#     # -------------------------------------------------------------------------
#     dynamic_weight = F.softplus(batch_rel_rank_score / margin_temp) + base_weight

#     # -------------------------------------------------------------------------
#     # 失败定义：本级batch内没进TopK = 失败（完全匹配beam search）
#     # -------------------------------------------------------------------------
#     is_topk_hit = (batch_in_step_ranks <= topk) & valid_mask
#     failed = (~is_topk_hit) & valid_mask

#     # -------------------------------------------------------------------------
#     # 原有逻辑：第一个失败点（完全不变）
#     # -------------------------------------------------------------------------
#     has_failure = failed.any(dim=1)
#     first_fail_idx = failed.float().argmax(dim=1)

#     # -------------------------------------------------------------------------
#     # 原有加权逻辑（完全不变）
#     # -------------------------------------------------------------------------
#     weight = torch.where(failed, dynamic_weight, base_weight)
#     weight = weight * valid_mask.float()

#     # -------------------------------------------------------------------------
#     # 损失计算（完全不变）
#     # -------------------------------------------------------------------------
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
#     per_token_loss = loss_fct(logits.view(-1, V), labels.view(-1)).view(B, T)

#     denom = weight[valid_mask].sum().clamp(min=1e-8)
#     total_loss = (per_token_loss * weight).sum() / denom

#     # -------------------------------------------------------------------------
#     # 调试打印（1%概率）
#     # -------------------------------------------------------------------------
#     if random.random() < 0.01:
#         print("="*60)
#         print(f"【逐级·Batch内排名】| TopK={topk}")
#         print(f"样本0 每级Batch内排名: {batch_in_step_ranks[0].long()}")
#         print(f"样本0 每级相对得分: {batch_rel_rank_score[0]}")
#         print(f"样本0 每级TopK命中: {is_topk_hit[0]}")
#         print(f"样本0 权重: {weight[0]}")
#         print("="*60)

#     return total_loss, {
#         "weight_mean": weight[valid_mask].mean().item(),
#         "failure_rate": has_failure.float().mean().item(),
#         "first_fail_pos": first_fail_idx[has_failure].float().mean().item() if has_failure.any() else -1,
#         "topk_hit_rate": is_topk_hit[valid_mask].float().mean().item(),
#         "avg_batch_rank": batch_in_step_ranks[valid_mask].mean().item(),
#     }

def first_failure_weighted_loss(
    logits,
    labels,
    base_weight: float = 1.0,
    margin_temp: float = 1.0,  # 控制对数边际的温度
    entropy_gamma: float = 2.0, # 🔥 新增：控制信息熵置信度的幂次敏感度
    topk: int = 10,  
):
    B, T, V = logits.shape
    device = logits.device

    if T == 0:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits.view(-1, V), labels.view(-1)), {"failure_rate": 0.0, "first_fail_pos": -1}

    probs = torch.softmax(logits.float(), dim=-1)   # ⚠️ cast to FP32 to avoid FP16 log(0) NaN
    valid_mask = (labels != -100)                    

    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0                     

    # 用于记录联合惩罚得分
    batch_complex_penalty = torch.zeros(B, T, device=device)  
    batch_in_step_ranks = torch.zeros(B, T, device=device)   
    
    # 监控指标
    batch_entropies = torch.zeros(B, T, device=device)
    batch_margins = torch.zeros(B, T, device=device)

    for t in range(T):
        prob_t = probs[:, t, :]          
        label_t = safe_labels[:, t]      
        valid_t = valid_mask[:, t]       

        if not valid_t.any():
            continue

        # ---------------------------------------------------------------------
        # 支柱 1：全局信息熵探索度评估 (Global Entropy Modulator)
        # 用整个词表计算真实熵，没有任何 detach，模型依靠自然梯度寻找平衡
        # ---------------------------------------------------------------------
        # H = - \sum p * log(p)
        entropy_t = -torch.sum(prob_t * torch.log(prob_t + 1e-8), dim=-1) # [B]
        max_entropy = math.log(V)
        norm_entropy_t = entropy_t / max_entropy # [0, 1]，越接近1越是在探索
        batch_entropies[:, t] = norm_entropy_t
        
        # 探索度越高 (norm_entropy->1)，confidence_factor 越趋近 0，宽容度极高
        # 盲目自信出错 (norm_entropy->0)，confidence_factor 越趋近 1，重拳出击
        confidence_factor = (1.0 - norm_entropy_t) ** entropy_gamma

        # ---------------------------------------------------------------------
        # 基础计算：选出本级 batch_tokens 用于计算秩（保留你原本的业务逻辑）
        # ---------------------------------------------------------------------
        batch_tokens_t = label_t[valid_t].unique()   
        C_t = batch_tokens_t.size(0)                 
        if C_t <= 1:
            continue

        batch_prob_t = prob_t[:, batch_tokens_t]
        token_to_idx = {tok.item(): i for i, tok in enumerate(batch_tokens_t)}
        label_idx = torch.tensor([token_to_idx[lb.item()] for lb in label_t], device=device)

        # 正例在本级batch内的概率
        pos_prob_t = batch_prob_t[torch.arange(B), label_idx]

        # ---------------------------------------------------------------------
        # 支柱 2：Beam Search 存活边界边际 (Log-Likelihood Survival Margin)
        # 不跟第 1 名比，而是跟第 K 名（存活线）比。算的是对数似然比差值。
        # ---------------------------------------------------------------------
        k_idx = min(topk, C_t) - 1
        topk_probs, _ = torch.topk(batch_prob_t, k=k_idx + 1, dim=-1)
        boundary_prob_t = topk_probs[:, -1] # 第 K 名的概率，即"存活线"

        pos_log_prob = torch.log(pos_prob_t + 1e-8)
        boundary_log_prob = torch.log(boundary_prob_t + 1e-8)
        
        # 如果正例概率 > 存活线概率，Log-Likelihood Ratio < 0，Softplus 后逼近于 0（不罚）
        # 如果正例概率 远低于 存活线概率，产生平滑递增的连续惩罚值
        survival_margin = F.softplus((boundary_log_prob - pos_log_prob) / margin_temp)
        batch_margins[:, t] = survival_margin

        # ---------------------------------------------------------------------
        # 支柱 3：对数平滑轶序惩罚 (Logarithmic Rank Penalty)
        # 完美解决“防震荡”但“不抛弃长尾”的需求
        # ---------------------------------------------------------------------
        rank_t = (batch_prob_t > pos_prob_t.unsqueeze(-1)).sum(dim=-1) + 1  
        batch_in_step_ranks[:, t] = rank_t

        # 超出 TopK 的部分才开始计算秩罚
        rank_diff = F.relu(rank_t - topk).float()
        
        # 核心：使用 Log2 函数。
        # 排 20 名（相差10），惩罚是 log2(1+2) = 1.58
        # 排 500 名（相差490），惩罚是 log2(1+98) = 6.62
        # 它永远在增长（梯度始终存在，模型永远想让1000变成500），但绝对不会造成线性爆炸震荡！
        rank_scale = topk / 2.0 + 1e-5  # 缩放因子，让前段坡度稍微平滑
        log_rank_penalty = torch.log2(1.0 + rank_diff / rank_scale)

        # ---------------------------------------------------------------------
        # 终极融合：三大机制乘法协同
        # 只有【掉出TopK】且【距离存活线极远】且【模型极度盲目自信】时，惩罚才会达到最大峰值
        # ---------------------------------------------------------------------
        batch_complex_penalty[:, t] = log_rank_penalty * survival_margin * confidence_factor


    # 掩码无效位置
    batch_complex_penalty = batch_complex_penalty * valid_mask.float()

    # 直接将连续计算的联合惩罚作为动态增量加到基础权重上
    dynamic_weight = base_weight + batch_complex_penalty

    is_topk_hit = (batch_in_step_ranks <= topk) & valid_mask
    failed = (~is_topk_hit) & valid_mask

    has_failure = failed.any(dim=1)
    first_fail_idx = failed.float().argmax(dim=1)

    # 这里的 dynamic_weight 已经在 TopK 命中的地方通过 ReLU(rank-topk) 自然化为 base_weight 了
    # 所以直接用 dynamic_weight 即可，保持全局梯度的连续性
    weight = dynamic_weight * valid_mask.float()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(logits.view(-1, V), labels.view(-1)).view(B, T)

    denom = weight[valid_mask].sum().clamp(min=1e-8)
    total_loss = (per_token_loss * weight).sum() / denom

    return total_loss, {
        "weight_mean": weight[valid_mask].mean().item(),
        "failure_rate": has_failure.float().mean().item(),
        "first_fail_pos": first_fail_idx[has_failure].float().mean().item() if has_failure.any() else -1,
        "topk_hit_rate": is_topk_hit[valid_mask].float().mean().item(),
        "avg_batch_rank": batch_in_step_ranks[valid_mask].mean().item(),
        "avg_entropy": batch_entropies[valid_mask].mean().item(), # 新增指标：监控模型整体认知状态
        "avg_survival_margin": batch_margins[valid_mask].mean().item(),
    }

def evaluate_helper(
    model,
    device,
    val_dataloader_dict,
    val_unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    keyword="eval",
    KEYS=[10],  # recall@k, k list
    RETRIEVE_KEY=[20, 40, 60, 80, 100],  # retrieve then rank
):
    """
    :param KEYS: the keys for the recall@k
    :param RETRIEVE_KEY: the keys for the retrieve then rank, only used for liger
    """

    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
            recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate(
                model,
                dataloader,
                all_semantic_ids,
                device,
                method_config=method_config,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        else:
            returned_cand = None
            returned_embd = None
        return logs, returned_cand, returned_embd

    def _dense_evaluate(logs, dataloader, name):
        if method_config["embedding_loss_weight"] > 0:
            if method_config["use_id"] == "item_id":
                recall_dict, ndcg_dict = evaluate_dense_ids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _unified_evaluate(logs, dataloader, returned_cand, returned_embd, name):
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            recall_dict, ndcg_dict = generate_then_dense(
                model,
                dataloader,
                val_unseen_semantic_ids,
                device,
                method_config=method_config,
                returned_cand=returned_cand,
                returned_embd=returned_embd,
                item2sid=item2sid,
                item_embedding=item_embedding,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for _retrieve_key in recall_dict.keys():
                for key in recall_dict[_retrieve_key].keys():
                    logs = add_log(
                        logs,
                        recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}",
                        name,
                    )
                    logs = add_log(
                        logs,
                        ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}",
                        name,
                    )
        return logs

    if "test" in keyword:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        logs, returned_cand_cold, returned_embd_cold = _evaluate(
            logs, val_dataloader_dict["cold_start"], f"genret_cold_{keyword}"
        )

        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )
    else:  # during training, do selected eval
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )

    if method_config["flag_use_output_embedding"] and "test" in keyword:
        logs = _dense_evaluate(
            logs, val_dataloader_dict["cold_start_embd"], f"dense_cold_{keyword}"
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["in_set"],
            returned_cand_in,
            returned_embd_in,
            f"uni_in_{keyword}",
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["cold_start"],
            returned_cand_cold,
            returned_embd_cold,
            f"uni_cold_{keyword}",
        )

    if (
        method_config["evaluation_method"] == "dense"
        and method_config["embedding_loss_weight"] > 0
    ):
        ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
    else:
        if method_config["sid_loss_weight"] == 0:
            ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
        else:
            ndcg_at_10 = logs[f"genret_in_{keyword}/NDCG@10"]

    return logs, ndcg_at_10


def train_epoch(
    epoch,
    train_dataloader,
    model,
    optimizer,
    device,
    scaler,
    scheduler,
    writer,
    seen_ids,
    n_semantic_codebook,
    n_codebook,
    method_config,
    item2sid,
    item_embedding,
):
    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)

    # Initialize accumulators so they are defined even if all batches are skipped
    loss = torch.tensor(0.0, device=device)
    hard_loss = 0.0
    codebook_loss_val = torch.tensor(0.0, device=device)
    grad_norm = 0.0
    first_failure_log = {}
    embedding_loss = 0.0
    num_valid_batches = 0

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        outputs, _ = model_forward(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Calculating the loss
        loss = 0
        hard_loss = outputs.loss
        loss += hard_loss * method_config["sid_loss_weight"]
        #获取预测embedding，与sid构建损失函数
        # pred_embedding=output.decoder_hidden_states[-1]# [batch_size, (n_tokens), n_embd]
        # logits=outputs.logits
        # loss = 0
        # hard_loss,first_failuer_log = first_failure_weighted_loss(logits,batch["labels_sids"].to(device))
        # loss += hard_loss * method_config["sid_loss_weight"]

        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = (
                model.predicted_embedding
            )  # [batch_size, (n_tokens), n_embd]，预测的向量
            _, logits = get_target_embed(
                predicted_embedding, model, method_config, item_embedding
            )
            
            logits_label = batch["labels_ids"][:, 0].to(device) - 1
            supposed_sid_label = item2sid[logits_label.cpu()]  # [batch_size, n_code]
            assert (
                supposed_sid_label == batch["labels_sids"][:, :n_codebook].numpy()
            ).all()
            # do not update the embedding for the unseen items
            logits[:, unseen_ids - 1] = -100
            embedding_loss = F.cross_entropy(logits, logits_label)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    progress_bar.close()

    logs = {
        "train/loss": loss.item(),
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
    }
    
    logs["train/sid_loss"] = hard_loss
    logs["train/embedding_loss"] = embedding_loss
    # for k,v in first_failuer_log.items():
    #     logs[f"train/{k}"]=v
    logs["train_ori_loss"]=outputs.loss
    writer.log(logs)

    return model


def train_tiger(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
    codebook_sizes=None,
):

    output_path = config["output_path"]
    # codebook_sizes: list of per-level codebook sizes (for LETTER with variable sizes)
    # Falls back to config["RQ-VAE"]["code_book_size"] (int) for TIGER/LIGER
    if codebook_sizes is None:
        codebook_sizes = config["RQ-VAE"]["code_book_size"]
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    ######### NOW RE-WRITE CONFIGS ###########
    config = config["TIGER"]
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    (
        training_data,
        val_data,
        test_data,
        unseen_val_data,
        unseen_test_data,
        seen_semantic_ids,
        val_unseen_semantic_ids,
        test_unseen_semantic_ids,
        max_last_semantic_ids,
        n_semantic_codebook,
        n_codebook,
        item2sid,
    ) = load_data(
        id_save_location,
        user_sequence,
        unseen_val,
        unseen_test,
        seen,
        item_embedding,
        method_config,
        max_length=config["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=max_items_per_seq,
    )
    all_semantic_ids = np.unique(
        np.concatenate(
            [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
            axis=0,
        ),
        axis=0,
    )  # [n_items, n_code]
    unseen_semantic_ids = np.unique(
        np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
        axis=0,
    )  # [n_items, n_code]

    # deal with the output embedding
    if method_config["flag_use_output_embedding"]:
        item_embedding = item_embedding.to(device)

    unseen_val_dataset = CustomDataset(unseen_val_data)
    unseen_test_dataset = CustomDataset(unseen_test_data)
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    seen_semantic_ids = torch.from_numpy(seen_semantic_ids)
    val_unseen_semantic_ids = torch.from_numpy(val_unseen_semantic_ids)
    test_unseen_semantic_ids = torch.from_numpy(test_unseen_semantic_ids)
    all_semantic_ids = torch.from_numpy(all_semantic_ids)
    unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

    # Check the fourth semantic ID size
    # For LETTER with variable codebook sizes per level, the total SID vocabulary
    # is the sum of all per-level codebook sizes (plus the extra collision-avoidance level).
    # For TIGER/LIGER with uniform codebook_size, this reduces to the original formula.
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size

    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + sid_vocab_size + 2
        )  # by default this is 3026
        # 2000 for user
    else:
        this_vocab_size = sid_vocab_size + 2

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    t5_config = config["T5"]
    trainer_config = config["trainer"]
    model_config = T5Config(
        num_layers=t5_config["encoder_layers"],
        num_decoder_layers=t5_config["decoder_layers"],
        d_model=t5_config["d_model"],
        d_ff=t5_config["d_ff"],
        num_heads=t5_config["num_heads"],
        d_kv=t5_config["d_kv"],
        dropout_rate=t5_config["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config["feed_forward_proj"],
        n_positions=config["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    # Initialize the model with the custom configuration
    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader_embedding = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    unseen_val_dataloader = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader_embedding = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader_embedding = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    val_dataloader_dict = {
        "in_set": val_dataloader,
        "in_set_embd": val_dataloader_embedding,
        "cold_start": unseen_val_dataloader,
        "cold_start_embd": unseen_val_dataloader_embedding,
    }
    test_dataloader_dict = {
        "in_set": test_dataloader,
        "in_set_embd": test_dataloader_embedding,
        "cold_start": unseen_test_dataloader,
        "cold_start_embd": unseen_test_dataloader_embedding,
    }

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})

    print(f"Total number of parameters: {total_params}")
    print(
        f"Total number of epochs: {int(np.ceil(total_steps / len(train_dataloader)))}"
    )

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        # then this is the liger method
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        # then this could be TIGER or dense method only
        RETRIEVE_KEY = [10]

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"

    optimizer = AdamW(
        model.parameters(),
        lr=trainer_config["lr"],
        weight_decay=trainer_config["weight_decay"],
    )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    scheduler = None
    if trainer_config["scheduler"] != "none":
        scheduler = get_scheduler(
            name=trainer_config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=trainer_config["warmup_steps"],
            num_training_steps=total_steps,
        )

    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )  # NOTE: change to cpu if OOM
        state_dict = training_state["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg_10 = training_state["best_ndcg_10"]
        global_step = training_state["global_step"]
        best_epoch = training_state["best_epoch"]
        start_epoch = training_state["train_step"]
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Load the model from: ", state_path)

    for epoch in range(
        start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))
    ):
        model = train_epoch(
            epoch,
            train_dataloader,
            model,
            optimizer,
            device,
            scaler,
            scheduler,
            writer,
            seen,
            n_semantic_codebook,
            n_codebook,
            method_config,
            item2sid,
            item_embedding,
        )
        global_step += len(train_dataloader)

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:

            logs, ndcg_at_10 = evaluate_helper(
                model,
                device,
                val_dataloader_dict,
                unseen_semantic_ids,
                all_semantic_ids,
                item2sid,
                item_embedding,
                method_config,
                keyword="val",
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            logs["train/step"] = global_step

            if ndcg_at_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            # save from time to time
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg_10,
                "global_step": global_step,
                "best_epoch": best_epoch,
            }
            torch.save(training_state, state_path)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            print("Finish because the patience run out.")
            break

    print("Testing...")

    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    model.load_state_dict(torch.load(best_state_path), strict=True)

    logs, _ = evaluate_helper(
        model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
    )

    writer.log(logs)
    writer.finish()

    return


# =============================================================================
# Soft Label Training Functions (TIGER_SoftLabel — no residual, soft labels)
# =============================================================================


def evaluate_helper_softlabel(
    model,
    device,
    val_dataloader_dict,
    val_unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    keyword="eval",
    KEYS=[10],
    RETRIEVE_KEY=[20, 40, 60, 80, 100],
):
    """
    Evaluation helper for TIGER_SoftLabel model.
    Uses standard TIGER evaluate (since generation is identical to TIGER — no residual).
    """

    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
            # TIGER_SoftLabel uses standard TIGER generation (no residual interleaving)
            # So we can reuse the same evaluate function
            recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate(
                model,
                dataloader,
                all_semantic_ids,
                device,
                method_config=method_config,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        else:
            returned_cand = None
            returned_embd = None
        return logs, returned_cand, returned_embd

    def _dense_evaluate(logs, dataloader, name):
        if method_config["embedding_loss_weight"] > 0:
            if method_config["use_id"] == "item_id":
                recall_dict, ndcg_dict = evaluate_dense_ids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _unified_evaluate(logs, dataloader, returned_cand, returned_embd, name):
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            recall_dict, ndcg_dict = generate_then_dense(
                model,
                dataloader,
                val_unseen_semantic_ids,
                device,
                method_config=method_config,
                returned_cand=returned_cand,
                returned_embd=returned_embd,
                item2sid=item2sid,
                item_embedding=item_embedding,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for _retrieve_key in recall_dict.keys():
                for key in recall_dict[_retrieve_key].keys():
                    logs = add_log(
                        logs,
                        recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}",
                        name,
                    )
                    logs = add_log(
                        logs,
                        ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}",
                        name,
                    )
        return logs

    if "test" in keyword:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        logs, returned_cand_cold, returned_embd_cold = _evaluate(
            logs, val_dataloader_dict["cold_start"], f"genret_cold_{keyword}"
        )

        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )
    else:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )

    if method_config["flag_use_output_embedding"] and "test" in keyword:
        logs = _dense_evaluate(
            logs, val_dataloader_dict["cold_start_embd"], f"dense_cold_{keyword}"
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["in_set"],
            returned_cand_in,
            returned_embd_in,
            f"uni_in_{keyword}",
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["cold_start"],
            returned_cand_cold,
            returned_embd_cold,
            f"uni_cold_{keyword}",
        )

    if (
        method_config["evaluation_method"] == "dense"
        and method_config["embedding_loss_weight"] > 0
    ):
        ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
    else:
        if method_config["sid_loss_weight"] == 0:
            ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
        else:
            ndcg_at_10 = logs[f"genret_in_{keyword}/NDCG@10"]

    return logs, ndcg_at_10


def train_epoch_softlabel(
    epoch,
    train_dataloader,
    model,
    optimizer,
    device,
    scaler,
    scheduler,
    writer,
    seen_ids,
    n_semantic_codebook,
    n_codebook,
    method_config,
    item2sid,
    item_embedding,
):
    """Training epoch for TIGER_SoftLabel model."""
    from .evaluation import get_target_embed

    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)

    # Initialize accumulators so they are defined even if all batches are skipped
    loss = torch.tensor(0.0, device=device)
    hard_loss = 0.0
    grad_norm = 0.0
    embedding_loss = 0.0
    num_valid_batches = 0

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        outputs, _ = model_forward_softlabel(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Extract losses from softlabel forward
        hard_loss = outputs["sid_loss"]
        logits = outputs["logits"]  # [B, n_codebook, V]

        # NaN guard
        logits_has_nan = torch.isnan(logits).any().item()
        logits_has_inf = torch.isinf(logits).any().item()
        sid_loss_is_nan = torch.isnan(outputs["loss"]).item() if isinstance(outputs["loss"], torch.Tensor) else False

        if logits_has_nan or logits_has_inf or sid_loss_is_nan:
            print(f"\n[NaN 诊断] epoch={epoch}")
            print(f"  forward_softlabel outputs['loss'] NaN? {sid_loss_is_nan}")
            print(f"  logits NaN? {logits_has_nan}  logits inf? {logits_has_inf}")
            if logits_has_nan:
                nan_count = torch.isnan(logits).sum().item()
                print(f"  ⚠️ NaN 来自 forward_softlabel 的 logits ({nan_count} 个 NaN)")
            if logits_has_inf:
                inf_count = torch.isinf(logits).sum().item()
                print(f"  ⚠️ inf 来自 forward_softlabel 的 logits ({inf_count} 个 inf)")
            print(f"  → 跳过此 batch")
            continue

        loss = hard_loss * method_config["sid_loss_weight"]
        num_valid_batches += 1

        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = model.predicted_embedding
            _, logits_dense = get_target_embed(
                predicted_embedding, model, method_config, item_embedding
            )
            logits_label = batch["labels_ids"][:, 0].to(device) - 1
            supposed_sid_label = item2sid[logits_label.cpu()]
            assert (
                supposed_sid_label == batch["labels_sids"][:, :n_codebook].numpy()
            ).all()
            logits_dense[:, unseen_ids - 1] = -100
            embedding_loss = F.cross_entropy(logits_dense, logits_label)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    progress_bar.close()

    if num_valid_batches == 0:
        print(f"⚠️ No valid batches in epoch {epoch + 1} — all batches were skipped due to NaN/inf")

    logs = {
        "train/loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
        "train/sid_loss": hard_loss if isinstance(hard_loss, float) else hard_loss.item(),
        "train/embedding_loss": embedding_loss,
    }
    writer.log(logs)

    return model


def train_tiger_softlabel(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
    rqvae_codebook_weights=None,
    codebook_sizes=None,
):
    """
    Main training function for TIGER_SoftLabel.

    Uses distance-based soft labels for SID prediction (from RQ-VAE codebook),
    but no residual interleaving. This isolates the soft label effect from
    the residual information effect.

    Args:
        rqvae_codebook_weights: list of [codebook_size, latent_size] tensors
                                 from RQ-VAE codebooks. Required for computing
                                 distance-based soft labels.
    """
    from .tiger_softlabel import TIGER_SoftLabel

    output_path = config["output_path"]
    # codebook_sizes: list of per-level codebook sizes (for LETTER with variable sizes)
    # Falls back to config["RQ-VAE"]["code_book_size"] (int) for TIGER/LIGER
    if codebook_sizes is None:
        codebook_sizes = config["RQ-VAE"]["code_book_size"]
    # Keep codebook_size as int for TIGER_SoftLabel (used for codebook weight dims)
    if isinstance(codebook_sizes, int):
        codebook_size = codebook_sizes
    else:
        codebook_size = codebook_sizes[0]  # default to first level's size
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    config = config["TIGER"]
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    (
        training_data,
        val_data,
        test_data,
        unseen_val_data,
        unseen_test_data,
        seen_semantic_ids,
        val_unseen_semantic_ids,
        test_unseen_semantic_ids,
        max_last_semantic_ids,
        n_semantic_codebook,
        n_codebook,
        item2sid,
    ) = load_data(
        id_save_location,
        user_sequence,
        unseen_val,
        unseen_test,
        seen,
        item_embedding,
        method_config,
        max_length=config["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=max_items_per_seq,
    )
    all_semantic_ids = np.unique(
        np.concatenate(
            [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
            axis=0,
        ),
        axis=0,
    )
    unseen_semantic_ids = np.unique(
        np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
        axis=0,
    )

    if method_config["flag_use_output_embedding"]:
        item_embedding = item_embedding.to(device)

    unseen_val_dataset = CustomDataset(unseen_val_data)
    unseen_test_dataset = CustomDataset(unseen_test_data)
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    seen_semantic_ids = torch.from_numpy(seen_semantic_ids)
    val_unseen_semantic_ids = torch.from_numpy(val_unseen_semantic_ids)
    test_unseen_semantic_ids = torch.from_numpy(test_unseen_semantic_ids)
    all_semantic_ids = torch.from_numpy(all_semantic_ids)
    unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

    # For LETTER with variable codebook sizes per level, the total SID vocabulary
    # is the sum of all per-level codebook sizes (plus the extra collision-avoidance level).
    # For TIGER/LIGER with uniform codebook_size, this reduces to the original formula.
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size

    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + sid_vocab_size + 2
        )
    else:
        this_vocab_size = sid_vocab_size + 2

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    t5_config = config["T5"]
    trainer_config = config["trainer"]
    model_config = T5Config(
        num_layers=t5_config["encoder_layers"],
        num_decoder_layers=t5_config["decoder_layers"],
        d_model=t5_config["d_model"],
        d_ff=t5_config["d_ff"],
        num_heads=t5_config["num_heads"],
        d_kv=t5_config["d_kv"],
        dropout_rate=t5_config["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config["feed_forward_proj"],
        n_positions=config["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    # Determine codebook weights and latent_size
    latent_size = t5_config["d_model"]  # default fallback
    if rqvae_codebook_weights is not None:
        latent_size = rqvae_codebook_weights[0].shape[-1]
    else:
        print(
            "⚠️ No codebook weights provided for TIGER_SoftLabel — "
            "using random initialization. Soft labels require codebook "
            "embeddings to compute distances! Please provide "
            "rqvae_codebook_weights from a trained RQ-VAE."
        )

    # Soft label config
    soft_label_K = method_config.get("soft_label_K", 0)
    soft_label_temperature = method_config.get("soft_label_temperature", 1.0)

    model = TIGER_SoftLabel(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
        rqvae_codebook_weights=rqvae_codebook_weights,
        codebook_size=codebook_size,
        latent_size=latent_size,
        soft_label_K=soft_label_K,
        soft_label_temperature=soft_label_temperature,
    ).to(device)

    # Set codebook offsets for LETTER tokenizer (variable codebook sizes per level).
    # This ensures _get_codebook_idx and _compute_soft_label_for_level correctly
    # map between SID tokens and codebook indices using cumulative offsets.
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)
    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]

    # ── Continue Training Setup ──────────────────────────────────────────────
    # Two-phase schedule when continue_from_tiger_checkpoint is set:
    #   Phase 1 (adapter warm-up): backbone frozen, only adapters train
    #   Phase 2 (joint fine-tuning): all params trainable, differential LR
    continue_checkpoint = method_config.get("continue_from_tiger_checkpoint", None)
    adapter_warmup_steps = method_config.get("adapter_warmup_steps", 0)
    backbone_lr_factor = method_config.get("backbone_lr_factor", 0.1)
    adapter_lr_factor = method_config.get("adapter_lr_factor", 1.0)
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"
    backbone_is_frozen = False  # track current phase

    if continue_checkpoint and not os.path.exists(state_path):
        _load_continue_checkpoint(model, continue_checkpoint, device)
        if adapter_warmup_steps > 0:
            _freeze_backbone(model)
            backbone_is_frozen = True
            print(f"Phase 1 (adapter warm-up): {adapter_warmup_steps} steps, "
                  f"then Phase 2 (joint fine-tuning) for remaining steps.")
        else:
            print(f"Skipping Phase 1 (adapter_warmup_steps=0). "
                  f"Starting directly in Phase 2 (joint fine-tuning).")
            _unfreeze_all(model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader_embedding = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader_embedding = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader_embedding = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    val_dataloader_dict = {
        "in_set": val_dataloader,
        "in_set_embd": val_dataloader_embedding,
        "cold_start": unseen_val_dataloader,
        "cold_start_embd": unseen_val_dataloader_embedding,
    }
    test_dataloader_dict = {
        "in_set": test_dataloader,
        "in_set_embd": test_dataloader_embedding,
        "cold_start": unseen_test_dataloader,
        "cold_start_embd": unseen_test_dataloader_embedding,
    }

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})
    print(f"Total number of parameters: {total_params}")
    print(
        f"Total number of epochs: {int(np.ceil(total_steps / len(train_dataloader)))}"
    )

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        RETRIEVE_KEY = [10]

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"

    optimizer = AdamW(
        model.parameters(),
        lr=trainer_config["lr"],
        weight_decay=trainer_config["weight_decay"],
    )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    scheduler = None
    if trainer_config["scheduler"] != "none":
        scheduler = get_scheduler(
            name=trainer_config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=trainer_config["warmup_steps"],
            num_training_steps=total_steps,
        )

    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )
        state_dict = training_state["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg_10 = training_state["best_ndcg_10"]
        global_step = training_state["global_step"]
        best_epoch = training_state["best_epoch"]
        start_epoch = training_state["train_step"]
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Load the model from: ", state_path)

    for epoch in range(
        start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))
    ):
        model = train_epoch_softlabel(
            epoch,
            train_dataloader,
            model,
            optimizer,
            device,
            scaler,
            scheduler,
            writer,
            seen,
            n_semantic_codebook,
            n_codebook,
            method_config,
            item2sid,
            item_embedding,
        )
        global_step += len(train_dataloader)

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:
            logs, ndcg_at_10 = evaluate_helper_softlabel(
                model,
                device,
                val_dataloader_dict,
                unseen_semantic_ids,
                all_semantic_ids,
                item2sid,
                item_embedding,
                method_config,
                keyword="val",
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            logs["train/step"] = global_step

            if ndcg_at_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg_10,
                "global_step": global_step,
                "best_epoch": best_epoch,
            }
            torch.save(training_state, state_path)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            print("Finish because the patience run out.")
            break

    print("Testing...")

    model = TIGER_SoftLabel(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
        rqvae_codebook_weights=rqvae_codebook_weights,
        codebook_size=codebook_size,
        latent_size=latent_size,
        soft_label_K=soft_label_K,
        soft_label_temperature=soft_label_temperature,
    ).to(device)
    # Set codebook offsets for LETTER tokenizer (variable codebook sizes per level).
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    model.load_state_dict(torch.load(best_state_path), strict=False)

    logs, _ = evaluate_helper_softlabel(
        model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
    )

    writer.log(logs)
    writer.finish()

    return


# =============================================================================
# Residual Decoder Training Functions
# =============================================================================


def evaluate_helper_residual(
    model,
    device,
    val_dataloader_dict,
    val_unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    keyword="eval",
    KEYS=[10],
    RETRIEVE_KEY=[20, 40, 60, 80, 100],
):
    """
    Evaluation helper for TIGER_Residual model.
    Uses evaluate_residual instead of evaluate for generative retrieval.
    """
    from .evaluation import (
        evaluate_dense_ids,
        evaluate_dense_sids,
        evaluate_residual,
        generate_then_dense,
        get_target_embed,
    )

    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
            recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate_residual(
                model,
                dataloader,
                all_semantic_ids,
                device,
                method_config=method_config,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        else:
            returned_cand = None
            returned_embd = None
        return logs, returned_cand, returned_embd

    def _dense_evaluate(logs, dataloader, name):
        if method_config["embedding_loss_weight"] > 0:
            if method_config["use_id"] == "item_id":
                recall_dict, ndcg_dict = evaluate_dense_ids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _unified_evaluate(logs, dataloader, returned_cand, returned_embd, name):
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            recall_dict, ndcg_dict = generate_then_dense(
                model,
                dataloader,
                val_unseen_semantic_ids,
                device,
                method_config=method_config,
                returned_cand=returned_cand,
                returned_embd=returned_embd,
                item2sid=item2sid,
                item_embedding=item_embedding,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for _retrieve_key in recall_dict.keys():
                for key in recall_dict[_retrieve_key].keys():
                    logs = add_log(
                        logs,
                        recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}",
                        name,
                    )
                    logs = add_log(
                        logs,
                        ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}",
                        name,
                    )
        return logs

    if "test" in keyword:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        logs, returned_cand_cold, returned_embd_cold = _evaluate(
            logs, val_dataloader_dict["cold_start"], f"genret_cold_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )
    else:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )

    if method_config["flag_use_output_embedding"] and "test" in keyword:
        logs = _dense_evaluate(
            logs, val_dataloader_dict["cold_start_embd"], f"dense_cold_{keyword}"
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["in_set"],
            returned_cand_in,
            returned_embd_in,
            f"uni_in_{keyword}",
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["cold_start"],
            returned_cand_cold,
            returned_embd_cold,
            f"uni_cold_{keyword}",
        )

    if (
        method_config["evaluation_method"] == "dense"
        and method_config["embedding_loss_weight"] > 0
    ):
        ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
    else:
        if method_config["sid_loss_weight"] == 0:
            ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
        else:
            ndcg_at_10 = logs[f"genret_in_{keyword}/NDCG@10"]

    return logs, ndcg_at_10


def train_epoch_residual(
    epoch,
    train_dataloader,
    model,
    optimizer,
    device,
    scaler,
    scheduler,
    writer,
    seen_ids,
    n_semantic_codebook,
    n_codebook,
    method_config,
    item2sid,
    item_embedding,
):
    """Training epoch for TIGER_Residual model."""
    from .evaluation import model_forward_residual, get_target_embed

    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)

    # Initialize accumulators so they are defined even if all batches are skipped
    loss = torch.tensor(0.0, device=device)
    hard_loss = 0.0
    codebook_loss_val = torch.tensor(0.0, device=device)
    grad_norm = 0.0
    first_failure_log = {}
    embedding_loss = 0.0
    num_valid_batches = 0
    # Non-TF statistics accumulators
    nontf_resid_pct_sum = 0.0
    nontf_ntp_pct_sum = 0.0
    # Soft-label temperature tracking
    current_soft_label_temp = None
    soft_label_temp_progress = 0.0
    soft_label_temp_step = 0

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        outputs, _ = model_forward_residual(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Extract losses from residual forward
        hard_loss = outputs["sid_loss"]
        codebook_loss_val = outputs["codebook_loss"]
        cumulative_residual_loss_val = outputs.get("cumulative_residual_loss", 0.0)

        # Accumulate non-TF statistics
        nontf_resid_pct_sum += outputs.get("nontf_resid_pct", 0.0)
        nontf_ntp_pct_sum += outputs.get("nontf_ntp_pct", 0.0)

        # Track current soft-label temperature for logging
        if "current_soft_label_temp" in outputs:
            current_soft_label_temp = outputs["current_soft_label_temp"]
        soft_label_temp_progress = outputs.get("soft_label_temp_progress", 0.0)
        soft_label_temp_step = outputs.get("soft_label_temp_step", 0)

        logits = outputs["logits"]  # [B, n_codebook, V]

        # print(logits)
        # ── NaN 诊断：精确定位 NaN 来源 ──
        logits_has_nan = torch.isnan(logits).any().item()
        logits_has_inf = torch.isinf(logits).any().item()
        sid_loss_is_nan = torch.isnan(outputs["loss"]).item() if isinstance(outputs["loss"], torch.Tensor) else False
        codebook_loss_is_nan = torch.isnan(codebook_loss_val).item() if isinstance(codebook_loss_val, torch.Tensor) else False

        if logits_has_nan or logits_has_inf or sid_loss_is_nan or codebook_loss_is_nan:
            print(f"\n[NaN 诊断] epoch={epoch}")
            print(f"  forward_residual outputs['loss'] NaN? {sid_loss_is_nan}")
            print(f"  forward_residual sid_loss  NaN? {torch.isnan(hard_loss).item() if isinstance(hard_loss, torch.Tensor) else 'scalar'}")
            print(f"  forward_residual codebook_loss NaN? {codebook_loss_is_nan}")
            print(f"  logits NaN? {logits_has_nan}  logits inf? {logits_has_inf}")
            if logits_has_nan:
                nan_count = torch.isnan(logits).sum().item()
                print(f"  ⚠️ NaN 来自 forward_residual 的 logits ({nan_count} 个 NaN)")
            if logits_has_inf:
                inf_count = torch.isinf(logits).sum().item()
                print(f"  ⚠️ inf 来自 forward_residual 的 logits ({inf_count} 个 inf)")
            print(f"  logits dtype={logits.dtype}, range=[{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print(f"  → 跳过此 batch")
            continue

        # Use standard CE loss from forward_residual (no first_failure_weighted_loss)
        loss = hard_loss * method_config["sid_loss_weight"]
        loss += codebook_loss_val * method_config.get("codebook_loss_weight", 1.0)
        loss += cumulative_residual_loss_val * method_config.get("cumulative_residual_loss_weight", 0.0)
        num_valid_batches += 1

        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = model.predicted_embedding
            _, logits_dense = get_target_embed(
                predicted_embedding, model, method_config, item_embedding
            )
            logits_label = batch["labels_ids"][:, 0].to(device) - 1
            supposed_sid_label = item2sid[logits_label.cpu()]
            assert (
                supposed_sid_label == batch["labels_sids"][:, :n_codebook].numpy()
            ).all()
            logits_dense[:, unseen_ids - 1] = -100
            embedding_loss = F.cross_entropy(logits_dense, logits_label)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    progress_bar.close()

    if num_valid_batches == 0:
        print(f"⚠️ No valid batches in epoch {epoch + 1} — all batches were skipped due to NaN/inf")

    # Compute average non-TF percentages over the epoch
    nontf_resid_pct_avg = nontf_resid_pct_sum / num_valid_batches if num_valid_batches > 0 else 0.0
    nontf_ntp_pct_avg = nontf_ntp_pct_sum / num_valid_batches if num_valid_batches > 0 else 0.0

    logs = {
        "train/loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
        "train/sid_loss": hard_loss if isinstance(hard_loss, float) else hard_loss.item(),
        "train/codebook_loss": codebook_loss_val.item() if isinstance(codebook_loss_val, torch.Tensor) else codebook_loss_val,
        "train/cumulative_residual_loss": cumulative_residual_loss_val if isinstance(cumulative_residual_loss_val, float) else cumulative_residual_loss_val,
        "train/embedding_loss": embedding_loss,
        "train/nontf_resid_pct": nontf_resid_pct_avg,
        "train/nontf_ntp_pct": nontf_ntp_pct_avg,
    }
    if current_soft_label_temp is not None:
        logs["train/soft_label_temp"] = current_soft_label_temp
        logs["train/soft_label_temp_progress"] = soft_label_temp_progress
        logs["train/soft_label_temp_step"] = soft_label_temp_step
    # Log per-position codebook weight decay alphas
    per_position_alphas = outputs.get("per_position_alphas", None)
    if per_position_alphas is not None:
        for k, alpha_k in enumerate(per_position_alphas):
            logs[f"train/cb_weight_alpha_k{k}"] = alpha_k
        logs["train/cb_weight_decay_step"] = outputs.get("codebook_weight_decay_step", 0)
    writer.log(logs)

    return model


def train_tiger_residual(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
    rqvae_codebook_weights=None,
    codebook_sizes=None,
):
    """
    Main training function for TIGER_Residual.

    Args:
        rqvae_codebook_weights: list of [codebook_size, latent_size] tensors
                                 from RQ-VAE codebooks. If None, codebooks will
                                 be initialized randomly (likely causes NaN).
        codebook_sizes: list of per-level codebook sizes (for LETTER with variable sizes).
                        Falls back to config["RQ-VAE"]["code_book_size"] (int) for TIGER/LIGER.
    """
    from .tiger_residual import TIGER_Residual

    output_path = config["output_path"]
    # codebook_sizes: list of per-level codebook sizes (for LETTER with variable sizes)
    # Falls back to config["RQ-VAE"]["code_book_size"] (int) for TIGER/LIGER
    if codebook_sizes is None:
        codebook_sizes = config["RQ-VAE"]["code_book_size"]
    # Keep codebook_size as int for TIGER_Residual (used for codebook weight dims)
    if isinstance(codebook_sizes, int):
        codebook_size = codebook_sizes
    else:
        codebook_size = codebook_sizes[0]  # default to first level's size
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    config = config["TIGER"]
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    (
        training_data,
        val_data,
        test_data,
        unseen_val_data,
        unseen_test_data,
        seen_semantic_ids,
        val_unseen_semantic_ids,
        test_unseen_semantic_ids,
        max_last_semantic_ids,
        n_semantic_codebook,
        n_codebook,
        item2sid,
    ) = load_data(
        id_save_location,
        user_sequence,
        unseen_val,
        unseen_test,
        seen,
        item_embedding,
        method_config,
        max_length=config["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=max_items_per_seq,
    )
    all_semantic_ids = np.unique(
        np.concatenate(
            [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
            axis=0,
        ),
        axis=0,
    )
    unseen_semantic_ids = np.unique(
        np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
        axis=0,
    )

    if method_config["flag_use_output_embedding"]:
        item_embedding = item_embedding.to(device)

    unseen_val_dataset = CustomDataset(unseen_val_data)
    unseen_test_dataset = CustomDataset(unseen_test_data)
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    seen_semantic_ids = torch.from_numpy(seen_semantic_ids)
    val_unseen_semantic_ids = torch.from_numpy(val_unseen_semantic_ids)
    test_unseen_semantic_ids = torch.from_numpy(test_unseen_semantic_ids)
    all_semantic_ids = torch.from_numpy(all_semantic_ids)
    unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

    # For LETTER with variable codebook sizes per level, the total SID vocabulary
    # is the sum of all per-level codebook sizes (plus the extra collision-avoidance level).
    # For TIGER/LIGER with uniform codebook_size, this reduces to the original formula.
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size
    flag_separate_bos_representation = method_config.get("flag_separate_bos_representation", False)
    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + sid_vocab_size + 2
        )
    else:
        this_vocab_size = sid_vocab_size + 2

    # When separating BOS representation from SID generation, we need an extra
    # special token (SID_START) to trigger SID generation. Add 1 to vocab size.
    if flag_separate_bos_representation:
        this_vocab_size += 1

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    # SID_START token: placed just before EOS in the vocabulary
    sid_start_token_id = this_vocab_size - 2 if flag_separate_bos_representation else None

    t5_config = config["T5"]
    trainer_config = config["trainer"]
    model_config = T5Config(
        num_layers=t5_config["encoder_layers"],
        num_decoder_layers=t5_config["decoder_layers"],
        d_model=t5_config["d_model"],
        d_ff=t5_config["d_ff"],
        num_heads=t5_config["num_heads"],
        d_kv=t5_config["d_kv"],
        dropout_rate=t5_config["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config["feed_forward_proj"],
        n_positions=config["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)


    # Determine codebook weights and latent_size
    latent_size = t5_config["d_model"]  # default fallback
    if rqvae_codebook_weights is not None:
        latent_size = rqvae_codebook_weights[0].shape[-1]
    else:
        print(
            "⚠️ No codebook weights provided — using random initialization. "
            "This is VERY LIKELY to cause NaN in residual decoder! "
            "Please provide rqvae_codebook_weights from a trained RQ-VAE."
        )

    # Residual-specific config
    codebook_loss_weight = method_config.get("codebook_loss_weight", 1.0)
    num_residual_levels = method_config.get(
        "num_residual_levels", n_semantic_codebook - 1
    )
    soft_label_K = method_config.get("soft_label_K", 0)
    soft_label_temperature = method_config.get("soft_label_temperature", 1.0)
    soft_label_temp_min = method_config.get("soft_label_temp_min", None)
    soft_label_temp_decay_steps = method_config.get("soft_label_temp_decay_steps", 10000)
    codebook_loss_weight_decay_steps = method_config.get("codebook_loss_weight_decay_steps", None)
    cumulative_residual_loss_weight = method_config.get("cumulative_residual_loss_weight", 0.0)
    cumulative_residual_loss_type = method_config.get("cumulative_residual_loss_type", "mse")
    cumulative_residual_loss_temperature = method_config.get("cumulative_residual_loss_temperature", 1.0)
    resid_nontf_ratio = method_config.get("resid_nontf_ratio", 0.0)
    ntp_nontf_ratio = method_config.get("ntp_nontf_ratio", 0.0)

    model = TIGER_Residual(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
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
    ).to(device)

    # Set codebook offsets for LETTER tokenizer (variable codebook sizes per level).
    # This ensures _get_codebook_idx and _compute_soft_label_for_level correctly
    # map between SID tokens and codebook indices using cumulative offsets.
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)
    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]

    # ── Continue Training Setup ──────────────────────────────────────────────
    # Two-phase schedule when continue_from_tiger_checkpoint is set:
    #   Phase 1 (adapter warm-up): backbone frozen, only adapters train
    #   Phase 2 (joint fine-tuning): all params trainable, differential LR
    continue_checkpoint = method_config.get("continue_from_tiger_checkpoint", None)
    adapter_warmup_steps = method_config.get("adapter_warmup_steps", 0)
    backbone_lr_factor = method_config.get("backbone_lr_factor", 0.1)
    adapter_lr_factor = method_config.get("adapter_lr_factor", 1.0)
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"
    backbone_is_frozen = False  # track current phase

    if continue_checkpoint and not os.path.exists(state_path):
        _load_continue_checkpoint(model, continue_checkpoint, device)
        if adapter_warmup_steps > 0:
            _freeze_backbone(model)
            backbone_is_frozen = True
            print(f"Phase 1 (adapter warm-up): {adapter_warmup_steps} steps, "
                  f"then Phase 2 (joint fine-tuning) for remaining steps.")
        else:
            print(f"Skipping Phase 1 (adapter_warmup_steps=0). "
                  f"Starting directly in Phase 2 (joint fine-tuning).")
            _unfreeze_all(model)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader_embedding = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader_embedding = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader_embedding = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    val_dataloader_dict = {
        "in_set": val_dataloader,
        "in_set_embd": val_dataloader_embedding,
        "cold_start": unseen_val_dataloader,
        "cold_start_embd": unseen_val_dataloader_embedding,
    }
    test_dataloader_dict = {
        "in_set": test_dataloader,
        "in_set_embd": test_dataloader_embedding,
        "cold_start": unseen_test_dataloader,
        "cold_start_embd": unseen_test_dataloader_embedding,
    }

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})
    print(f"Total number of parameters: {total_params}")
    print(
        f"Total number of epochs: {int(np.ceil(total_steps / len(train_dataloader)))}"
    )

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        RETRIEVE_KEY = [10]

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1

    # ── Optimizer & Scheduler (phase-aware) ──────────────────────────────────
    if backbone_is_frozen:
        # Phase 1: Only adapter params in optimizer (backbone is frozen)
        adapter_only_params = [p for n, p in model.named_parameters()
                              if _is_adapter_param(n) and p.requires_grad]
        optimizer = AdamW(
            adapter_only_params,
            lr=trainer_config["lr"],
            weight_decay=trainer_config["weight_decay"],
        )
        # Phase 1 scheduler: warmup over adapter_warmup_steps
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=min(trainer_config["warmup_steps"], adapter_warmup_steps // 4),
                num_training_steps=adapter_warmup_steps,
            )
    elif continue_checkpoint and adapter_warmup_steps == 0:
        # Skip Phase 1, go directly to Phase 2 with differential LR
        bp, ap = _separate_params(model)
        optimizer = _build_phase2_optimizer(
            model,
            base_lr=trainer_config["lr"],
            backbone_lr_factor=backbone_lr_factor,
            adapter_lr_factor=adapter_lr_factor,
            weight_decay=trainer_config["weight_decay"],
            backbone_params=[p for _, p in bp],
            adapter_params=[p for _, p in ap],
        )
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=trainer_config["warmup_steps"],
                num_training_steps=total_steps,
            )
    else:
        # Standard training (no continue checkpoint)
        optimizer = AdamW(
            model.parameters(),
            lr=trainer_config["lr"],
            weight_decay=trainer_config["weight_decay"],
        )
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"],
                optimizer=optimizer,
                num_warmup_steps=trainer_config["warmup_steps"],
                num_training_steps=total_steps,
            )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # ── Resume from interrupted training (overrides continue checkpoint) ──────
    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )
        state_dict = training_state["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg_10 = training_state["best_ndcg_10"]
        global_step = training_state["global_step"]
        best_epoch = training_state["best_epoch"]
        start_epoch = training_state["train_step"]
        # Restore phase state from checkpoint
        saved_phase = training_state.get("training_phase", 2)
        if saved_phase == 1 and global_step < adapter_warmup_steps:
            backbone_is_frozen = True
            _freeze_backbone(model)
        else:
            backbone_is_frozen = False
            _unfreeze_all(model)
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Load the model from: ", state_path)

    # ── Training Loop (with phase transition) ───────────────────────────────
    for epoch in range(
        start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))
    ):
        # ── Phase 1 → Phase 2 transition ─────────────────────────────────────
        # When we've completed adapter_warmup_steps, transition to joint training
        if backbone_is_frozen and global_step >= adapter_warmup_steps:
            print(f"\n{'='*70}")
            print(f"🔄 Phase transition: Adapter warm-up → Joint fine-tuning")
            print(f"   Completed {global_step} warm-up steps (target: {adapter_warmup_steps})")
            print(f"{'='*70}")

            # Unfreeze all parameters
            _unfreeze_all(model)
            backbone_is_frozen = False

            # Rebuild optimizer with two param groups (differential LR)
            bp, ap = _separate_params(model)
            optimizer = _build_phase2_optimizer(
                model,
                base_lr=trainer_config["lr"],
                backbone_lr_factor=backbone_lr_factor,
                adapter_lr_factor=adapter_lr_factor,
                weight_decay=trainer_config["weight_decay"],
                backbone_params=[p for _, p in bp],
                adapter_params=[p for _, p in ap],
            )

            # Rebuild scheduler for Phase 2
            remaining_steps = total_steps - global_step
            scheduler = None
            if trainer_config["scheduler"] != "none":
                scheduler = get_scheduler(
                    name=trainer_config["scheduler"],
                    optimizer=optimizer,
                    num_warmup_steps=min(trainer_config["warmup_steps"], remaining_steps // 10),
                    num_training_steps=remaining_steps,
                )
            print(f"   Remaining joint training steps: {remaining_steps}")
            print(f"   Backbone LR: {trainer_config['lr'] * backbone_lr_factor:.6f}")
            print(f"   Adapter  LR: {trainer_config['lr'] * adapter_lr_factor:.6f}")
            print(f"{'='*70}\n")

        model = train_epoch_residual(
            epoch,
            train_dataloader,
            model,
            optimizer,
            device,
            scaler,
            scheduler,
            writer,
            seen,
            n_semantic_codebook,
            n_codebook,
            method_config,
            item2sid,
            item_embedding,
        )
        global_step += len(train_dataloader)

        # Log current training phase
        current_phase = 1 if backbone_is_frozen else 2
        phase_name = "adapter_warmup" if backbone_is_frozen else "joint_finetuning"

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:
            logs, ndcg_at_10 = evaluate_helper_residual(
                model,
                device,
                val_dataloader_dict,
                unseen_semantic_ids,
                all_semantic_ids,
                item2sid,
                item_embedding,
                method_config,
                keyword="val",
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            logs["train/step"] = global_step
            logs["train/phase"] = current_phase

            if ndcg_at_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg_10,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "training_phase": current_phase,  # save phase for resume
            }
            torch.save(training_state, state_path)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            print("Finish because the patience run out.")
            break

    print("Testing...")

    model = TIGER_Residual(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
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
    ).to(device)
    # Set codebook offsets for LETTER tokenizer (variable codebook sizes per level).
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    model.load_state_dict(torch.load(best_state_path), strict=False)

    # Disable non-TF for evaluation (use full teacher forcing context)
    model.resid_nontf_ratio = 0.0
    model.ntp_nontf_ratio = 0.0

    logs, _ = evaluate_helper_residual(
        model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
    )

    writer.log(logs)
    writer.finish()

    return
