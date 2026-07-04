#!/usr/bin/env python3
"""
Custom dataset training entry point for TIGER / TIGER_Residual.

Supports:
  - Pre-split data files (train/val/test × all/normal/surprise)
  - Tab-separated sequence files (uid\\titem1\\titem2\\t...)
  - Item embeddings from .pt file (built by build_embedding.py)
  - Auto-training RQ-VAE if SID file doesn't exist (with wandb logging)
  - Evaluation on all 3 subsets with Recall, NDCG @ 5, 10, 20, 100

Workflow (2 steps):
  Step 0:  build_embedding.py     →  item_embedding.pt  (external, one-time)
  Step 1:  run_custom.py          →  RQ-VAE training (if needed) + TIGER training + eval

  If --sid_file doesn't exist, run_custom.py automatically trains RQ-VAE
  and saves the SID pickle + codebook weights — just like run.py.
  RQ-VAE training uses the same wandb writer as the main pipeline.

Usage (TIGER base model):
  python run_custom.py \
    --data_dir ./data/custom/ \
    --embedding_file ./data/custom/item_embedding.pt \
    --sid_file ./ID_generation/ID/custom_semantic_0.pkl

Usage (auto-train RQ-VAE if sid_file missing):
  python run_custom.py \
    --data_dir ./data/custom/ \
    --embedding_file ./data/custom/item_embedding.pt \
    --sid_file ./ID_generation/ID/custom_semantic_0.pkl \
    --rqvae_epochs 300 \
    --wandb_mode online

Usage (TIGER_Residual):
  python run_custom.py \
    --data_dir ./data/custom/ \
    --embedding_file ./data/custom/item_embedding.pt \
    --sid_file ./ID_generation/ID/custom_semantic_0.pkl \
    --use_residual_decoder True

All method-specific flags from the original pipeline are supported:
  --use_residual_decoder, --use_simple_residual, --use_softlabel_sid,
  --soft_label_K, --soft_label_temperature, etc.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load_custom_data import (
    load_sequence_file,
    load_item_embedding,
    compute_id_split,
    load_custom_data,
)
from src.training import (
    _is_adapter_param,
    _separate_params,
    _freeze_backbone,
    _unfreeze_all,
    _load_continue_checkpoint,
    _build_phase2_optimizer,
    train_epoch,
    train_epoch_residual,
    train_epoch_simple_residual,
    train_epoch_softlabel,
)
from src.evaluation import evaluate, evaluate_residual, evaluate_simple_residual
from src.tiger_residual import TIGER_Residual
from src.tiger import TIGER
from utils import set_seed, CustomDataset, setup_logging

from transformers import T5Config
from torch.optim import AdamW
from transformers import get_scheduler


# ---------------------------------------------------------------------------
# Custom evaluation helper — evaluates ALL 3 subsets
# ---------------------------------------------------------------------------

def evaluate_helper_custom(
    model,
    device,
    dataloader_dict,
    all_semantic_ids,
    method_config,
    keyword="eval",
    KEYS=[5, 10, 20, 100],
    RETRIEVE_KEY=[100],
    use_residual=False,
    use_simple_residual=False,
):
    """
    Evaluate on all 3 subsets (all, normal, surprise) with Recall, NDCG.

    Unlike the original evaluate_helper which only has in_set/cold_start,
    this evaluates each subset independently and logs metrics per subset.

    Args:
        dataloader_dict: dict of subset_name -> dataloader
            e.g., {"all": dl, "normal": dl, "surprise": dl}
        keyword: "val" or "test" for logging prefix
        KEYS: list of K values for Recall@K, NDCG@K
        RETRIEVE_KEY: beam search width
        use_residual: use evaluate_residual (TIGER_Residual)
        use_simple_residual: use evaluate_simple_residual
    """
    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    best_ndcg_key = KEYS[1] if len(KEYS) > 1 else KEYS[0]  # default: NDCG@10
    ndcg_at_best = -0.01

    for subset_name, dataloader in dataloader_dict.items():
        if len(dataloader) == 0:
            print(f"  Skipping empty subset '{subset_name}'")
            continue

        log_prefix = f"{subset_name}_{keyword}"

        if use_residual:
            recall_dict, ndcg_dict, _, _ = evaluate_residual(
                model, dataloader, all_semantic_ids, device,
                method_config=method_config, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
            )
        elif use_simple_residual:
            recall_dict, ndcg_dict, _, _ = evaluate_simple_residual(
                model, dataloader, all_semantic_ids, device,
                method_config=method_config, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
            )
        else:
            recall_dict, ndcg_dict, _, _ = evaluate(
                model, dataloader, all_semantic_ids, device,
                method_config=method_config, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
            )

        for key in recall_dict.keys():
            logs = add_log(logs, recall_dict[key], f"Recall@{key}", log_prefix)
            logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", log_prefix)

        # Track best NDCG from the "all" subset
        if subset_name == "all" and best_ndcg_key in ndcg_dict:
            ndcg_at_best = torch.tensor(ndcg_dict[best_ndcg_key]).mean().item()

    # If "all" subset wasn't evaluated, use first available
    if ndcg_at_best < 0:
        for subset_name in dataloader_dict:
            log_key = f"{subset_name}_{keyword}/NDCG@{best_ndcg_key}"
            if log_key in logs:
                ndcg_at_best = logs[log_key].item()
                break

    return logs, ndcg_at_best


# ---------------------------------------------------------------------------
# Main training functions
# ---------------------------------------------------------------------------

def train_tiger_custom(
    config_dict,
    method_config,
    sequences_dict,
    id_split,
    item_embedding,
    id_save_location,
    device,
    codebook_sizes=None,
):
    """Train TIGER base model on custom dataset."""
    output_path = config_dict["output_path"]

    if codebook_sizes is None:
        codebook_sizes = config_dict["RQ-VAE"]["code_book_size"]
    max_items_per_seq = config_dict.get("max_items_per_seq", 20)

    writer = setup_logging(config_dict)

    # Load custom data
    datasets, semantic_info = load_custom_data(
        id_save_location,
        sequences_dict,
        id_split,
        item_embedding,
        method_config,
        max_length=config_dict["TIGER"]["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=max_items_per_seq,
    )

    # Semantic ID arrays
    all_semantic_ids = np.unique(
        np.concatenate([
            semantic_info["seen_semantic_ids"],
            semantic_info["val_unseen_semantic_ids"],
            semantic_info["test_unseen_semantic_ids"],
        ], axis=0),
        axis=0,
    )
    all_semantic_ids = torch.from_numpy(all_semantic_ids)

    n_semantic_codebook = semantic_info["n_semantic_codebook"]
    n_codebook = semantic_info["n_codebook"]
    max_last_semantic_ids = semantic_info["max_last_semantic_ids"]

    # Vocab size
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size

    if method_config.get("include_user_id", False):
        this_vocab_size = 2000 + sid_vocab_size + 2
    else:
        this_vocab_size = sid_vocab_size + 2

    # Build T5 config
    t5_config = config_dict["TIGER"]["T5"]
    trainer_config = config_dict["TIGER"]["trainer"]
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
        n_positions=config_dict["TIGER"]["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config.get("flag_use_output_embedding", False),
        flag_use_learnable_text_embed=method_config.get("flag_add_input_embedding", False),
        embedding_head_dict=method_config.get("embedding_head_dict", {}),
    ).to(device)

    # Dataloaders
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]
    eval_keys = config_dict.get("eval_keys", [5, 10, 20, 100])
    retrieve_key = [max(eval_keys)]

    # Train on "all" training data
    train_dataset = CustomDataset(datasets["train_all"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Val subsets
    val_dataloaders = {}
    for cat in ["all", "normal", "surprise"]:
        key = f"val_{cat}"
        if key in datasets:
            val_dataloaders[cat] = DataLoader(
                CustomDataset(datasets[key]), batch_size=eval_batch_size, shuffle=False
            )

    # Test subsets
    test_dataloaders = {}
    for cat in ["all", "normal", "surprise"]:
        key = f"test_{cat}"
        if key in datasets:
            test_dataloaders[cat] = DataLoader(
                CustomDataset(datasets[key]), batch_size=eval_batch_size, shuffle=False
            )

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})
    print(f"Total parameters: {total_params}")
    print(f"Training epochs: {int(np.ceil(trainer_config['steps'] / len(train_dataloader)))}")

    total_steps = trainer_config["steps"]
    best_ndcg = -0.01
    best_epoch = 0
    global_step = 0
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

    # Resume from checkpoint
    if os.path.exists(state_path):
        training_state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(training_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg = training_state.get("best_ndcg_10", -0.01)
        global_step = training_state["global_step"]
        best_epoch = training_state.get("best_epoch", 0)
        start_epoch = training_state["train_step"]
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed from {state_path}, step={global_step}")

    # Training loop
    for epoch in range(start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))):
        model = train_epoch(
            epoch, train_dataloader, model, optimizer, device, scaler, scheduler,
            writer, id_split["seen"], n_semantic_codebook, n_codebook,
            method_config, semantic_info["item2sid"], item_embedding,
        )
        global_step += len(train_dataloader)

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:
            logs, ndcg_val = evaluate_helper_custom(
                model, device, val_dataloaders, all_semantic_ids, method_config,
                keyword="val", KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
            )
            logs["train/step"] = global_step

            if ndcg_val > best_ndcg:
                best_ndcg = ndcg_val
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg,
                "global_step": global_step,
                "best_epoch": best_epoch,
            }
            torch.save(training_state, state_path)

        if (best_epoch + trainer_config["patience"] < epoch) and global_step > trainer_config["warmup_steps"]:
            print("Early stopping (patience exhausted)")
            break

    # Test
    print("Testing...")
    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config.get("flag_use_output_embedding", False),
        flag_use_learnable_text_embed=method_config.get("flag_add_input_embedding", False),
        embedding_head_dict=method_config.get("embedding_head_dict", {}),
    ).to(device)
    model.load_state_dict(torch.load(best_state_path), strict=True)

    logs, _ = evaluate_helper_custom(
        model, device, test_dataloaders, all_semantic_ids, method_config,
        keyword="test", KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
    )
    writer.log(logs)
    writer.finish()


def train_tiger_residual_custom(
    config_dict,
    method_config,
    sequences_dict,
    id_split,
    item_embedding,
    id_save_location,
    device,
    rqvae_codebook_weights=None,
    codebook_sizes=None,
):
    """Train TIGER_Residual model on custom dataset."""
    output_path = config_dict["output_path"]

    if codebook_sizes is None:
        codebook_sizes = config_dict["RQ-VAE"]["code_book_size"]
    if isinstance(codebook_sizes, int):
        codebook_size = codebook_sizes
    else:
        codebook_size = codebook_sizes[0]
    max_items_per_seq = config_dict.get("max_items_per_seq", 20)

    writer = setup_logging(config_dict)

    # Load custom data
    datasets, semantic_info = load_custom_data(
        id_save_location,
        sequences_dict,
        id_split,
        item_embedding,
        method_config,
        max_length=config_dict["TIGER"]["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=max_items_per_seq,
    )

    # Semantic ID arrays
    all_semantic_ids = np.unique(
        np.concatenate([
            semantic_info["seen_semantic_ids"],
            semantic_info["val_unseen_semantic_ids"],
            semantic_info["test_unseen_semantic_ids"],
        ], axis=0),
        axis=0,
    )
    all_semantic_ids = torch.from_numpy(all_semantic_ids)

    n_semantic_codebook = semantic_info["n_semantic_codebook"]
    n_codebook = semantic_info["n_codebook"]
    max_last_semantic_ids = semantic_info["max_last_semantic_ids"]

    # Vocab size
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size

    flag_separate_bos_representation = method_config.get("flag_separate_bos_representation", False)

    if method_config.get("include_user_id", False):
        this_vocab_size = 2000 + sid_vocab_size + 2
    else:
        this_vocab_size = sid_vocab_size + 2

    if flag_separate_bos_representation:
        this_vocab_size += 1

    sid_start_token_id = this_vocab_size - 2 if flag_separate_bos_representation else None

    # Build T5 config
    t5_config = config_dict["TIGER"]["T5"]
    trainer_config = config_dict["TIGER"]["trainer"]
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
        n_positions=config_dict["TIGER"]["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    # Residual-specific config
    latent_size = t5_config["d_model"]
    if rqvae_codebook_weights is not None:
        latent_size = rqvae_codebook_weights[0].shape[-1]

    codebook_loss_weight = method_config.get("codebook_loss_weight", 1.0)
    num_residual_levels = method_config.get("num_residual_levels", n_semantic_codebook - 1)
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
        flag_use_output_embedding=method_config.get("flag_use_output_embedding", False),
        flag_use_learnable_text_embed=method_config.get("flag_add_input_embedding", False),
        embedding_head_dict=method_config.get("embedding_head_dict", {}),
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

    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    # Dataloaders
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]
    eval_keys = config_dict.get("eval_keys", [5, 10, 20, 100])
    retrieve_key = [max(eval_keys)]

    train_dataset = CustomDataset(datasets["train_all"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloaders = {}
    for cat in ["all", "normal", "surprise"]:
        key = f"val_{cat}"
        if key in datasets:
            val_dataloaders[cat] = DataLoader(
                CustomDataset(datasets[key]), batch_size=eval_batch_size, shuffle=False
            )

    test_dataloaders = {}
    for cat in ["all", "normal", "surprise"]:
        key = f"test_{cat}"
        if key in datasets:
            test_dataloaders[cat] = DataLoader(
                CustomDataset(datasets[key]), batch_size=eval_batch_size, shuffle=False
            )

    # Continue training setup
    continue_checkpoint = method_config.get("continue_from_tiger_checkpoint", None)
    adapter_warmup_steps = method_config.get("adapter_warmup_steps", 0)
    backbone_lr_factor = method_config.get("backbone_lr_factor", 0.1)
    adapter_lr_factor = method_config.get("adapter_lr_factor", 1.0)
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"
    backbone_is_frozen = False

    if continue_checkpoint and not os.path.exists(state_path):
        _load_continue_checkpoint(model, continue_checkpoint, device)
        if adapter_warmup_steps > 0:
            _freeze_backbone(model)
            backbone_is_frozen = True
        else:
            _unfreeze_all(model)

    # Optimizer
    if backbone_is_frozen:
        adapter_only_params = [p for n, p in model.named_parameters()
                              if _is_adapter_param(n) and p.requires_grad]
        optimizer = AdamW(adapter_only_params, lr=trainer_config["lr"],
                          weight_decay=trainer_config["weight_decay"])
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"], optimizer=optimizer,
                num_warmup_steps=min(trainer_config["warmup_steps"], adapter_warmup_steps // 4),
                num_training_steps=adapter_warmup_steps,
            )
    elif continue_checkpoint and adapter_warmup_steps == 0:
        bp, ap = _separate_params(model)
        optimizer = _build_phase2_optimizer(
            model, base_lr=trainer_config["lr"],
            backbone_lr_factor=backbone_lr_factor,
            adapter_lr_factor=adapter_lr_factor,
            weight_decay=trainer_config["weight_decay"],
            backbone_params=[p for _, p in bp],
            adapter_params=[p for _, p in ap],
        )
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"], optimizer=optimizer,
                num_warmup_steps=trainer_config["warmup_steps"],
                num_training_steps=trainer_config["steps"],
            )
    else:
        optimizer = AdamW(model.parameters(), lr=trainer_config["lr"],
                          weight_decay=trainer_config["weight_decay"])
        scheduler = None
        if trainer_config["scheduler"] != "none":
            scheduler = get_scheduler(
                name=trainer_config["scheduler"], optimizer=optimizer,
                num_warmup_steps=trainer_config["warmup_steps"],
                num_training_steps=trainer_config["steps"],
            )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # Resume
    total_steps = trainer_config["steps"]
    best_ndcg = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1

    if os.path.exists(state_path):
        training_state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(training_state["model_state_dict"], strict=False)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg = training_state.get("best_ndcg_10", -0.01)
        global_step = training_state["global_step"]
        best_epoch = training_state.get("best_epoch", 0)
        start_epoch = training_state["train_step"]
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
        print(f"Resumed from {state_path}, step={global_step}")

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})
    print(f"Total parameters: {total_params}")

    # Training loop
    for epoch in range(start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))):
        # Phase transition
        if backbone_is_frozen and global_step >= adapter_warmup_steps:
            _unfreeze_all(model)
            backbone_is_frozen = False
            bp, ap = _separate_params(model)
            optimizer = _build_phase2_optimizer(
                model, base_lr=trainer_config["lr"],
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
                    name=trainer_config["scheduler"], optimizer=optimizer,
                    num_warmup_steps=min(trainer_config["warmup_steps"], remaining_steps // 10),
                    num_training_steps=remaining_steps,
                )

        model = train_epoch_residual(
            epoch, train_dataloader, model, optimizer, device, scaler, scheduler,
            writer, id_split["seen"], n_semantic_codebook, n_codebook,
            method_config, semantic_info["item2sid"], item_embedding,
        )
        global_step += len(train_dataloader)

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:
            current_phase = 1 if backbone_is_frozen else 2
            logs, ndcg_val = evaluate_helper_custom(
                model, device, val_dataloaders, all_semantic_ids, method_config,
                keyword="val", KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
                use_residual=True,
            )
            logs["train/step"] = global_step
            logs["train/phase"] = current_phase

            if ndcg_val > best_ndcg:
                best_ndcg = ndcg_val
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "training_phase": current_phase,
            }
            torch.save(training_state, state_path)

        if (best_epoch + trainer_config["patience"] < epoch) and global_step > trainer_config["warmup_steps"]:
            print("Early stopping (patience exhausted)")
            break

    # Test
    print("Testing...")
    model = TIGER_Residual(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config.get("flag_use_output_embedding", False),
        flag_use_learnable_text_embed=method_config.get("flag_add_input_embedding", False),
        embedding_head_dict=method_config.get("embedding_head_dict", {}),
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
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    model.load_state_dict(torch.load(best_state_path), strict=False)
    model.resid_nontf_ratio = 0.0
    model.ntp_nontf_ratio = 0.0

    logs, _ = evaluate_helper_custom(
        model, device, test_dataloaders, all_semantic_ids, method_config,
        keyword="test", KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
        use_residual=True,
    )
    writer.log(logs)
    writer.finish()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train TIGER on custom dataset")

    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing data files")
    parser.add_argument("--embedding_file", type=str, required=True,
                        help="Path to .pt embedding file (from build_embedding.py)")
    parser.add_argument("--sid_file", type=str, required=True,
                        help="Path to pre-built semantic ID pickle file")
    parser.add_argument("--codebook_weights_file", type=str, default=None,
                        help="Path to RQ-VAE codebook weights (.pt) for TIGER_Residual")

    # Model selection
    parser.add_argument("--use_residual_decoder", type=bool, default=False,
                        help="Use TIGER_Residual model")
    parser.add_argument("--use_simple_residual", type=bool, default=False,
                        help="Use Simple_Residual model")
    parser.add_argument("--use_softlabel_sid", type=bool, default=False,
                        help="Use TIGER_SoftLabel model")

    # Training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--experiment_id", type=str, default="custom_exp")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Override output path (default: auto-generated)")
    parser.add_argument("--codebook_sizes", type=int, nargs="+", default=None,
                        help="Per-level codebook sizes (default: 256)")

    # Method config overrides (key TIGER_Residual params)
    parser.add_argument("--soft_label_K", type=int, default=0)
    parser.add_argument("--soft_label_temperature", type=float, default=1.0)
    parser.add_argument("--soft_label_temp_min", type=float, default=None)
    parser.add_argument("--soft_label_temp_decay_steps", type=int, default=10000)
    parser.add_argument("--codebook_loss_weight", type=float, default=1.0)
    parser.add_argument("--codebook_loss_weight_decay_steps", type=str, default=None,
                        help="Per-position decay steps, e.g. '5000,null,null,null'")
    parser.add_argument("--num_residual_levels", type=int, default=None)
    parser.add_argument("--cumulative_residual_loss_weight", type=float, default=0.0)
    parser.add_argument("--cumulative_residual_loss_type", type=str, default="mse")
    parser.add_argument("--cumulative_residual_loss_temperature", type=float, default=1.0)
    parser.add_argument("--resid_nontf_ratio", type=float, default=0.0)
    parser.add_argument("--ntp_nontf_ratio", type=float, default=0.0)
    parser.add_argument("--resid_beams", type=int, default=None)
    parser.add_argument("--resid_score_weight", type=float, default=1.0)
    parser.add_argument("--resid_beam_mode", type=str, default="pruning")
    parser.add_argument("--flag_separate_bos_representation", type=bool, default=False)
    parser.add_argument("--continue_from_tiger_checkpoint", type=str, default=None)
    parser.add_argument("--adapter_warmup_steps", type=int, default=0)
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--adapter_lr_factor", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--eval_keys", type=int, nargs="+", default=[5, 10, 20, 100],
                        help="K values for Recall@K, NDCG@K")

    # Wandb
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        help="wandb mode: disabled, offline, online")

    # RQ-VAE / SID construction (auto-triggered if sid_file doesn't exist)
    parser.add_argument("--sid_output_dir", type=str, default="./ID_generation/ID/",
                        help="Output directory for SID files (used when auto-training RQ-VAE)")
    parser.add_argument("--rqvae_epochs", type=int, default=300,
                        help="RQ-VAE training epochs (default=300, sufficient for 800K items)")
    parser.add_argument("--rqvae_batch_size", type=int, default=2048)
    parser.add_argument("--rqvae_lr", type=float, default=0.001)
    parser.add_argument("--rqvae_beta", type=float, default=0.25)
    parser.add_argument("--rqvae_codebook_size", type=int, default=256)
    parser.add_argument("--rqvae_num_layers", type=int, default=3)
    parser.add_argument("--rqvae_latent_dim", type=int, default=None,
                        help="RQ-VAE latent dim (default: auto from custom.yaml)")
    parser.add_argument("--rqvae_hidden_dim", type=int, nargs="+", default=None,
                        help="RQ-VAE hidden dims (default: auto from custom.yaml)")
    parser.add_argument("--rqvae_dropout", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed)

    # -----------------------------------------------------------------------
    # Build config from YAML + CLI overrides
    # -----------------------------------------------------------------------
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    dataset_cfg = OmegaConf.load(os.path.join(config_dir, "dataset", "custom.yaml"))
    method_cfg = OmegaConf.load(os.path.join(config_dir, "method", "base.yaml"))
    logging_cfg = OmegaConf.load(os.path.join(config_dir, "logging", "wandb.yaml"))

    # Override with CLI args
    dataset_cfg["type"] = "Custom"
    dataset_cfg["data_dir"] = args.data_dir
    dataset_cfg["embedding_file"] = args.embedding_file
    dataset_cfg["eval_keys"] = args.eval_keys

    # Update RQ-VAE input_dim if codebook_sizes given
    if args.codebook_sizes is not None:
        if len(args.codebook_sizes) == 1:
            dataset_cfg["RQ-VAE"]["code_book_size"] = args.codebook_sizes[0]
        else:
            dataset_cfg["RQ-VAE"]["code_book_size"] = max(args.codebook_sizes)
            dataset_cfg["RQ-VAE"]["num_layers"] = len(args.codebook_sizes)

    logging_cfg["mode"] = args.wandb_mode

    # Build method_config dict from args
    method_config = OmegaConf.to_container(method_cfg, resolve=True)
    method_config["use_residual_decoder"] = args.use_residual_decoder
    method_config["use_simple_residual"] = args.use_simple_residual
    method_config["use_softlabel_sid"] = args.use_softlabel_sid
    method_config["soft_label_K"] = args.soft_label_K
    method_config["soft_label_temperature"] = args.soft_label_temperature
    method_config["soft_label_temp_min"] = args.soft_label_temp_min
    method_config["soft_label_temp_decay_steps"] = args.soft_label_temp_decay_steps
    method_config["codebook_loss_weight"] = args.codebook_loss_weight
    if args.codebook_loss_weight_decay_steps is not None:
        parts = args.codebook_loss_weight_decay_steps.split(",")
        parsed = []
        for p in parts:
            p = p.strip()
            if p.lower() == "null" or p == "":
                parsed.append(None)
            else:
                parsed.append(int(p))
        method_config["codebook_loss_weight_decay_steps"] = parsed
    method_config["num_residual_levels"] = args.num_residual_levels
    method_config["cumulative_residual_loss_weight"] = args.cumulative_residual_loss_weight
    method_config["cumulative_residual_loss_type"] = args.cumulative_residual_loss_type
    method_config["cumulative_residual_loss_temperature"] = args.cumulative_residual_loss_temperature
    method_config["resid_nontf_ratio"] = args.resid_nontf_ratio
    method_config["ntp_nontf_ratio"] = args.ntp_nontf_ratio
    method_config["resid_beams"] = args.resid_beams
    method_config["resid_score_weight"] = args.resid_score_weight
    method_config["resid_beam_mode"] = args.resid_beam_mode
    method_config["flag_separate_bos_representation"] = args.flag_separate_bos_representation
    method_config["continue_from_tiger_checkpoint"] = args.continue_from_tiger_checkpoint
    method_config["adapter_warmup_steps"] = args.adapter_warmup_steps
    method_config["backbone_lr_factor"] = args.backbone_lr_factor
    method_config["adapter_lr_factor"] = args.adapter_lr_factor

    # -----------------------------------------------------------------------
    # Output path
    # -----------------------------------------------------------------------
    if args.output_path is not None:
        output_path = args.output_path
    else:
        model_name = "residual" if args.use_residual_decoder else (
            "simple_residual" if args.use_simple_residual else (
            "softlabel" if args.use_softlabel_sid else "tiger"))
        output_path = f"./results/custom/{model_name}/{args.experiment_id}_seed_{args.seed}"
    os.makedirs(output_path, exist_ok=True)

    # Build combined config dict
    dataset_dict = OmegaConf.to_container(dataset_cfg, resolve=True)
    config_dict = {
        **dataset_dict,
        "output_path": output_path,
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "device_id": args.device_id,
        "logging": OmegaConf.to_container(logging_cfg, resolve=True),
        "dataset": dataset_dict,  # needed by setup_logging
        "method": method_config,  # needed by setup_logging
    }

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading data...")
    data_dir = args.data_dir

    # Item embedding (from build_embedding.py)
    item_embedding, vec_dim, num_items = load_item_embedding(args.embedding_file, device)
    print(f"Item embedding: shape={item_embedding.shape}")

    # Auto-detect input_dim (must update BOTH OmegaConf and plain dict)
    dataset_cfg["RQ-VAE"]["input_dim"] = vec_dim
    dataset_dict["RQ-VAE"]["input_dim"] = vec_dim

    # Load sequence files (tab-separated: uid\titem1\titem2...)
    # For training we only need "all"; normal/surprise are not used during training.
    print("Loading sequence files...")
    train_sequences = {}
    train_uids = {}
    train_all_fname = dataset_dict["train_files"].get("all")
    if train_all_fname:
        seqs, uids = load_sequence_file(os.path.join(data_dir, train_all_fname))
        train_sequences["train_all"] = seqs
        train_uids["train_all"] = uids
    else:
        raise ValueError("train_files must contain an 'all' key in the dataset config.")

    val_sequences = {}
    val_uids = {}
    for cat, fname in dataset_dict["val_files"].items():
        seqs, uids = load_sequence_file(os.path.join(data_dir, fname))
        val_sequences[f"val_{cat}"] = seqs
        val_uids[f"val_{cat}"] = uids

    test_sequences = {}
    test_uids = {}
    for cat, fname in dataset_dict["test_files"].items():
        seqs, uids = load_sequence_file(os.path.join(data_dir, fname))
        test_sequences[f"test_{cat}"] = seqs
        test_uids[f"test_{cat}"] = uids

    # Merge all sequences into one dict for load_custom_data
    all_sequences = {**train_sequences, **val_sequences, **test_sequences}
    for key, seqs in all_sequences.items():
        print(f"  {key}: {len(seqs)} sequences")

    # Compute id_split from training data
    id_split = compute_id_split(
        train_sequences["train_all"],
        val_sequences["val_all"],
        test_sequences["test_all"],
    )

    # -----------------------------------------------------------------------
    # Auto-train RQ-VAE if sid_file doesn't exist (same as run.py)
    # Uses wandb writer for logging — just like the main pipeline
    # -----------------------------------------------------------------------
    sid_file = args.sid_file
    sid_dir = os.path.dirname(sid_file) if os.path.dirname(sid_file) else args.sid_output_dir
    os.makedirs(sid_dir, exist_ok=True)

    if not os.path.exists(sid_file):
        print(f"\n{'='*60}")
        print(f"SID file not found: {sid_file}")
        print(f"Auto-training RQ-VAE with wandb logging...")
        print(f"{'='*60}\n")

        from ID_generation.rqvae.rqvae import RQVAE
        from ID_generation.train_rqvae import train_rqvae, calc_cos_sim

        # Override RQ-VAE config with CLI args (if provided)
        rqvae_cfg = dataset_dict["RQ-VAE"]
        if args.rqvae_epochs is not None:
            rqvae_cfg["epochs"] = args.rqvae_epochs
        if args.rqvae_batch_size is not None:
            rqvae_cfg["batch_size"] = args.rqvae_batch_size
        if args.rqvae_lr is not None:
            rqvae_cfg["lr"] = args.rqvae_lr
        if args.rqvae_beta is not None:
            rqvae_cfg["beta"] = args.rqvae_beta
        if args.rqvae_codebook_size is not None:
            rqvae_cfg["code_book_size"] = args.rqvae_codebook_size
        if args.rqvae_num_layers is not None:
            rqvae_cfg["num_layers"] = args.rqvae_num_layers
        if args.rqvae_latent_dim is not None:
            rqvae_cfg["latent_dim"] = args.rqvae_latent_dim
        if args.rqvae_hidden_dim is not None:
            rqvae_cfg["hidden_dim"] = args.rqvae_hidden_dim
        if args.rqvae_dropout is not None:
            rqvae_cfg["dropout"] = args.rqvae_dropout

        # Print RQ-VAE config for debugging
        print(f"RQ-VAE config: input_dim={rqvae_cfg['input_dim']}, "
              f"hidden_dim={rqvae_cfg['hidden_dim']}, latent_dim={rqvae_cfg['latent_dim']}, "
              f"num_layers={rqvae_cfg['num_layers']}, code_book_size={rqvae_cfg['code_book_size']}")

        # Build RQ-VAE model
        rqvae_model = RQVAE(
            input_size=rqvae_cfg["input_dim"],
            hidden_sizes=rqvae_cfg["hidden_dim"],
            latent_size=rqvae_cfg["latent_dim"],
            num_levels=rqvae_cfg["num_layers"],
            codebook_size=rqvae_cfg["code_book_size"],
            dropout=rqvae_cfg["dropout"],
            latent_loss_weight=rqvae_cfg["beta"],
        )

        # Use wandb writer for RQ-VAE training (same as run.py)
        rqvae_writer = setup_logging(config_dict)

        # Train on seen items
        seen_embeddings = item_embedding[id_split["seen"] - 1]
        print(f"Training RQ-VAE: {len(id_split['seen'])} seen items, "
              f"dim={rqvae_cfg['input_dim']}, epochs={rqvae_cfg['epochs']}")

        train_rqvae(rqvae_model, seen_embeddings, device, rqvae_writer, rqvae_cfg)
        rqvae_writer.finish()

        # Assign SIDs to ALL items
        rqvae_model.to(device)
        rqvae_model.eval()
        all_ids = rqvae_model.get_codes(item_embedding).cpu().numpy()

        # Save SID pickle
        import pickle
        with open(sid_file, "wb") as f:
            pickle.dump(all_ids, f)
        print(f"Saved SIDs to {sid_file}, shape={all_ids.shape}")

        # Save codebook weights (for TIGER_Residual)
        codebook_weights_data = [cb.weight.data.clone() for cb in rqvae_model.quantizer.codebooks]
        cb_save_path = os.path.join(sid_dir, f"custom_codebook_weights_{args.seed}.pt")
        torch.save(codebook_weights_data, cb_save_path)
        print(f"Saved codebook weights to {cb_save_path}")

        # Print quality metrics
        cos_sim_array = calc_cos_sim(rqvae_model, seen_embeddings, rqvae_cfg)
        for i in range(rqvae_cfg["num_layers"]):
            print(f"  Cosine similarity @ L{i+1}: {cos_sim_array[i]:.4f}")

        print(f"\nSID construction complete!\n")
    else:
        print(f"SID file found: {sid_file}, skipping RQ-VAE training.")

    # Codebook sizes
    codebook_sizes = args.codebook_sizes
    if codebook_sizes is not None and len(codebook_sizes) == 1:
        codebook_sizes = codebook_sizes[0]

    # Codebook weights (load from file, or from auto-trained RQ-VAE above)
    rqvae_codebook_weights = None
    if args.codebook_weights_file and os.path.exists(args.codebook_weights_file):
        rqvae_codebook_weights = torch.load(args.codebook_weights_file, map_location=device)
        print(f"Loaded codebook weights from {args.codebook_weights_file}")
    elif not os.path.exists(sid_file) or args.codebook_weights_file is None:
        # Try to load auto-saved codebook weights from SID construction
        auto_cb_path = os.path.join(sid_dir, f"custom_codebook_weights_{args.seed}.pt")
        if os.path.exists(auto_cb_path):
            rqvae_codebook_weights = torch.load(auto_cb_path, map_location=device)
            print(f"Loaded auto-saved codebook weights from {auto_cb_path}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    try:
        if args.use_residual_decoder:
            print("Training TIGER_Residual...")
            train_tiger_residual_custom(
                config_dict, method_config, all_sequences, id_split,
                item_embedding, args.sid_file, device=device,
                rqvae_codebook_weights=rqvae_codebook_weights,
                codebook_sizes=codebook_sizes,
            )
        else:
            print("Training TIGER base model...")
            train_tiger_custom(
                config_dict, method_config, all_sequences, id_split,
                item_embedding, args.sid_file, device=device,
                codebook_sizes=codebook_sizes,
            )
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
