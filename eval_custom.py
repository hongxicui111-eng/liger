#!/usr/bin/env python3
"""
Evaluate a trained TIGER/TIGER_Residual model on custom dataset.

Evaluates on all 3 subsets (all, normal, surprise) with
Recall, NDCG @ 5, 10, 20, 100.

Usage:
  python eval_custom.py \
    --data_dir ./data/custom/ \
    --embedding_file ./data/custom/item_embedding.pt \
    --sid_file ./ID_generation/ID/custom_semantic_0.pkl \
    --codebook_weights_file ./ID_generation/ID/custom_codebook_weights_0.pt \
    --checkpoint ./results/custom/residual/custom_exp_seed_0/results/ckpt_best.pt \
    --use_residual_decoder True \
    --eval_keys 5 10 20 100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load_custom_data import (
    load_sequence_file,
    load_item_embedding,
    compute_id_split,
    load_custom_data,
)
from src.evaluation import evaluate, evaluate_residual, evaluate_simple_residual
from src.tiger_residual import TIGER_Residual
from src.tiger import TIGER
from utils import CustomDataset

from transformers import T5Config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TIGER on custom dataset")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--embedding_file", type=str, required=True,
                        help="Path to .pt embedding file (from build_embedding.py)")
    parser.add_argument("--sid_file", type=str, required=True)
    parser.add_argument("--codebook_weights_file", type=str, default=None)

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--use_residual_decoder", type=bool, default=False)
    parser.add_argument("--use_simple_residual", type=bool, default=False)
    parser.add_argument("--codebook_sizes", type=int, nargs="+", default=None)

    # Method config (must match training config)
    parser.add_argument("--soft_label_K", type=int, default=0)
    parser.add_argument("--soft_label_temperature", type=float, default=1.0)
    parser.add_argument("--soft_label_temp_min", type=float, default=None)
    parser.add_argument("--soft_label_temp_decay_steps", type=int, default=10000)
    parser.add_argument("--codebook_loss_weight", type=float, default=1.0)
    parser.add_argument("--codebook_loss_weight_decay_steps", type=str, default=None)
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
    parser.add_argument("--include_user_id", type=bool, default=False)
    parser.add_argument("--flag_add_input_embedding", type=bool, default=False)
    parser.add_argument("--flag_use_output_embedding", type=bool, default=False)

    # Evaluation
    parser.add_argument("--eval_keys", type=int, nargs="+", default=[5, 10, 20, 100])
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    dataset_cfg = OmegaConf.load(os.path.join(config_dir, "dataset", "custom.yaml"))
    method_cfg = OmegaConf.load(os.path.join(config_dir, "method", "base.yaml"))

    dataset_dict = OmegaConf.to_container(dataset_cfg, resolve=True)
    method_config = OmegaConf.to_container(method_cfg, resolve=True)

    # Override method config
    method_config["use_residual_decoder"] = args.use_residual_decoder
    method_config["use_simple_residual"] = args.use_simple_residual
    method_config["include_user_id"] = args.include_user_id
    method_config["flag_add_input_embedding"] = args.flag_add_input_embedding
    method_config["flag_use_output_embedding"] = args.flag_use_output_embedding
    method_config["soft_label_K"] = args.soft_label_K
    method_config["soft_label_temperature"] = args.soft_label_temperature
    method_config["soft_label_temp_min"] = args.soft_label_temp_min
    method_config["soft_label_temp_decay_steps"] = args.soft_label_temp_decay_steps
    method_config["codebook_loss_weight"] = args.codebook_loss_weight
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

    # Load data
    data_dir = args.data_dir
    item_embedding, vec_dim, num_items = load_item_embedding(args.embedding_file, device)

    # Load sequences
    val_sequences = {}
    for cat, fname in dataset_dict["val_files"].items():
        seqs, _ = load_sequence_file(os.path.join(data_dir, fname))
        val_sequences[f"val_{cat}"] = seqs

    test_sequences = {}
    for cat, fname in dataset_dict["test_files"].items():
        seqs, _ = load_sequence_file(os.path.join(data_dir, fname))
        test_sequences[f"test_{cat}"] = seqs

    # Also need training data for id_split
    train_all, _ = load_sequence_file(os.path.join(data_dir, dataset_dict["train_files"]["all"]))

    id_split = compute_id_split(train_all, val_sequences["val_all"], test_sequences["test_all"])

    codebook_sizes = args.codebook_sizes
    if codebook_sizes is not None and len(codebook_sizes) == 1:
        codebook_sizes = codebook_sizes[0]

    # Load SID data
    all_sequences = {**val_sequences, **test_sequences}
    datasets, semantic_info = load_custom_data(
        args.sid_file, all_sequences, id_split, item_embedding, method_config,
        max_length=dataset_dict["TIGER"]["n_positions"],
        codebook_sizes=codebook_sizes,
        max_items_per_seq=dataset_dict.get("max_items_per_seq", 20),
    )

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
        codebook_size = codebook_sizes
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size
        codebook_size = codebook_sizes[0] if codebook_sizes else 256

    flag_separate_bos = method_config.get("flag_separate_bos_representation", False)
    if method_config.get("include_user_id", False):
        this_vocab_size = 2000 + sid_vocab_size + 2
    else:
        this_vocab_size = sid_vocab_size + 2
    if flag_separate_bos:
        this_vocab_size += 1

    sid_start_token_id = this_vocab_size - 2 if flag_separate_bos else None

    # Build model
    t5_config_dict = dataset_dict["TIGER"]["T5"]
    model_config = T5Config(
        num_layers=t5_config_dict["encoder_layers"],
        num_decoder_layers=t5_config_dict["decoder_layers"],
        d_model=t5_config_dict["d_model"],
        d_ff=t5_config_dict["d_ff"],
        num_heads=t5_config_dict["num_heads"],
        d_kv=t5_config_dict["d_kv"],
        dropout_rate=t5_config_dict["dropout_rate"],
        vocab_size=this_vocab_size,
        pad_token_id=0,
        eos_token_id=int(this_vocab_size - 1),
        decoder_start_token_id=0,
        feed_forward_proj=t5_config_dict["feed_forward_proj"],
        n_positions=dataset_dict["TIGER"]["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config_dict["initializer_factor"],
    )

    rqvae_codebook_weights = None
    if args.codebook_weights_file and os.path.exists(args.codebook_weights_file):
        rqvae_codebook_weights = torch.load(args.codebook_weights_file, map_location=device)

    latent_size = t5_config_dict["d_model"]
    if rqvae_codebook_weights is not None:
        latent_size = rqvae_codebook_weights[0].shape[-1]

    if args.use_residual_decoder:
        model = TIGER_Residual(
            config=model_config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=dataset_dict.get("max_items_per_seq", 20),
            flag_use_output_embedding=method_config.get("flag_use_output_embedding", False),
            flag_use_learnable_text_embed=method_config.get("flag_add_input_embedding", False),
            embedding_head_dict=method_config.get("embedding_head_dict", {}),
            rqvae_codebook_weights=rqvae_codebook_weights,
            codebook_size=codebook_size,
            latent_size=latent_size,
            codebook_loss_weight=method_config.get("codebook_loss_weight", 1.0),
            num_residual_levels=method_config.get("num_residual_levels", n_semantic_codebook - 1),
            soft_label_K=method_config.get("soft_label_K", 0),
            soft_label_temperature=method_config.get("soft_label_temperature", 1.0),
            soft_label_temp_min=method_config.get("soft_label_temp_min", None),
            soft_label_temp_decay_steps=method_config.get("soft_label_temp_decay_steps", 10000),
            codebook_loss_weight_decay_steps=method_config.get("codebook_loss_weight_decay_steps", None),
            cumulative_residual_loss_weight=method_config.get("cumulative_residual_loss_weight", 0.0),
            cumulative_residual_loss_type=method_config.get("cumulative_residual_loss_type", "mse"),
            cumulative_residual_loss_temperature=method_config.get("cumulative_residual_loss_temperature", 1.0),
            resid_nontf_ratio=0.0,  # Always 0 for evaluation
            ntp_nontf_ratio=0.0,    # Always 0 for evaluation
            flag_separate_bos_representation=flag_separate_bos,
            sid_start_token_id=sid_start_token_id,
        ).to(device)
        if not isinstance(codebook_sizes, int):
            model.set_codebook_offsets(codebook_sizes)
    else:
        model = TIGER(
            config=model_config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=dataset_dict.get("max_items_per_seq", 20),
            flag_use_output_embedding=method_config.get("flag_use_output_embedding", False),
            flag_use_learnable_text_embed=method_config.get("flag_add_input_embedding", False),
            embedding_head_dict=method_config.get("embedding_head_dict", {}),
        ).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    # Build dataloaders
    eval_keys = args.eval_keys
    retrieve_key = [max(eval_keys)]

    eval_dataloaders = {}
    for cat in ["all", "normal", "surprise"]:
        for prefix in ["val", "test"]:
            key = f"{prefix}_{cat}"
            if key in datasets:
                eval_dataloaders[key] = DataLoader(
                    CustomDataset(datasets[key]), batch_size=args.eval_batch_size, shuffle=False
                )

    # Evaluate
    print(f"\n{'='*80}")
    print(f"Evaluation Results — KEYS={eval_keys}, BEAMS={retrieve_key}")
    print(f"{'='*80}\n")

    for subset_name, dataloader in eval_dataloaders.items():
        if len(dataloader) == 0:
            print(f"  [{subset_name}] Skipped (empty)")
            continue

        if args.use_residual_decoder:
            recall_dict, ndcg_dict, _, _ = evaluate_residual(
                model, dataloader, all_semantic_ids, device,
                method_config=method_config, KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
            )
        elif args.use_simple_residual:
            recall_dict, ndcg_dict, _, _ = evaluate_simple_residual(
                model, dataloader, all_semantic_ids, device,
                method_config=method_config, KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
            )
        else:
            recall_dict, ndcg_dict, _, _ = evaluate(
                model, dataloader, all_semantic_ids, device,
                method_config=method_config, KEYS=eval_keys, RETRIEVE_KEY=retrieve_key,
            )

        print(f"  [{subset_name}] ({len(dataloader.dataset)} samples)")
        for k in eval_keys:
            r = torch.tensor(recall_dict[k]).mean().item()
            n = torch.tensor(ndcg_dict[k]).mean().item()
            print(f"    Recall@{k}: {r:.4f}  |  NDCG@{k}: {n:.4f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
