#!/usr/bin/env python3
"""
Build Semantic IDs (SIDs) for a custom dataset.

Prerequisite: Run build_embedding.py first to produce the .pt embedding file.

This script:
  1. Loads pre-built item embedding tensor from .pt file
  2. Loads training sequences to determine seen/unseen items
  3. Trains RQ-VAE on 'seen' items
  4. Assigns SIDs to ALL items using the trained RQ-VAE
  5. Saves the SID pickle and codebook weights for TIGER training

Usage:
  python build_custom_sid.py \
    --embedding_file ./data/custom/item_embedding.pt \
    --data_dir ./data/custom/ \
    --train_file train_all.txt \
    --output_dir ./ID_generation/ID/ \
    --seed 0
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ID_generation.rqvae.rqvae import RQVAE
from ID_generation.train_rqvae import train_rqvae, calc_cos_sim
from src.load_custom_data import load_sequence_file, load_item_embedding

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build Semantic IDs for custom dataset")
    parser.add_argument("--embedding_file", type=str, required=True,
                        help="Path to .pt embedding file (from build_embedding.py)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing sequence data files")
    parser.add_argument("--train_file", type=str, default="train_all.txt",
                        help="Training file for determining seen items (relative to data_dir)")
    parser.add_argument("--output_dir", type=str, default="./ID_generation/ID/",
                        help="Output directory for SID files")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device_id", type=int, default=0,
                        help="CUDA device ID")

    # RQ-VAE hyperparameters
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, nargs="+", default=[768, 512, 256])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Step 1: Load item embedding from .pt file
    # -----------------------------------------------------------------------
    item_embedding, input_dim, num_items = load_item_embedding(args.embedding_file, device)
    print(f"Item embedding: shape={item_embedding.shape}, dim={input_dim}, num_items={num_items}")

    # -----------------------------------------------------------------------
    # Step 2: Load training sequences to determine seen items
    # -----------------------------------------------------------------------
    train_path = os.path.join(args.data_dir, args.train_file)
    train_sequences, _ = load_sequence_file(train_path)

    train_items = set()
    for seq in train_sequences:
        train_items.update(seq)

    seen_items = np.array(sorted(train_items))
    print(f"Seen items (training): {len(seen_items)}")

    # -----------------------------------------------------------------------
    # Step 3: Train RQ-VAE on seen items
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Simple console writer (no wandb for SID construction)
    class ConsoleWriter:
        def log(self, d):
            pass
        def finish(self):
            pass

    writer = ConsoleWriter()

    rqvae_config = {
        "code_book_size": args.codebook_size,
        "num_layers": args.num_layers,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 0.1,
    }

    rqvae = RQVAE(
        input_size=input_dim,
        hidden_sizes=args.hidden_dim,
        latent_size=args.latent_dim,
        num_levels=args.num_layers,
        codebook_size=args.codebook_size,
        dropout=args.dropout,
        latent_loss_weight=args.beta,
    )

    # Train on seen items only
    seen_embeddings = item_embedding[seen_items - 1]
    print(f"Training RQ-VAE on {len(seen_items)} seen items, dim={input_dim}")
    train_rqvae(rqvae, seen_embeddings, device, writer, rqvae_config)

    # -----------------------------------------------------------------------
    # Step 4: Assign SIDs to ALL items
    # -----------------------------------------------------------------------
    rqvae.to(device)
    rqvae.eval()
    all_ids = rqvae.get_codes(item_embedding).cpu().numpy()  # [num_items, num_layers]

    # Save SID pickle (same format as original pipeline)
    save_name = f"custom_semantic_{args.seed}.pkl"
    save_path = os.path.join(args.output_dir, save_name)
    import pickle
    with open(save_path, "wb") as f:
        pickle.dump(all_ids, f)
    print(f"Saved SIDs to {save_path}, shape={all_ids.shape}")

    # Save codebook weights (needed for TIGER_Residual)
    codebook_weights = [cb.weight.data.clone() for cb in rqvae.quantizer.codebooks]
    cb_save_path = os.path.join(args.output_dir, f"custom_codebook_weights_{args.seed}.pt")
    torch.save(codebook_weights, cb_save_path)
    print(f"Saved codebook weights to {cb_save_path}")

    # -----------------------------------------------------------------------
    # Step 5: Print quality metrics
    # -----------------------------------------------------------------------
    cos_sim_array = calc_cos_sim(rqvae, seen_embeddings, rqvae_config)
    for i in range(args.num_layers):
        print(f"  Cosine similarity @ L{i+1}: {cos_sim_array[i]:.4f}")

    print("\nSID construction complete!")
    print(f"  SID file:      {save_path}")
    print(f"  Codebook file: {cb_save_path}")
    print(f"  Vocab size:    {args.codebook_size * args.num_layers} + collision_avoidance_dim")


if __name__ == "__main__":
    main()
