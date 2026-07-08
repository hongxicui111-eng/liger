"""
Train SASRec and extract CF (Collaborative Filtering) item embeddings.

This script:
1. Trains SASRec on user-item interaction sequences
2. Extracts the item_embeddings layer as CF embeddings
3. Saves them as a .pt file for use in LETTER's CF alignment loss

Usage (standalone):
    python -m ID_generation.sasrec.train_sasrec \
        --data_file ./ID_generation/preprocessing/processed/Beauty_inter.txt \
        --num_items 12102 \
        --hidden_units 32 \
        --num_heads 2 \
        --num_blocks 2 \
        --max_len 50 \
        --epochs 200 \
        --lr 0.001 \
        --batch_size 128 \
        --save_path ./ID_generation/sasrec/ckpt/Beauty-32d-sasrec.pt

Or called from run.py via train_sasrec() / extract_cf_embeddings().
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import SASRec


class SasRecDataset(Dataset):
    """Dataset for SASRec training.

    Generates (input_seq, positive_seq, negative_seq) triplets for BPR training.
    """

    def __init__(self, user_sequences, num_items, max_len, neg_sample_count=1):
        """
        Args:
            user_sequences: list of list of int, each inner list is a user's item sequence
            num_items: total number of items
            max_len: maximum sequence length
            neg_sample_count: number of negative samples per positive
        """
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.max_len = max_len
        self.neg_sample_count = neg_sample_count

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        seq = self.user_sequences[idx]

        # Truncate from the left if too long
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]

        # Build input sequence and targets
        # For position t, input is seq[0:t], target is seq[t]
        # We create input_seq = seq[:-1], pos_seq = seq[1:]
        input_seq = seq[:-1] if len(seq) > 1 else seq
        pos_seq = seq[1:] if len(seq) > 1 else seq

        # Pad to max_len
        input_seq = _pad_sequence(input_seq, self.max_len)
        pos_seq = _pad_sequence(pos_seq, self.max_len)

        # Negative sampling: for each position, sample a random item not in the sequence
        neg_seq = np.random.randint(1, self.num_items + 1, size=self.max_len)
        # Avoid sampling items that are in the actual sequence
        seq_set = set(seq)
        for i in range(self.max_len):
            while neg_seq[i] in seq_set:
                neg_seq[i] = np.random.randint(1, self.num_items + 1)

        return (
            torch.LongTensor(input_seq),
            torch.LongTensor(pos_seq),
            torch.LongTensor(neg_seq),
        )


def _pad_sequence(seq, max_len, pad_value=0):
    """Left-pad a sequence to max_len."""
    if len(seq) >= max_len:
        return list(seq[-max_len:])
    return [pad_value] * (max_len - len(seq)) + list(seq)


def train_sasrec(
    user_sequences,
    num_items,
    device,
    save_path,
    hidden_units=32,
    num_heads=2,
    num_blocks=2,
    max_len=50,
    dropout=0.2,
    epochs=200,
    lr=0.001,
    batch_size=128,
    weight_decay=0.0,
    eval_steps=10,
    patience=20,
    seed=42,
):
    """Train SASRec model on user-item interaction sequences.

    Args:
        user_sequences: list of list of int — each user's item sequence (1-indexed)
        num_items: total number of items
        device: torch device
        save_path: path to save the CF embeddings .pt file
        hidden_units: SASRec embedding dimension (also CF embedding dim)
        num_heads: number of attention heads
        num_blocks: number of self-attention blocks
        max_len: maximum sequence length
        dropout: dropout rate
        epochs: number of training epochs
        lr: learning rate
        batch_size: training batch size
        weight_decay: L2 regularization weight
        eval_steps: evaluate every N steps
        patience: early stopping patience (in eval_steps)
        seed: random seed

    Returns:
        model: trained SASRec model
    """
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Build dataset and dataloader
    dataset = SasRecDataset(user_sequences, num_items, max_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    # Create model
    model = SASRec(
        num_items=num_items,
        hidden_units=hidden_units,
        num_heads=num_heads,
        num_blocks=num_blocks,
        max_len=max_len,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training SASRec for CF embeddings")
    print(f"  Items: {num_items}, Hidden dim: {hidden_units}")
    print(f"  Heads: {num_heads}, Blocks: {num_blocks}, Max len: {max_len}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"  Save path: {save_path}")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    patience_counter = 0
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_seq, pos_seq, neg_seq = [x.to(device) for x in batch]

            optimizer.zero_grad()
            loss = model(input_seq, pos_seq, neg_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            step += 1

        avg_loss = total_loss / max(num_batches, 1)

        if epoch % eval_steps == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.6f}")

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                    break

    # Restore best model
    if "best_state" in dir():
        model.load_state_dict(best_state)

    # Extract and save CF embeddings
    cf_embeddings = extract_cf_embeddings(model, num_items, device)
    torch.save(cf_embeddings, save_path)
    print(f"\n✓ CF embeddings saved to {save_path}")
    print(f"  Shape: {cf_embeddings.shape} (n_items × hidden_units)")

    return model


def extract_cf_embeddings(model, num_items, device):
    """Extract item embeddings from a trained SASRec model.

    This matches the original LETTER repo's CF_ckpt_process.ipynb:
        matrix = model.item_embeddings(torch.LongTensor([1, 2, ..., num_items]))

    Args:
        model: trained SASRec model
        num_items: total number of items
        device: torch device

    Returns:
        cf_embeddings: tensor of shape [num_items, hidden_units]
                       cf_embeddings[i] is the embedding for item ID (i+1)
    """
    model.eval()
    with torch.no_grad():
        item_ids = torch.LongTensor(list(range(1, num_items + 1))).to(device)
        cf_embeddings = model.item_embeddings(item_ids)  # [num_items, hidden_units]
    return cf_embeddings.cpu()


def get_num_items_from_sequences(user_sequences):
    """Get the total number of unique items from user sequences.

    Args:
        user_sequences: list of list of int

    Returns:
        num_items: maximum item ID (items are 1-indexed)
    """
    max_id = 0
    for seq in user_sequences:
        if seq:
            max_id = max(max_id, max(seq))
    return max_id
