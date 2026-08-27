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
from tqdm import tqdm

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


@torch.no_grad()
def evaluate_recall_ndcg(model, user_sequences, num_items, device, max_len,
                         k=10, batch_size=256, val_target_index=-2):
    """Evaluate Recall@K and NDCG@K on a held-out validation set.

    For each user, the item at ``val_target_index`` (default: -2, i.e. the
    val item) is the target; the sequence up to (but excluding) it is the
    input. We score all items and check whether the target is in the top-K.

    Args:
        model: trained SASRec model
        user_sequences: list of list of int (1-indexed item IDs).
            These should be the *full* sequences (including val & test),
            so that val_target_index picks the correct held-out item.
        num_items: total number of items
        device: torch device
        max_len: max sequence length
        k: top-K cutoff
        batch_size: evaluation batch size
        val_target_index: index of the val item within each sequence
            (default -2; -1 would be the test item).

    Returns:
        recall@k, ndcg@k (floats)
    """
    model.eval()

    # Build eval set: input = seq up to val_target_index (exclusive),
    # target = seq[val_target_index]
    eval_seqs = []
    targets = []
    for seq in user_sequences:
        # Need at least val_target_index + 1 items
        if len(seq) < abs(val_target_index) + 1:
            continue
        input_seq = seq[:val_target_index]
        if len(input_seq) == 0:
            continue
        if len(input_seq) > max_len:
            input_seq = input_seq[-max_len:]
        eval_seqs.append(_pad_sequence(input_seq, max_len))
        targets.append(seq[val_target_index])

    if not eval_seqs:
        return 0.0, 0.0

    all_item_ids = torch.LongTensor(list(range(1, num_items + 1))).to(device)
    hits = 0
    ndcg_sum = 0.0

    for start in range(0, len(eval_seqs), batch_size):
        end = min(start + batch_size, len(eval_seqs))
        batch_seqs = torch.LongTensor(eval_seqs[start:end]).to(device)
        batch_targets = torch.LongTensor(targets[start:end]).to(device)

        # Score all items for each user in the batch
        # model.predict expects [B, num_candidates]
        # For efficiency, use the last hidden state directly
        hidden = model.log2feats(batch_seqs)  # [B, max_len, hidden]
        mask = (batch_seqs > 0).float()
        seq_lengths = mask.sum(dim=-1).long()
        last_indices = (seq_lengths - 1).clamp(min=0)
        user_repr = hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]
        # user_repr: [B, hidden]

        # Score all items: [num_items, hidden] @ [B, hidden] -> [B, num_items]
        all_emb = model._item_emb(all_item_ids)  # [num_items, hidden]
        scores = user_repr @ all_emb.t()  # [B, num_items]

        # Top-K
        _, topk_indices = scores.topk(k, dim=-1)  # [B, k]
        # topk_indices are 0-indexed into [1, num_items], so +1 to get item IDs
        topk_item_ids = topk_indices + 1  # [B, k]

        # Check if target is in top-K
        target_expanded = batch_targets.unsqueeze(1)  # [B, 1]
        hit_mask = (topk_item_ids == target_expanded).any(dim=-1)  # [B]
        hits += hit_mask.sum().item()

        # NDCG: DCG = 1/log2(rank+2) for the rank of the target (if hit)
        n_in_batch = end - start
        for i in range(n_in_batch):
            if hit_mask[i]:
                # Find rank of target in top-K
                rank = (topk_item_ids[i] == batch_targets[i]).nonzero(as_tuple=True)[0].item()
                ndcg_sum += 1.0 / np.log2(rank + 2)  # +2 because rank is 0-indexed

    n_eval = len(eval_seqs)
    recall = hits / n_eval
    ndcg = ndcg_sum / n_eval
    return recall, ndcg


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
    eval_steps=20,
    patience=5,
    eval_metric="metric",
    seed=42,
    semantic_embeddings=None,
    writer=None,
    eval_sequences=None,
):
    """Train SASRec model on user-item interaction sequences.

    Args:
        user_sequences: list of list of int — each user's item sequence (1-indexed)
        num_items: total number of items
        device: torch device
        save_path: path to save the fused embeddings .pt file
        hidden_units: SASRec embedding dimension (also CF embedding dim).
            If semantic_embeddings is provided, this is overridden by
            semantic_embeddings.shape[1] (bold fusion: no projection).
        num_heads: number of attention heads
        num_blocks: number of self-attention blocks
        max_len: maximum sequence length
        dropout: dropout rate
        epochs: number of training epochs
        lr: learning rate
        batch_size: training batch size
        weight_decay: L2 regularization weight
        eval_steps: log & check early-stopping every N epochs.
                    For large-item datasets (>100K items), set this to 20–50
                    to avoid spending most of training time just logging.
        patience: early-stopping patience counted in eval checkpoints.
                  Triggered only when the monitored metric/loss is *rising* vs
                  the previous checkpoint (not merely failing to improve) —
                  a stricter but faster criterion suitable for large datasets.
        eval_metric: early-stopping criterion — "metric" (default) or "loss".
                  "metric" — Recall@10/NDCG@10 composite (NDCG*0.6+Recall*0.4),
                             evaluated on val item every eval_steps epochs.
                  "loss"   — training loss only, no Recall/NDCG eval. Faster.
        seed: random seed
        semantic_embeddings: optional [num_items+1, D] tensor of frozen semantic
            embeddings (index 0 = zeros for padding). When provided, each item's
            representation = semantic_emb + learned CF emb (bold fusion, no
            projection). hidden_units must equal D; the trained model's fused
            embeddings (semantic + CF) are saved instead of CF-only.

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
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Build dataset and dataloader
    dataset = SasRecDataset(user_sequences, num_items, max_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    # Create model — if semantic_embeddings is provided, hidden_units must match
    if semantic_embeddings is not None:
        semantic_embeddings = semantic_embeddings.to(device)
        hidden_units = semantic_embeddings.shape[1]
        # Sanity: padding row (index 0) should be zeros
        assert semantic_embeddings[0].abs().max() < 1e-6, \
            "semantic_embeddings[0] (padding) should be all zeros"

    model = SASRec(
        num_items=num_items,
        hidden_units=hidden_units,
        num_heads=num_heads,
        num_blocks=num_blocks,
        max_len=max_len,
        dropout=dropout,
        semantic_embeddings=semantic_embeddings,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    print(f"\n{'='*60}")
    fusion_tag = "fused (semantic + CF)" if semantic_embeddings is not None else "CF-only"
    print(f"Training SASRec for item embeddings [{fusion_tag}]")
    print(f"  Items: {num_items}, Hidden dim: {hidden_units}")
    print(f"  Heads: {num_heads}, Blocks: {num_blocks}, Max len: {max_len}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"  Eval every {eval_steps} epochs | Early-stop patience: {patience} checks")
    print(f"  Early-stop metric: {eval_metric}")
    print(f"  Save path: {save_path}")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    best_state = None
    # best_metric initial value: -inf so the first checkpoint always wins.
    # (metric mode: NDCG*0.6+Recall*0.4 ≥ 0; loss mode: -avg_loss, can be
    # very negative when loss is large, so -1.0 would wrongly dominate.)
    best_metric = float("-inf")
    patience_counter = 0

    # Outer progress bar: one tick per epoch
    epoch_bar = tqdm(range(1, epochs + 1), desc="SASRec", unit="ep",
                     dynamic_ncols=True)

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Inner progress bar: one tick per batch
        batch_bar = tqdm(dataloader, desc=f"  Ep {epoch:4d}", unit="batch",
                         leave=False, dynamic_ncols=True)
        for batch in batch_bar:
            input_seq, pos_seq, neg_seq = [x.to(device) for x in batch]

            optimizer.zero_grad()
            loss = model(input_seq, pos_seq, neg_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            # Show running loss in the inner bar's postfix
            batch_bar.set_postfix(loss=f"{total_loss / num_batches:.5f}")

        avg_loss = total_loss / max(num_batches, 1)

        # Log to wandb (every epoch)
        if writer is not None:
            writer.log({"sasrec/train_loss": avg_loss, "sasrec/epoch": epoch})

        # Update outer bar postfix every batch (always visible)
        epoch_bar.set_postfix(loss=f"{avg_loss:.5f}", best=f"{best_loss:.5f}",
                              pat=f"{patience_counter}/{patience}")

        if epoch % eval_steps == 0 or epoch == 1:
            if eval_metric == "loss":
                # --- Loss mode: use training loss for early stopping ---
                # Lower is better
                metric = -avg_loss  # negate so "higher is better" logic holds
                recall, ndcg = 0.0, 0.0

                tqdm.write(f"  [Epoch {epoch:4d}/{epochs}] loss={avg_loss:.6f}"
                           f"  best_loss={best_loss:.6f}"
                           f"  pat={patience_counter}/{patience}")
            else:
                # --- Metric mode: Recall@10/NDCG@10 composite (default) ---
                eval_seqs = eval_sequences if eval_sequences is not None else user_sequences
                recall, ndcg = evaluate_recall_ndcg(
                    model, eval_seqs, num_items, device, max_len, k=10
                )
                # Composite metric for model selection & early stopping
                metric = ndcg * 0.6 + recall * 0.4

                tqdm.write(f"  [Epoch {epoch:4d}/{epochs}] loss={avg_loss:.6f}"
                           f"  Recall@10={recall:.4f}  NDCG@10={ndcg:.4f}"
                           f"  metric={metric:.4f}"
                           f"  best={best_metric:.4f}  pat={patience_counter}/{patience}")

            # Track the all-time best model and update patience counter.
            # NOTE: must check "is new best" BEFORE updating best_metric, and
            # update patience in the same if/else — a second `metric > best_metric`
            # check after the update would always be False (metric == best_metric).
            is_new_best = metric > best_metric
            if is_new_best:
                best_metric = metric
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                tqdm.write(f"  ↓ No improvement ({patience_counter}/{patience}), "
                           f"metric={metric:.4f} best={best_metric:.4f}")
                if patience_counter >= patience:
                    tqdm.write(f"  Early stopping at epoch {epoch} "
                               f"(best metric: {best_metric:.4f})")
                    break

            if writer is not None:
                log_dict = {
                    "sasrec/train_loss": avg_loss,
                    "sasrec/best_metric": best_metric,
                    "sasrec/patience_counter": patience_counter,
                }
                if eval_metric != "loss":
                    log_dict.update({
                        "sasrec/recall@10": recall,
                        "sasrec/ndcg@10": ndcg,
                        "sasrec/composite_metric": metric,
                    })
                writer.log(log_dict)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract and save embeddings
    if model.semantic_embeddings is not None:
        # Bold fusion: save semantic + CF embeddings
        fused_embeddings = extract_fused_embeddings(model, num_items, device)
        torch.save(fused_embeddings, save_path)
        print(f"\n✓ Fused embeddings (semantic + CF) saved to {save_path}")
        print(f"  Shape: {fused_embeddings.shape} (n_items × hidden_units)")
    else:
        cf_embeddings = extract_cf_embeddings(model, num_items, device)
        torch.save(cf_embeddings, save_path)
        print(f"\n✓ CF embeddings saved to {save_path}")
        print(f"  Shape: {cf_embeddings.shape} (n_items × hidden_units)")

    return model


def extract_cf_embeddings(model, num_items, device):
    """Extract CF-only item embeddings from a trained SASRec model.

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


def extract_fused_embeddings(model, num_items, device):
    """Extract fused (semantic + CF) item embeddings from a trained SASRec model.

    Args:
        model: trained SASRec model (must have semantic_embeddings buffer)
        num_items: total number of items
        device: torch device

    Returns:
        fused_embeddings: tensor of shape [num_items, hidden_units]
                          fused_embeddings[i] = semantic + CF for item ID (i+1)
    """
    model.eval()
    with torch.no_grad():
        item_ids = torch.LongTensor(list(range(1, num_items + 1))).to(device)
        fused_embeddings = model._item_emb(item_ids)  # [num_items, hidden_units]
    return fused_embeddings.cpu()


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
