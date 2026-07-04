"""
Custom data loading for pre-split datasets with semantic vectors.

Unlike the original load_data.py which expects Amazon/Steam raw data
processed through data_process.py, this module:
  1. Reads pre-split train/val/test text files (tab-separated, uid first)
  2. Loads pid_mapping.json (original_pid -> custom_id)
  3. Loads item embeddings from .pt file (built by build_embedding.py)
  4. Builds SID-based input sequences for TIGER training

Sequence file format: each line is tab-separated:
    uid\\titem1\\titem2\\t...\\titemN
  where uid is the user ID (first field), and items are custom IDs.
  Each line is already a complete training example (pre-split).
  The last item is the target, everything before it is the context.
"""

import json
import os
import pickle

import numpy as np
import torch
from tqdm import trange

from .load_data import (
    expand_id,
    expand_id_arr,
    get_unique_semantic_ids_by_extra_position,
    pad_sequence,
)


# ---------------------------------------------------------------------------
# Sequence file parsing
# ---------------------------------------------------------------------------

def load_sequence_file(filepath):
    """Load a pre-split sequence file.

    Format: each line is tab-separated, first field is uid, rest are item IDs.
        uid\\titem1\\titem2\\t...\\titemN

    Returns:
        sequences: list of lists of int (item IDs only, uid stripped)
        uids: list of int (user IDs, one per line)
    """
    sequences = []
    uids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Tab-separated: uid\titem1\titem2\t...
            parts = line.split("\t")
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            if len(items) >= 2:  # need at least 1 input + 1 target
                sequences.append(items)
                uids.append(uid)
    return sequences, uids


def load_all_split_files(data_dir, file_dict):
    """Load all split files from a dict of category -> filename.

    Returns dict of category -> (list of sequences, list of uids).
    """
    result = {}
    for category, filename in file_dict.items():
        filepath = os.path.join(data_dir, filename)
        seqs, uids = load_sequence_file(filepath)
        result[category] = (seqs, uids)
        print(f"  Loaded {len(seqs)} sequences from {filename}")
    return result


# ---------------------------------------------------------------------------
# PID mapping and embedding loading
# ---------------------------------------------------------------------------

def load_pid_mapping(data_dir, mapping_file):
    """Load pid_mapping.json (original_pid -> custom_id).

    Returns:
        pid_to_id: dict mapping original_pid (str) -> custom_id (int)
        id_to_pid: dict mapping custom_id (int) -> original_pid (str)
        num_items: total number of items
    """
    filepath = os.path.join(data_dir, mapping_file)
    with open(filepath, "r") as f:
        pid_to_id = json.load(f)

    # Convert keys/values to proper types
    pid_to_id = {str(k): int(v) for k, v in pid_to_id.items()}
    id_to_pid = {v: k for k, v in pid_to_id.items()}
    num_items = max(pid_to_id.values()) if pid_to_id else 0

    print(f"  Loaded pid_mapping: {len(pid_to_id)} items, max_id={num_items}")
    return pid_to_id, id_to_pid, num_items


def load_item_embedding(embedding_file, device):
    """Load item embedding tensor from .pt file.

    The .pt file is produced by build_embedding.py and contains:
        item_embedding: FloatTensor [num_items, dim]
        embedding_dim: int
        num_items: int
        standardized: bool

    Items are 1-indexed: row i corresponds to item with custom_id = i + 1.

    Returns:
        item_embedding: torch.Tensor of shape [num_items, dim] on device
        embedding_dim: int
        num_items: int
    """
    data = torch.load(embedding_file, map_location=device, weights_only=False)
    item_embedding = data["item_embedding"].to(device)
    embedding_dim = data["embedding_dim"]
    num_items = data["num_items"]
    standardized = data.get("standardized", False)

    print(f"  Loaded item_embedding: shape={item_embedding.shape}, "
          f"dim={embedding_dim}, num_items={num_items}, standardized={standardized}")
    return item_embedding, embedding_dim, num_items


# ---------------------------------------------------------------------------
# Compute id_split (seen/unseen items) from the sequences
# ---------------------------------------------------------------------------

def compute_id_split(train_sequences, val_sequences, test_sequences):
    """Compute the seen/unseen item split for SID construction.

    In the original pipeline, RQ-VAE is trained on 'seen' items
    (those that appear in training data). Unseen items get their SID
    assigned by the trained RQ-VAE in inference mode.

    Returns:
        id_split: dict with 'seen', 'unseen_val', 'unseen_test' arrays
    """
    # Collect all items from training sequences
    train_items = set()
    for seq in train_sequences:
        train_items.update(seq)

    # Collect val items (last item of each val sequence = the target)
    val_items = set()
    for seq in val_sequences:
        val_items.add(seq[-1])

    # Collect test items (last item of each test sequence = the target)
    test_items = set()
    for seq in test_sequences:
        test_items.add(seq[-1])

    train_items = np.array(sorted(train_items))
    unseen_val = np.array(sorted(val_items - set(train_items)))
    unseen_test = np.array(sorted(test_items - set(train_items)))

    print(f"  id_split: seen={len(train_items)}, "
          f"unseen_val={len(unseen_val)}, unseen_test={len(unseen_test)}")

    return {
        "seen": train_items,
        "unseen_val": unseen_val,
        "unseen_test": unseen_test,
    }


# ---------------------------------------------------------------------------
# Build training data from sequences (leverages existing load_data logic)
# ---------------------------------------------------------------------------

def generate_input_sequence_custom(
    user_sequence,
    item_2_semantic_id,
    max_items_per_seq,
    max_sequence_length,
    codebook_sizes,
    item_embedding,
    include_user_id=False,
    user_id_offset=None,
):
    """Generate input sequences from pre-split data.

    Each sequence is a list of item IDs where the last item is the target
    and everything before it is the context.

    Args:
        user_sequence: list of item IDs (the full sequence for one example)
        item_2_semantic_id: dict mapping item_id -> tuple of semantic IDs
        max_items_per_seq: max number of items in a sequence
        max_sequence_length: max total sequence length (items × codebook_depth)
        codebook_sizes: list of codebook sizes per level
        item_embedding: [n_item, dim] tensor
        include_user_id: whether to prepend user ID
        user_id_offset: offset for user ID tokens
    """
    # Truncate: keep at most max_items_per_seq history items + the last target item.
    # user_sequence[-1] is always the target; history = user_sequence[:-1]
    history = user_sequence[:-1]
    target = user_sequence[-1:]
    if len(history) > max_items_per_seq:
        history = history[-max_items_per_seq:]  # keep the most recent items
    user_sequence = history + target

    if include_user_id and user_id_offset is not None:
        user_id = user_id_offset  # simplified: no per-user hashing for custom data
        input_sids = [user_id]
        attention_mask_sids = [1]
        input_ids = [user_id]
        attention_mask_ids = [1]
    else:
        input_sids, input_ids, attention_mask_sids, attention_mask_ids = [], [], [], []

    input_embeddings, labels_sids, labels_ids, label_embeddings = [], [], [], []

    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            # Last item is the target/label
            labels_sids.extend(
                expand_id(item_2_semantic_id[user_sequence[i]], codebook_sizes)
            )
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            labels_ids.append(user_sequence[i])
            label_embeddings = this_item_embedding
        else:
            # Input items
            input_semantic_ids = expand_id(
                item_2_semantic_id[user_sequence[i]], codebook_sizes
            )
            input_sids.extend(input_semantic_ids)
            input_ids.append(user_sequence[i])
            attention_mask_sids.extend([1] * len(input_semantic_ids))
            attention_mask_ids.append(1)
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            input_embeddings.append(this_item_embedding)

    labels_sids = np.array(labels_sids)
    input_sids = np.array(
        pad_sequence(input_sids, max_sequence_length, pad_token=0)
    )
    attention_mask_sids = np.array(
        pad_sequence(attention_mask_sids, max_sequence_length, pad_token=0)
    )
    assert not np.any(labels_sids == 0)
    labels_sids[labels_sids == 0] = -100

    labels_ids = np.array(labels_ids)
    input_ids = np.array(pad_sequence(input_ids, max_sequence_length, pad_token=0))
    attention_mask_ids = np.array(
        pad_sequence(attention_mask_ids, max_sequence_length, pad_token=0)
    )

    # Pad input_embeddings to max_items_per_seq
    if len(input_embeddings) == 0:
        # Edge case: sequence of length 1 (no input, only target)
        # This shouldn't happen in practice since we filter len >= 2
        input_embeddings = torch.zeros(1, item_embedding.shape[1])
    else:
        padding_count = max_items_per_seq - len(input_embeddings)
        input_embeddings = torch.cat(
            input_embeddings
            + [torch.zeros_like(input_embeddings[0])] * padding_count,
            dim=0,
        )

    return (
        input_sids,
        input_ids,
        input_embeddings,
        attention_mask_sids,
        attention_mask_ids,
        labels_sids,
        labels_ids,
        label_embeddings,
    )


def build_dataset_from_sequences(
    sequences,
    item_2_semantic_id,
    item_embedding,
    method_config,
    max_items_per_seq,
    max_sequence_length,
    user_id_offset,
    codebook_sizes,
):
    """Build a dataset dict from a list of sequences.

    Each sequence in the list is already a complete training example:
    all items except the last are input, the last is the target.

    Returns a dict suitable for CustomDataset.
    """
    data = {
        "input_ids": [],
        "attention_mask_ids": [],
        "labels_ids": [],
        "input_embeddings": [],
        "label_embeddings": [],
        "input_sids": [],
        "labels_sids": [],
        "attention_mask_sids": [],
    }

    for seq in trange(len(sequences), desc="Building dataset"):
        (
            input_sids,
            input_ids,
            input_embeddings,
            attention_mask_sids,
            attention_mask_ids,
            labels_sids,
            labels_ids,
            labels_embeddings,
        ) = generate_input_sequence_custom(
            sequences[seq],
            item_2_semantic_id,
            max_items_per_seq,
            max_sequence_length,
            codebook_sizes,
            item_embedding,
            include_user_id=method_config.get("include_user_id", False),
            user_id_offset=user_id_offset,
        )

        data["input_sids"].append(input_sids)
        data["attention_mask_sids"].append(attention_mask_sids)
        data["labels_sids"].append(labels_sids)
        data["input_ids"].append(input_ids)
        data["input_embeddings"].append(input_embeddings.cpu())
        data["attention_mask_ids"].append(attention_mask_ids)
        data["labels_ids"].append(labels_ids)
        data["label_embeddings"].append(labels_embeddings.cpu())

    # Convert lists to tensors
    for sub_key in data.keys():
        if sub_key in ["label_embeddings"]:
            if len(data[sub_key]) > 0:
                data[sub_key] = torch.cat(data[sub_key])
            else:
                data[sub_key] = torch.tensor([])
        elif sub_key in ["input_embeddings"]:
            if len(data[sub_key]) > 0:
                data[sub_key] = torch.stack(data[sub_key])
            else:
                data[sub_key] = torch.tensor([])
        else:
            data[sub_key] = torch.tensor(data[sub_key], dtype=torch.long)

    return data


# ---------------------------------------------------------------------------
# Main data loading function
# ---------------------------------------------------------------------------

def load_custom_data(
    id_save_location,
    sequences_dict,
    id_split,
    item_embedding,
    method_config,
    max_length=258,
    codebook_sizes=None,
    max_items_per_seq=20,
):
    """Load custom data for TIGER training.

    This mirrors load_data.load_data() but works with pre-split sequences.

    Args:
        id_save_location: path to the semantic ID pickle file
        sequences_dict: dict of category -> list of item sequences
            Keys like 'train_all', 'val_all', 'val_normal', 'val_surprise',
            'test_all', 'test_normal', 'test_surprise'.
        id_split: dict with 'seen', 'unseen_val', 'unseen_test'
        item_embedding: [n_item, dim] tensor
        method_config: method configuration dict
        max_length: max sequence length for padding
        codebook_sizes: codebook sizes (int or list)
        max_items_per_seq: max items per sequence

    Returns:
        datasets: dict of category -> dataset dict
        semantic_id_info: dict with all_semantic_ids, unseen_semantic_ids, etc.
    """
    n_semantic_codebook = 3

    semantic_ids = pickle.load(open(id_save_location, "rb"))

    # Dynamically determine n_semantic_codebook
    if isinstance(semantic_ids, np.ndarray) and semantic_ids.ndim == 2:
        n_semantic_codebook = semantic_ids.shape[1]

    # Default codebook size
    if codebook_sizes is None:
        codebook_sizes = 256

    item_2_semantic_id, max_last_semantic_ids = (
        get_unique_semantic_ids_by_extra_position(semantic_ids, codebook_sizes)
    )

    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))

    n_codebook = n_semantic_codebook + 1

    # Compute user_id_offset
    if isinstance(codebook_sizes, int):
        user_id_offset = 1 + n_semantic_codebook * codebook_sizes + last_codebook_size
    else:
        user_id_offset = 1 + sum(codebook_sizes) + last_codebook_size

    # Build semantic ID arrays for evaluation
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    val_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in unseen_val])
    expand_id_arr(val_unseen_semantic_ids, codebook_sizes)
    test_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in unseen_test])
    expand_id_arr(test_unseen_semantic_ids, codebook_sizes)
    seen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in seen])
    expand_id_arr(seen_semantic_ids, codebook_sizes)
    all_semantic_ids = np.array(
        [item_2_semantic_id[idx] for idx in item_2_semantic_id.keys()]
    )
    expand_id_arr(all_semantic_ids, codebook_sizes)

    # Build datasets for each category
    datasets = {}
    for category, sequences in sequences_dict.items():
        print(f"Building dataset for category '{category}': {len(sequences)} sequences")
        datasets[category] = build_dataset_from_sequences(
            sequences,
            item_2_semantic_id,
            item_embedding,
            method_config,
            max_items_per_seq,
            max_length,
            user_id_offset,
            codebook_sizes,
        )

    semantic_id_info = {
        "seen_semantic_ids": seen_semantic_ids,
        "val_unseen_semantic_ids": val_unseen_semantic_ids,
        "test_unseen_semantic_ids": test_unseen_semantic_ids,
        "all_semantic_ids": all_semantic_ids,
        "max_last_semantic_ids": max_last_semantic_ids,
        "n_semantic_codebook": n_semantic_codebook,
        "n_codebook": n_codebook,
        "item2sid": all_semantic_ids,
        "user_id_offset": user_id_offset,
    }

    return datasets, semantic_id_info
