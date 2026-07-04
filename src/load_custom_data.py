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
import multiprocessing as mp
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm, trange

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
    """Generate input sequences from pre-split data (single-sequence, kept for
    backward compatibility).  New code should prefer build_dataset_from_sequences
    which calls the vectorised batch path internally.
    """
    # Truncate: keep at most max_items_per_seq history items + the last target item.
    history = user_sequence[:-1]
    target = user_sequence[-1:]
    if len(history) > max_items_per_seq:
        history = history[-max_items_per_seq:]
    user_sequence = history + target

    if include_user_id and user_id_offset is not None:
        user_id = user_id_offset
        input_sids = [user_id]
        attention_mask_sids = [1]
        input_ids = [user_id]
        attention_mask_ids = [1]
    else:
        input_sids, input_ids, attention_mask_sids, attention_mask_ids = [], [], [], []

    input_embeddings, labels_sids, labels_ids, label_embeddings = [], [], [], []

    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            labels_sids.extend(
                expand_id(item_2_semantic_id[user_sequence[i]], codebook_sizes)
            )
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            labels_ids.append(user_sequence[i])
            label_embeddings = this_item_embedding
        else:
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

    if len(input_embeddings) == 0:
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


# ---------------------------------------------------------------------------
# Vectorised batch builder (fast path)
# ---------------------------------------------------------------------------

def _build_chunk(args):
    """Worker function for multiprocessing — processes a chunk of sequences.

    All heavy objects are passed as plain Python/numpy objects so they can be
    pickled safely across process boundaries (no GPU tensors here).

    Returns numpy arrays for each field (no torch tensors — caller converts).
    """
    (
        sequences_chunk,
        sid_array,          # np.ndarray [n_items+1, n_levels], row 0 unused (pad)
        emb_np,             # np.ndarray [n_items, emb_dim], CPU, 0-indexed
        max_items_per_seq,
        max_sequence_length,
        n_levels,
        include_user_id,
        user_id_offset,
    ) = args

    N = len(sequences_chunk)
    emb_dim = emb_np.shape[1]

    # Pre-allocate output arrays
    out_input_sids        = np.zeros((N, max_sequence_length), dtype=np.int64)
    out_attn_sids         = np.zeros((N, max_sequence_length), dtype=np.int64)
    out_input_ids         = np.zeros((N, max_sequence_length), dtype=np.int64)
    out_attn_ids          = np.zeros((N, max_sequence_length), dtype=np.int64)
    out_labels_sids       = np.zeros((N, n_levels),            dtype=np.int64)
    out_labels_ids        = np.zeros((N,),                     dtype=np.int64)
    out_input_embeddings  = np.zeros((N, max_items_per_seq, emb_dim), dtype=np.float32)
    out_label_embeddings  = np.zeros((N, emb_dim),             dtype=np.float32)

    for i, seq in enumerate(sequences_chunk):
        # --- truncate ---
        history = seq[:-1]
        target_item = seq[-1]
        if len(history) > max_items_per_seq:
            history = history[-max_items_per_seq:]

        # --- user-id prefix ---
        sid_prefix_len = 0
        id_prefix_len  = 0
        if include_user_id and user_id_offset is not None:
            out_input_sids[i, 0] = user_id_offset
            out_attn_sids [i, 0] = 1
            out_input_ids [i, 0] = user_id_offset
            out_attn_ids  [i, 0] = 1
            sid_prefix_len = 1
            id_prefix_len  = 1

        # --- history items ---
        n_hist = len(history)
        sid_pos = sid_prefix_len
        for j, item_id in enumerate(history):
            sids = sid_array[item_id]          # shape [n_levels]
            out_input_sids[i, sid_pos : sid_pos + n_levels] = sids
            out_attn_sids [i, sid_pos : sid_pos + n_levels] = 1
            sid_pos += n_levels

            out_input_ids[i, id_prefix_len + j] = item_id
            out_attn_ids [i, id_prefix_len + j] = 1

            # embedding (item_id is 1-indexed, emb_np is 0-indexed)
            out_input_embeddings[i, j] = emb_np[item_id - 1]

        # --- target item ---
        out_labels_sids[i]      = sid_array[target_item]
        out_labels_ids[i]       = target_item
        out_label_embeddings[i] = emb_np[target_item - 1]

    return (
        out_input_sids,
        out_attn_sids,
        out_input_ids,
        out_attn_ids,
        out_labels_sids,
        out_labels_ids,
        out_input_embeddings,
        out_label_embeddings,
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
    num_workers: int = 0,
):
    """Build a dataset dict from a list of sequences — vectorised + parallel.

    Args:
        sequences: list of item-ID lists (each sequence ends with the target)
        item_2_semantic_id: dict item_id -> tuple of raw SID indices
        item_embedding: torch.Tensor [n_items, emb_dim] (CPU or GPU)
        method_config: method configuration dict
        max_items_per_seq: max history length (truncate from the left)
        max_sequence_length: padded SID sequence length
        user_id_offset: token offset for user-id prepend
        codebook_sizes: int or list[int]
        num_workers: parallel worker processes (0 = auto = cpu_count // 2)

    Returns a dict of torch tensors suitable for CustomDataset.
    """
    N = len(sequences)
    if N == 0:
        emb_dim = item_embedding.shape[1]
        return {
            "input_sids":        torch.zeros(0, max_sequence_length, dtype=torch.long),
            "attention_mask_sids": torch.zeros(0, max_sequence_length, dtype=torch.long),
            "input_ids":         torch.zeros(0, max_sequence_length, dtype=torch.long),
            "attention_mask_ids": torch.zeros(0, max_sequence_length, dtype=torch.long),
            "labels_sids":       torch.zeros(0, dtype=torch.long),
            "labels_ids":        torch.zeros(0, dtype=torch.long),
            "input_embeddings":  torch.zeros(0, max_items_per_seq, emb_dim),
            "label_embeddings":  torch.zeros(0, emb_dim),
        }

    include_user_id = method_config.get("include_user_id", False)

    # ------------------------------------------------------------------
    # 1. Build a compact numpy SID lookup array indexed by item_id.
    #    sid_array[item_id] = expanded SID tuple (already offset-added).
    #    Row 0 is unused (item IDs are 1-indexed).
    # ------------------------------------------------------------------
    # Determine n_levels from item_2_semantic_id
    sample_item = next(iter(item_2_semantic_id))
    n_levels = len(item_2_semantic_id[sample_item])

    max_item_id = max(item_2_semantic_id.keys())
    sid_array = np.zeros((max_item_id + 1, n_levels), dtype=np.int64)
    for item_id, raw_sids in item_2_semantic_id.items():
        sid_array[item_id] = expand_id(raw_sids, codebook_sizes)

    # ------------------------------------------------------------------
    # 2. Move item_embedding to CPU numpy once (avoids per-row GPU sync).
    # ------------------------------------------------------------------
    if isinstance(item_embedding, torch.Tensor):
        emb_np = item_embedding.detach().cpu().numpy().astype(np.float32)
    else:
        emb_np = np.asarray(item_embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Split sequences into chunks for parallel processing.
    # ------------------------------------------------------------------
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() // 2)

    chunk_size = max(1, (N + num_workers - 1) // num_workers)
    chunks = [sequences[s : s + chunk_size] for s in range(0, N, chunk_size)]
    actual_workers = len(chunks)

    worker_args = [
        (
            chunk,
            sid_array,
            emb_np,
            max_items_per_seq,
            max_sequence_length,
            n_levels,
            include_user_id,
            user_id_offset,
        )
        for chunk in chunks
    ]

    print(f"  Building {N} sequences with {actual_workers} workers "
          f"(chunk_size={chunk_size}) ...")

    if actual_workers == 1:
        # Single-process path (avoids fork overhead for small datasets)
        results = [_build_chunk(worker_args[0])]
    else:
        with mp.Pool(processes=actual_workers) as pool:
            results = list(tqdm(
                pool.imap(_build_chunk, worker_args),
                total=actual_workers,
                desc="Building dataset (parallel)",
            ))

    # ------------------------------------------------------------------
    # 4. Concatenate chunk results and convert to torch tensors.
    # ------------------------------------------------------------------
    def _cat(idx):
        return np.concatenate([r[idx] for r in results], axis=0)

    input_sids_np       = _cat(0)
    attn_sids_np        = _cat(1)
    input_ids_np        = _cat(2)
    attn_ids_np         = _cat(3)
    labels_sids_np      = _cat(4)
    labels_ids_np       = _cat(5)
    input_emb_np        = _cat(6)
    label_emb_np        = _cat(7)

    data = {
        "input_sids":          torch.from_numpy(input_sids_np),
        "attention_mask_sids": torch.from_numpy(attn_sids_np),
        "input_ids":           torch.from_numpy(input_ids_np),
        "attention_mask_ids":  torch.from_numpy(attn_ids_np),
        "labels_sids":         torch.from_numpy(labels_sids_np),
        "labels_ids":          torch.from_numpy(labels_ids_np),
        "input_embeddings":    torch.from_numpy(input_emb_np),
        "label_embeddings":    torch.from_numpy(label_emb_np),
    }

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
