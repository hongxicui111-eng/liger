#!/usr/bin/env python3
"""
Build item embedding tensor from caption embedding files.

This is an EXTERNAL pre-processing program — run it BEFORE the TIGER pipeline.
It reads caption embedding files (CSV or parquet) containing photo_id and
cap_embedding columns, maps original PIDs to custom IDs via pid_mapping.json,
and saves a PyTorch .pt tensor file for direct use by build_custom_sid.py
and run_custom.py.

Optimized for large-scale data:
  - Streaming: processes one file at a time, no intermediate dict
  - Vectorized: uses pandas/numpy operations instead of iterrows
  - Fast parsing: uses json.loads instead of ast.literal_eval (3-5x faster)
  - Memory efficient: only holds the output tensor + one file's data at a time
  - Preflight check: samples IDs from both sources before full processing

Output: A single .pt file containing:
  - item_embedding: FloatTensor of shape [num_items, dim]
      Row i corresponds to item with custom_id = i + 1 (1-indexed items).
  - embedding_dim: int
  - num_items: int
  - standardized: bool
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


def log(msg: str):
    print(f"[build_embedding] {msg}", flush=True)


def _parse_embedding_str_batch(str_series: pd.Series) -> np.ndarray:
    """Batch-parse embedding strings using json.loads (3-5x faster than ast.literal_eval).

    Handles string-encoded lists like '[0.1, 0.2, ...]' or '(0.1, 0.2, ...)'.
    Falls back to np.fromstring for space-separated formats.
    """
    def _safe_json(s):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None

    parsed = str_series.apply(_safe_json)
    failed = parsed.isna()

    if failed.sum() == 0:
        return np.stack(parsed.values)
    else:
        def _safe_fromstring(s):
            try:
                s = s.strip().replace("(", "[").replace(")", "]")
                return json.loads(s)
            except Exception:
                try:
                    return np.fromstring(s.strip("[]() "), sep=",").tolist()
                except Exception:
                    return None

        parsed[failed] = str_series[failed].apply(_safe_fromstring)
        still_failed = parsed.isna()
        if still_failed.sum() > 0:
            log(f"WARNING: {still_failed.sum()} embeddings could not be parsed")
            parsed[still_failed] = None

        return np.stack([np.array(p, dtype=np.float32) if p is not None
                         else np.zeros(1, dtype=np.float32) for p in parsed.values])


def preflight_id_check(caption_dir: str, pid_to_id: dict, max_sample: int = 10):
    """Preflight diagnostic: sample IDs from both sources and check overlap.

    Prints sample keys/values from pid_mapping and sample photo_ids from caption
    files, so the user can immediately see any mismatch.
    """
    # --- Sample from pid_mapping ---
    sample_mapping_keys = list(pid_to_id.keys())[:max_sample]
    sample_mapping_vals = [pid_to_id[k] for k in sample_mapping_keys]
    log(f"pid_mapping 前 {max_sample} 个 key → value:")
    for k, v in zip(sample_mapping_keys, sample_mapping_vals):
        log(f"    {k!r} (type={type(k).__name__}) → {v}")

    # --- Sample from caption files ---
    files = glob.glob(os.path.join(caption_dir, "*.csv"))
    if not files:
        files = glob.glob(os.path.join(caption_dir, "*.parquet"))
    if not files:
        log("ERROR: No caption files found!")
        return

    sample_file = files[0]
    if sample_file.endswith(".parquet"):
        df_sample = pd.read_parquet(sample_file, columns=["photo_id"])
    else:
        df_sample = pd.read_csv(sample_file, usecols=["photo_id"], nrows=max_sample)

    sample_photo_ids = df_sample["photo_id"].values[:max_sample]
    log(f"caption 文件 ({os.path.basename(sample_file)}) 前 {max_sample} 个 photo_id:")
    for pid in sample_photo_ids:
        log(f"    {pid!r} (type={type(pid).__name__}), str={str(pid)!r}")

    # --- Quick overlap check ---
    sample_pid_set = {str(p) for p in sample_photo_ids}
    overlap = sample_pid_set & set(pid_to_id.keys())
    log(f"采样范围 overlap: {len(overlap)}/{len(sample_pid_set)} 个 photo_id 在 pid_mapping 中找到")

    if len(overlap) == 0:
        log("=" * 60)
        log("WARNING: 采样范围内没有任何 photo_id 能在 pid_mapping 中匹配!")
        log("可能的原因:")
        log("  1. photo_id 和 pid_mapping 的 key 处于不同 ID 空间")
        log("  2. photo_id 列名不对 (不是 'photo_id')")
        log("  3. pid_mapping 的 key 类型与 photo_id 不一致 (如 int vs string)")
        log("  4. caption 文件中的 ID 需要额外转换才能与 pid_mapping 对齐")
        log("=" * 60)


def build_embedding_tensor_streaming(
    caption_dir: str,
    pid_to_id: dict,
    num_items: int,
    embedding_dim: int,
) -> np.ndarray:
    """Build item_embedding tensor by streaming caption files one at a time.

    No intermediate dict — fills the output array directly.
    Memory: only the output array [num_items, dim] + one file's DataFrame at a time.
    """
    item_embedding = np.zeros((num_items, embedding_dim), dtype=np.float32)
    matched_total = 0
    unmatched_total = 0
    # Track a few unmatched PIDs for debugging
    unmatched_pid_samples = []

    files = glob.glob(os.path.join(caption_dir, "*.csv"))
    if not files:
        files = glob.glob(os.path.join(caption_dir, "*.parquet"))

    if not files:
        log("ERROR: No caption files found!")
        return item_embedding, 0, 0

    log(f"找到 {len(files)} 个 caption 文件，开始流式构建 embedding 张量...")
    t0 = time.time()

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        matched_before = matched_total

        if fpath.endswith(".parquet"):
            df_cap = pd.read_parquet(fpath, columns=["photo_id", "cap_embedding"])
        else:
            df_cap = pd.read_csv(fpath, usecols=["photo_id", "cap_embedding"])

        n_rows = len(df_cap)
        log(f"  [{idx}/{len(files)}] {fname} — {n_rows} rows, processing...")

        # Parse embeddings
        pids = df_cap["photo_id"].values

        # Diagnostic: check cap_embedding dtype and first value
        first_emb = df_cap["cap_embedding"].iloc[0]
        log(f"    cap_embedding dtype={df_cap['cap_embedding'].dtype}, "
            f"first value type={type(first_emb).__name__}")
        if isinstance(first_emb, (list, np.ndarray)):
            log(f"    first value len/shape="
                f"{len(first_emb) if isinstance(first_emb, list) else first_emb.shape}, "
                f"sample={np.array(first_emb)[:3] if hasattr(first_emb, '__len__') and len(first_emb) > 0 else 'empty'}")
        elif isinstance(first_emb, (bytes, bytearray)):
            log(f"    first value is bytes, len={len(first_emb)}, first 40 bytes={first_emb[:40]}")
        elif isinstance(first_emb, str):
            log(f"    first value is string, len={len(first_emb)}, first 80 chars={first_emb[:80]}")
        else:
            log(f"    first value repr={repr(first_emb)[:200]}")

        if df_cap["cap_embedding"].dtype == object:
            # object dtype could be strings OR ndarrays (parquet stores arrays as object)
            first_val = df_cap["cap_embedding"].iloc[0]
            if isinstance(first_val, (np.ndarray, list)):
                # Already parsed arrays — just stack them
                try:
                    embeddings = np.stack(df_cap["cap_embedding"].values)
                    log(f"    object→np.stack succeeded, shape={embeddings.shape}")
                except Exception as e:
                    log(f"    object→np.stack failed: {e}, element-wise fallback")
                    embeddings = np.array(
                        [np.asarray(e, dtype=np.float32) for e in df_cap["cap_embedding"].values]
                    )
                    log(f"    element-wise shape={embeddings.shape}")
            elif isinstance(first_val, (bytes, bytearray)):
                # Binary-encoded — try to decode as numpy
                log(f"    object dtype with bytes values, decoding...")
                embeddings = np.array(
                    [np.frombuffer(e, dtype=np.float32) for e in df_cap["cap_embedding"].values]
                )
                log(f"    frombuffer shape={embeddings.shape}")
            else:
                # String-encoded embeddings — need text parsing
                embeddings = _parse_embedding_str_batch(df_cap["cap_embedding"])
        else:
            try:
                embeddings = np.stack(df_cap["cap_embedding"].values)
                log(f"    np.stack succeeded, shape={embeddings.shape}")
            except Exception as e:
                log(f"    np.stack failed: {e}, trying element-wise conversion")
                embeddings = np.array(
                    [np.array(e, dtype=np.float32) for e in df_cap["cap_embedding"].values]
                )
                log(f"    element-wise shape={embeddings.shape}")

        embeddings = embeddings.astype(np.float32)
        log(f"    final embeddings shape={embeddings.shape}, "
            f"nonzero={np.count_nonzero(embeddings)}/{embeddings.size}, "
            f"min={embeddings.min():.4f}, max={embeddings.max():.4f}")

        # Map PIDs to custom_ids and fill the array
        # Try multiple key formats to handle type mismatches
        for i in range(len(pids)):
            raw_pid = pids[i]
            matched = False

            # Try exact string match
            pid_str = str(raw_pid)
            if pid_str in pid_to_id:
                custom_id = pid_to_id[pid_str]
                matched = True
            # Try int match (if raw_pid is float like 12345.0)
            elif isinstance(raw_pid, (float, np.floating)):
                pid_str_int = str(int(raw_pid))
                if pid_str_int in pid_to_id:
                    custom_id = pid_to_id[pid_str_int]
                    matched = True

            if matched and 1 <= custom_id <= num_items:
                item_embedding[custom_id - 1] = embeddings[i]
                matched_total += 1
            else:
                unmatched_total += 1
                # Save first few unmatched PIDs for diagnosis
                if len(unmatched_pid_samples) < 5:
                    unmatched_pid_samples.append(
                        f"raw={raw_pid!r}, str={str(raw_pid)!r}, "
                        f"int_str={str(int(raw_pid)) if isinstance(raw_pid, (float, np.floating)) else 'N/A'}"
                    )

        elapsed = time.time() - t0
        log(f"  [{idx}/{len(files)}] {fname} | +{matched_total - matched_before} matched | "
            f"累计 {matched_total} | 耗时 {elapsed:.1f}s")

    log(f"构建完毕，共 {matched_total} matched, {unmatched_total} unmatched, "
        f"总耗时 {time.time() - t0:.1f}s")

    if matched_total == 0 and unmatched_pid_samples:
        log("前几个未匹配的 photo_id 示例:")
        for s in unmatched_pid_samples:
            log(f"  {s}")

    return item_embedding, matched_total, unmatched_total


def main():
    parser = argparse.ArgumentParser(
        description="Build item embedding tensor from caption files"
    )
    parser.add_argument("--caption_dir", type=str, default="/llm_reco_ssd/huangrui06/llmrec_data/reco_log_csv_1w_processed/qwen3_infer_output",
                        help="Directory containing caption CSV/parquet files")
    parser.add_argument("--pid_mapping_file", type=str, default="/home/cuihongxi/get_caption/10W_datacaption/results/1Wdata/7days/compare/two_qwen/rec_data/pid_mapping.json",
                        help="Path to pid_mapping.json (original_pid -> custom_id)")
    parser.add_argument("--output_file", type=str, default="/home/cuihongxi/get_caption/pid_embeddings/item_embedding.pt",
                        help="Output .pt file path")
    parser.add_argument("--no_standardize", action="store_true",
                        help="Skip StandardScaler (default: standardize=True, pass this flag to save raw embeddings)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load pid_mapping (lightweight, just int key-value pairs)
    # -----------------------------------------------------------------------
    log("Loading pid_mapping...")
    with open(args.pid_mapping_file, "r") as f:
        pid_to_id_raw = json.load(f)

    # Keep BOTH raw and str versions for matching flexibility
    pid_to_id = {}
    for k, v in pid_to_id_raw.items():
        pid_to_id[str(k)] = int(v)
        # Also register the int version if key was numeric
        try:
            int_k = int(k)
            pid_to_id[str(int_k)] = int(v)
        except (ValueError, TypeError):
            pass

    num_items = max(pid_to_id.values()) if pid_to_id else 0
    log(f"PID mapping: {len(pid_to_id_raw)} raw entries, "
        f"{len(pid_to_id)} lookup entries (after normalizing), max custom_id = {num_items}")

    # Show raw JSON key types for debugging
    sample_raw = list(pid_to_id_raw.items())[:3]
    log(f"pid_mapping 原始 JSON 前 3 个 key 的类型:")
    for k, v in sample_raw:
        log(f"    key={k!r} (type={type(k).__name__}) → value={v!r} (type={type(v).__name__})")

    # -----------------------------------------------------------------------
    # Step 2: Preflight ID check (before expensive processing)
    # -----------------------------------------------------------------------
    preflight_id_check(args.caption_dir, pid_to_id)

    # -----------------------------------------------------------------------
    # Step 3: Detect embedding dimension from first file's first row
    # -----------------------------------------------------------------------
    files = glob.glob(os.path.join(args.caption_dir, "*.csv"))
    if not files:
        files = glob.glob(os.path.join(args.caption_dir, "*.parquet"))
    if not files:
        log("ERROR: No caption files found!")
        sys.exit(1)

    sample_file = files[0]
    if sample_file.endswith(".parquet"):
        df_sample = pd.read_parquet(sample_file, columns=["cap_embedding"])
    else:
        df_sample = pd.read_csv(sample_file, usecols=["cap_embedding"], nrows=1)

    sample_emb = df_sample["cap_embedding"].iloc[0]
    if isinstance(sample_emb, str):
        sample_emb = np.array(json.loads(sample_emb), dtype=np.float32)
    else:
        sample_emb = np.array(sample_emb, dtype=np.float32)
    embedding_dim = sample_emb.shape[0]
    log(f"Embedding dimension: {embedding_dim}")

    # -----------------------------------------------------------------------
    # Step 4: Build item_embedding tensor (streaming, no intermediate dict)
    # -----------------------------------------------------------------------
    item_embedding, matched, unmatched = build_embedding_tensor_streaming(
        args.caption_dir, pid_to_id, num_items, embedding_dim
    )

    if matched == 0:
        log("ERROR: No embeddings matched any PID in mapping!")
        log("请检查上面的预检诊断输出，确认 photo_id 和 pid_mapping 的 key 是否在同一 ID 空间")
        sys.exit(1)

    # Check how many items have zero embeddings (missing)
    zero_rows = (item_embedding == 0).all(axis=1).sum()
    if zero_rows > 0:
        log(f"WARNING: {zero_rows}/{num_items} items have zero embeddings (missing from caption files)")

    # -----------------------------------------------------------------------
    # Step 5: Standardize (default: True, matches TIGER pipeline)
    # -----------------------------------------------------------------------
    do_standardize = not args.no_standardize
    if do_standardize:
        from sklearn.preprocessing import StandardScaler
        log("Applying StandardScaler to embeddings...")
        item_embedding = StandardScaler().fit_transform(item_embedding)
    else:
        log("Skipping StandardScaler (raw embeddings saved)")

    # -----------------------------------------------------------------------
    # Step 6: Save as .pt file
    # -----------------------------------------------------------------------
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    item_embedding_torch = torch.from_numpy(item_embedding).float()
    torch.save({
        "item_embedding": item_embedding_torch,
        "embedding_dim": embedding_dim,
        "num_items": num_items,
        "standardized": do_standardize,
    }, args.output_file)

    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    log(f"Saved to {args.output_file} ({file_size_mb:.1f} MB)")
    log(f"  item_embedding: shape={item_embedding_torch.shape}, dtype={item_embedding_torch.dtype}")
    log(f"  num_items={num_items}, embedding_dim={embedding_dim}, standardized={do_standardize}")
    log("Done!")


if __name__ == "__main__":
    main()
