#!/usr/bin/env python3
"""
SID Collision Rate Analyzer
============================
Given a SID pickle file (produced by build_custom_sid.py or run_custom.py),
analyze how many Semantic IDs collide (i.e., multiple items share the same
SID tuple).

SID file format: pickle containing a numpy array of shape [num_items, num_levels].
Each row is a semantic ID tuple, e.g., (c0, c1, c2) for 3-level RQ-VAE.

Usage:
    python check_sid_collision.py \
        --sid_file /home/sunyijia/cuihongxi/liger/ID_generation/ID/custom_semantic_0.pkl

    # Also compare with another SID file:
    python check_sid_collision.py \
        --sid_file ./ID_generation/ID/custom_semantic_0.pkl \
        --sid_file_2 ./ID_generation/ID/custom_semantic_0_alt.pkl
"""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np


def load_sid_file(filepath):
    """Load SID pickle file and return numpy array of shape [num_items, num_levels]."""
    with open(filepath, "rb") as f:
        sid_array = pickle.load(f)

    if not isinstance(sid_array, np.ndarray):
        raise TypeError(
            f"Expected numpy array in pickle, got {type(sid_array).__name__}"
        )

    if sid_array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {sid_array.ndim}D array with shape {sid_array.shape}"
        )

    return sid_array


def analyze_collisions(sid_array, label=""):
    """Analyze SID collision rate for a given SID array.

    Args:
        sid_array: numpy array of shape [num_items, num_levels]
        label: optional label for printing

    Returns:
        dict with collision statistics
    """
    num_items, num_levels = sid_array.shape
    prefix = f"[{label}] " if label else ""

    print(f"\n{'='*70}")
    print(f"{prefix}SID Collision Analysis")
    print(f"{'='*70}")
    print(f"  Number of items:   {num_items}")
    print(f"  SID depth (levels): {num_levels}")
    print(f"  Array shape:       {sid_array.shape}")
    print(f"  Value range:       [{sid_array.min()}, {sid_array.max()}]")

    # Per-level statistics
    print(f"\n  --- Per-Level Statistics ---")
    for level in range(num_levels):
        col = sid_array[:, level]
        unique_count = len(np.unique(col))
        max_val = col.max()
        min_val = col.min()
        print(
            f"  Level {level}: unique={unique_count}, range=[{min_val}, {max_val}]"
        )

    # Full SID tuple analysis
    sid_tuples = [tuple(row) for row in sid_array]
    sid_counter = Counter(sid_tuples)

    unique_sids = len(sid_counter)
    collision_count = sum(1 for count in sid_counter.values() if count > 1)
    total_collided_items = sum(count for count in sid_counter.values() if count > 1)

    print(f"\n  --- Full SID Tuple Analysis ---")
    print(f"  Total items:            {num_items}")
    print(f"  Unique SIDs:            {unique_sids}")
    print(f"  Collision groups:       {collision_count}")
    print(f"  Items in collision:     {total_collided_items}")
    print(f"  Collision rate:         {collision_count / unique_sids * 100:.2f}% "
          f"({collision_count}/{unique_sids} unique SIDs have collisions)")
    print(f"  Item collision rate:    {total_collided_items / num_items * 100:.2f}% "
          f"({total_collided_items}/{num_items} items share a SID with another item)")

    # Theoretical space
    level_unique_counts = [len(np.unique(sid_array[:, l])) for l in range(num_levels)]
    # Actually, the theoretical space is codebook_size^num_levels, but we may
    # not know codebook_size from the data alone. Let's estimate from max values.
    estimated_codebook_sizes = [sid_array[:, l].max() + 1 for l in range(num_levels)]
    theoretical_space = 1
    for cs in estimated_codebook_sizes:
        theoretical_space *= cs
    utilization = unique_sids / theoretical_space * 100

    print(f"\n  --- Theoretical Space ---")
    for level in range(num_levels):
        print(f"  Level {level} codebook size (est.): {estimated_codebook_sizes[level]}")
    print(f"  Theoretical SID space:  {theoretical_space}")
    print(f"  Space utilization:      {utilization:.4f}%")

    # Collision size distribution
    print(f"\n  --- Collision Size Distribution ---")
    collision_sizes = Counter(sid_counter.values())
    for size in sorted(collision_sizes.keys()):
        count = collision_sizes[size]
        print(f"  {size} items share same SID: {count} groups")

    # Top collisions (most items sharing same SID)
    top_n = 10
    print(f"\n  --- Top {top_n} Most Crowded SIDs ---")
    most_common = sid_counter.most_common(top_n)
    for rank, (sid_tuple, count) in enumerate(most_common, 1):
        if count == 1:
            break
        clean_sid = tuple(int(x) for x in sid_tuple)
        print(f"  #{rank}: SID={clean_sid}, {count} items")

    # Per-level collision analysis (partial SID collisions)
    print(f"\n  --- Partial SID Collision (prefix) ---")
    for prefix_len in range(1, num_levels + 1):
        prefix_tuples = [tuple(row[:prefix_len]) for row in sid_array]
        prefix_counter = Counter(prefix_tuples)
        prefix_unique = len(prefix_counter)
        prefix_collision_groups = sum(
            1 for c in prefix_counter.values() if c > 1
        )
        prefix_collided_items = sum(
            c for c in prefix_counter.values() if c > 1
        )
        print(
            f"  Prefix length {prefix_len}: unique={prefix_unique}, "
            f"collision_groups={prefix_collision_groups}, "
            f"collided_items={prefix_collided_items} "
            f"({prefix_collided_items/num_items*100:.1f}%)"
        )

    return {
        "num_items": num_items,
        "num_levels": num_levels,
        "unique_sids": unique_sids,
        "collision_groups": collision_count,
        "total_collided_items": total_collided_items,
        "collision_rate": collision_count / unique_sids * 100,
        "item_collision_rate": total_collided_items / num_items * 100,
    }


def compare_sid_files(sid_array_1, sid_array_2, label1, label2):
    """Compare collision rates between two SID files."""
    print(f"\n{'='*70}")
    print(f"SID File Comparison: {label1} vs {label2}")
    print(f"{'='*70}")

    # Basic shape comparison
    if sid_array_1.shape != sid_array_2.shape:
        print(f"  WARNING: Shape mismatch! {sid_array_1.shape} vs {sid_array_2.shape}")
        print(f"  Comparison may be invalid.")
    else:
        print(f"  Both have shape {sid_array_1.shape}")

    # SID overlap: how many SID tuples appear in both files?
    sids_1 = set(tuple(row) for row in sid_array_1)
    sids_2 = set(tuple(row) for row in sid_array_2)
    overlap = sids_1 & sids_2
    union = sids_1 | sids_2

    print(f"\n  --- SID Set Overlap ---")
    print(f"  Unique SIDs in {label1}: {len(sids_1)}")
    print(f"  Unique SIDs in {label2}: {len(sids_2)}")
    print(f"  Overlapping SIDs:      {len(overlap)}")
    print(f"  Jaccard similarity:    {len(overlap)/len(union)*100:.2f}%")

    # Item-level comparison: for each item, does it get the same SID?
    if sid_array_1.shape[0] == sid_array_2.shape[0]:
        same_sid_per_item = np.all(sid_array_1 == sid_array_2, axis=1)
        agreement_rate = same_sid_per_item.mean() * 100
        print(f"\n  --- Per-Item SID Agreement ---")
        print(f"  Items with same SID:   {same_sid_per_item.sum()}/{len(same_sid_per_item)}")
        print(f"  Agreement rate:        {agreement_rate:.2f}%")

        # Per-level agreement
        for level in range(sid_array_1.shape[1]):
            level_agree = (sid_array_1[:, level] == sid_array_2[:, level]).mean() * 100
            print(f"  Level {level} agreement:    {level_agree:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SID collision rate from a pickle file"
    )
    parser.add_argument(
        "--sid_file",
        type=str,
        required=True,
        help="Path to SID pickle file (e.g., custom_semantic_0.pkl)",
    )
    parser.add_argument(
        "--sid_file_2",
        type=str,
        default=None,
        help="Optional: second SID file for comparison",
    )
    args = parser.parse_args()

    # Load and analyze primary file
    if not os.path.exists(args.sid_file):
        print(f"ERROR: File not found: {args.sid_file}")
        sys.exit(1)

    print(f"Loading SID file: {args.sid_file}")
    sid_array = load_sid_file(args.sid_file)
    stats = analyze_collisions(sid_array, label=os.path.basename(args.sid_file))

    # Load and analyze second file if provided
    if args.sid_file_2:
        if not os.path.exists(args.sid_file_2):
            print(f"ERROR: File not found: {args.sid_file_2}")
            sys.exit(1)

        print(f"\nLoading second SID file: {args.sid_file_2}")
        sid_array_2 = load_sid_file(args.sid_file_2)
        stats_2 = analyze_collisions(
            sid_array_2, label=os.path.basename(args.sid_file_2)
        )

        # Compare
        compare_sid_files(
            sid_array,
            sid_array_2,
            os.path.basename(args.sid_file),
            os.path.basename(args.sid_file_2),
        )

    print(f"\n{'='*70}")
    print("Analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
