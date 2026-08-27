"""
SID Collision Analysis

Load a saved SID (RQ-VAE codes) .pkl file and analyze the collision
situation at each codebook layer.

Usage:
    python -m ID_generation.sasrec.analyze_sid_collision \
        --sid_path ./ID_generation/ID/Beauty_sentence-t5-xxl_fused_42.pkl \
        --num_layers 3

Or directly:
    python -m ID_generation.sasrec.analyze_sid_collision \
        --sid_path <path> --num_layers 3

Output: per-layer collision stats + full-item collision report.
"""

import argparse
import os
import pickle
from collections import Counter

import numpy as np


def analyze_collision(ids, num_layers):
    """Analyze SID collision at each layer.

    Args:
        ids: [N, num_layers] int array — RQ-VAE codes for all items
        num_layers: number of codebook layers

    Prints:
        - Per-layer: unique codes, collision count, collision rate
        - Full SID (all layers): unique SIDs, collision count, collision rate
        - Distribution of collision group sizes
    """
    n_items = ids.shape[0]
    print(f"{'='*70}")
    print(f"SID Collision Analysis")
    print(f"  Total items: {n_items}")
    print(f"  Codebook layers: {num_layers}")
    print(f"  Codebook size per layer: {ids.max(axis=0).max() + 1} (approx)")
    print(f"{'='*70}\n")

    # --- Per-layer analysis ---
    print(f"{'Layer':<12} {'Unique':>10} {'Collisions':>12} {'Collision%':>12}")
    print(f"{'-'*48}")

    for layer in range(num_layers):
        codes = ids[:, layer]
        counter = Counter(codes.tolist())
        n_unique = len(counter)
        n_collisions = n_items - n_unique
        collision_rate = n_collisions / n_items * 100

        print(f"  L{layer+1:<10} {n_unique:>10} {n_collisions:>12} "
              f"{collision_rate:>11.2f}%")

    # --- Cumulative prefix analysis ---
    print(f"\n{'Prefix':<12} {'Unique':>10} {'Collisions':>12} {'Collision%':>12}")
    print(f"{'-'*48}")

    for n_prefix in range(1, num_layers + 1):
        prefix_codes = [tuple(ids[i, :n_prefix]) for i in range(n_items)]
        counter = Counter(prefix_codes)
        n_unique = len(counter)
        n_collisions = n_items - n_unique
        collision_rate = n_collisions / n_items * 100
        label = f"L1-L{n_prefix}"
        print(f"  {label:<12} {n_unique:>10} {n_collisions:>12} "
              f"{collision_rate:>11.2f}%")

    # --- Full SID collision ---
    full_sids = [tuple(ids[i]) for i in range(n_items)]
    full_counter = Counter(full_sids)
    n_unique_full = len(full_counter)
    n_collisions_full = n_items - n_unique_full
    collision_rate_full = n_collisions_full / n_items * 100

    print(f"\n{'='*70}")
    print(f"Full SID (L1-L{num_layers}):")
    print(f"  Unique SIDs:  {n_unique_full} / {n_items}")
    print(f"  Collisions:   {n_collisions_full} items ({collision_rate_full:.2f}%)")
    print(f"{'='*70}\n")

    # --- Collision group size distribution ---
    group_sizes = Counter(full_counter.values())
    print(f"Collision group size distribution (full SID):")
    print(f"  {'Group size':>12} {'Count':>10} {'Items affected':>16}")
    print(f"  {'-'*40}")
    for size in sorted(group_sizes.keys()):
        count = group_sizes[size]
        items_affected = size * count
        print(f"  {size:>12} {count:>10} {items_affected:>16}")

    # --- Top collision groups ---
    collision_groups = [(sid, cnt) for sid, cnt in full_counter.items() if cnt > 1]
    collision_groups.sort(key=lambda x: -x[1])

    if collision_groups:
        print(f"\nTop 20 collision groups (items sharing the same full SID):")
        print(f"  {'SID':<25} {'# Items':>10}")
        print(f"  {'-'*37}")
        for sid, cnt in collision_groups[:20]:
            print(f"  {str(sid):<25} {cnt:>10}")
    else:
        print(f"\n  No collisions at full SID level — every item has a unique SID.")

    # --- Per-layer prefix collision detail ---
    print(f"\n{'='*70}")
    print(f"Per-layer prefix collision detail:")
    print(f"{'='*70}")

    for n_prefix in range(1, num_layers + 1):
        prefix_codes = [tuple(ids[i, :n_prefix]) for i in range(n_items)]
        counter = Counter(prefix_codes)
        collision_groups = [(sid, cnt) for sid, cnt in counter.items() if cnt > 1]
        collision_groups.sort(key=lambda x: -x[1])

        print(f"\n  L1-L{n_prefix} prefix: {len(collision_groups)} collision groups")
        if collision_groups and len(collision_groups) <= 20:
            print(f"    {'Prefix SID':<25} {'# Items':>10}")
            print(f"    {'-'*37}")
            for sid, cnt in collision_groups[:20]:
                print(f"    {str(sid):<25} {cnt:>10}")
        elif collision_groups:
            print(f"    (showing top 20 of {len(collision_groups)} groups)")
            print(f"    {'Prefix SID':<25} {'# Items':>10}")
            print(f"    {'-'*37}")
            for sid, cnt in collision_groups[:20]:
                print(f"    {str(sid):<25} {cnt:>10}")

    return full_counter


def main():
    parser = argparse.ArgumentParser(description="SID Collision Analysis")
    parser.add_argument(
        "--sid_path", type=str, required=True,
        help="Path to the SID .pkl file (RQ-VAE codes)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3,
        help="Number of codebook layers (default: 3)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.sid_path):
        print(f"Error: SID file not found: {args.sid_path}")
        return

    with open(args.sid_path, "rb") as f:
        ids = pickle.load(f)

    if isinstance(ids, list):
        ids = np.array(ids)
    print(f"\nLoaded SID file: {args.sid_path}")
    print(f"  Shape: {ids.shape}")
    print(f"  dtype: {ids.dtype}")
    print(f"  Value range: [{ids.min()}, {ids.max()}]")

    # Run analysis and collect results
    full_counter = analyze_collision(ids, args.num_layers)

    # Save report to the same directory as the input SID file
    sid_dir = os.path.dirname(args.sid_path)
    sid_name = os.path.splitext(os.path.basename(args.sid_path))[0]
    report_path = os.path.join(sid_dir, f"{sid_name}_collision_report.txt")

    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        analyze_collision(ids, args.num_layers)
    report_text = buffer.getvalue()

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
