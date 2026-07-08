"""
Test script: Visualize soft-label probability distribution from the TIGER soft-label mechanism.

This replicates the core logic from tiger_residual.py (lines 523-530):
    1. topk_dist_sq = dist_sq.gather(1, topk_indices)
    2. topk_dist = torch.sqrt(topk_dist_sq.clamp(min=1e-8))
    3. neg_scaled_dist = -topk_dist / temperature
    4. soft_probs_topk = F.softmax(neg_scaled_dist, dim=-1)

We use the real distance data from the training log to see what the final
probability distribution looks like under different temperatures.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ─── Real data from training log ───────────────────────────────────────
# Each row: topK L2 distances (K=22, first is always 0.00 = ground truth)
raw_data = [
    # sample 0, gt_idx=16, max_p=0.1365, min_p=0.0436
    [0.00, 5.10, 5.12, 5.17, 5.25, 5.26, 5.46, 5.48, 5.53, 5.55, 5.59, 5.64, 5.65, 5.65, 5.66, 5.69, 5.69, 5.70, 5.70, 5.71],
    # sample 1, gt_idx=247, max_p=0.1304, min_p=0.0431
    [0.00, 4.37, 4.95, 5.05, 5.08, 5.08, 5.12, 5.12, 5.15, 5.23, 5.28, 5.30, 5.33, 5.41, 5.50, 5.51, 5.53, 5.53, 5.54, 5.54],
    # sample 2, gt_idx=161, max_p=0.1207, min_p=0.0426
    [0.00, 3.57, 4.48, 4.53, 4.54, 4.57, 4.63, 4.69, 4.75, 4.80, 4.92, 4.95, 5.01, 5.04, 5.07, 5.09, 5.15, 5.18, 5.19, 5.21],
    # sample 3, gt_idx=80, max_p=0.1364, min_p=0.0439
    [0.00, 5.29, 5.29, 5.30, 5.37, 5.40, 5.42, 5.42, 5.43, 5.50, 5.51, 5.55, 5.55, 5.56, 5.60, 5.63, 5.63, 5.64, 5.66, 5.67],
    # sample 4, gt_idx=60, max_p=0.1334, min_p=0.0422
    [0.00, 4.74, 4.97, 5.02, 5.15, 5.19, 5.19, 5.27, 5.28, 5.32, 5.36, 5.41, 5.44, 5.56, 5.64, 5.65, 5.71, 5.72, 5.74, 5.75],
    # sample 5, gt_idx=53, max_p=0.1389, min_p=0.0426
    [0.00, 3.80, 5.10, 5.58, 5.59, 5.59, 5.61, 5.61, 5.69, 5.71, 5.71, 5.81, 5.85, 5.86, 5.86, 5.88, 5.90, 5.90, 5.90, 5.91],
    # sample 6, gt_idx=180, max_p=0.1204, min_p=0.0411
    [0.00, 3.88, 4.11, 4.29, 4.33, 4.41, 4.51, 4.53, 4.71, 4.76, 4.77, 4.79, 5.15, 5.16, 5.17, 5.25, 5.31, 5.34, 5.35, 5.37],
    # sample 7, gt_idx=32, max_p=0.1307, min_p=0.0428
    [0.00, 4.04, 4.85, 4.98, 5.18, 5.22, 5.22, 5.23, 5.26, 5.27, 5.35, 5.41, 5.42, 5.44, 5.45, 5.47, 5.48, 5.56, 5.56, 5.59],
    # sample 8, gt_idx=125, max_p=0.1304, min_p=0.0423
    [0.00, 4.00, 4.01, 4.90, 5.06, 5.15, 5.23, 5.31, 5.35, 5.36, 5.38, 5.45, 5.50, 5.56, 5.58, 5.58, 5.60, 5.62, 5.62, 5.62],
    # sample 9, gt_idx=163, max_p=0.1254, min_p=0.0417
    [0.00, 3.52, 3.74, 4.10, 4.62, 4.87, 4.89, 5.04, 5.24, 5.32, 5.38, 5.41, 5.45, 5.46, 5.46, 5.46, 5.47, 5.49, 5.50, 5.50],
]

gt_indices = [16, 247, 161, 80, 60, 53, 180, 32, 125, 163]

# Convert to torch tensor [B, K]
topk_dist = torch.tensor(raw_data, dtype=torch.float32)


def compute_soft_probs(topk_dist: torch.Tensor, temperature: float) -> torch.Tensor:
    """Replicate the exact logic from tiger_residual.py lines 529-530."""
    neg_scaled_dist = -topk_dist / temperature
    soft_probs = F.softmax(neg_scaled_dist, dim=-1)
    return soft_probs


def print_prob_table(probs: torch.Tensor, temp: float):
    """Print a nicely formatted table of probabilities."""
    B, K = probs.shape
    print(f"\n{'='*100}")
    print(f"  Temperature = {temp}")
    print(f"{'='*100}")
    print(
        f"  {'sample':>6}  {'gt_idx':>6}  {'GT_prob':>9}  {'rank1_prob':>10}  "
        f"{'min_prob':>9}  {'GT/rank1_ratio':>14}  {'entropy':>8}"
    )
    print(f"  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*14}  {'-'*8}")

    for i in range(B):
        gt_prob = probs[i, 0].item()         # ground truth always at index 0
        rank1_prob = probs[i, 1].item()      # nearest non-GT neighbor
        min_prob = probs[i, -1].item()        # farthest in top-K
        ratio = gt_prob / rank1_prob if rank1_prob > 0 else float('inf')

        # Compute entropy for this sample
        p = probs[i]
        entropy = -(p * torch.log(p + 1e-10)).sum().item()
        max_entropy = np.log(K)

        print(
            f"  {i:>6}  {gt_indices[i]:>6}  {gt_prob:>9.6f}  {rank1_prob:>10.6f}  "
            f"{min_prob:>9.6f}  {ratio:>14.2f}x  {entropy:>6.3f}/{max_entropy:.3f}"
        )


def print_full_distribution(probs: torch.Tensor, sample_idx: int, temp: float):
    """Print the full probability distribution for one sample."""
    K = probs.shape[1]
    print(f"\n  ┌─ Full probability distribution for sample {sample_idx} (gt_idx={gt_indices[sample_idx]}) "
          f"at temp={temp} ─┐")

    for j in range(K):
        dist_val = raw_data[sample_idx][j]
        prob_val = probs[sample_idx, j].item()
        bar_len = int(prob_val * 200)  # scale for visibility
        marker = " ◀ GT" if j == 0 else ""
        print(f"  │ dist={dist_val:5.2f}  prob={prob_val:.8f}  {'█' * bar_len}{marker}")

    print(f"  └{'─' * 70}┘")
    print(f"    Sum of probs = {probs[sample_idx].sum().item():.8f}")


def main():
    print("=" * 100)
    print("  TIGER Soft-Label Probability Distribution Test")
    print("  Core logic: softmax(-dist / temperature) over top-K codebook entries")
    print(f"  K = {topk_dist.shape[1]}, B = {topk_dist.shape[0]}")
    print("=" * 100)

    # Test with multiple temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for temp in temperatures:
        probs = compute_soft_probs(topk_dist, temp)
        print_prob_table(probs, temp)

    # ── Detailed view for a few representative samples ──────────────
    print("\n\n" + "=" * 100)
    print("  DETAILED VIEW: Full probability distributions")
    print("=" * 100)

    # Show samples with different distance patterns:
    #   - Sample 2: large gap between GT and nearest (3.57 vs 0.00)
    #   - Sample 3: small gap (5.29 vs 0.00) — nearly uniform distances
    #   - Sample 5: medium gap with large jump after rank-1 (0.00, 3.80, 5.10...)
    interesting_samples = [2, 3, 5]

    for temp in [0.1, 1.0, 5.0]:
        probs = compute_soft_probs(topk_dist, temp)
        for sidx in interesting_samples:
            print_full_distribution(probs, sidx, temp)

    # ── Summary statistics across temperatures ──────────────────────
    print("\n\n" + "=" * 100)
    print("  SUMMARY: How temperature affects probability concentration")
    print("=" * 100)
    print(
        f"\n  {'temp':>6}  {'avg_GT_prob':>11}  {'avg_rank1_prob':>14}  "
        f"{'avg_min_prob':>13}  {'avg_entropy':>11}  {'GT>80%?':>8}"
    )
    print(f"  {'-'*6}  {'-'*11}  {'-'*14}  {'-'*13}  {'-'*11}  {'-'*8}")

    for temp in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        probs = compute_soft_probs(topk_dist, temp)
        avg_gt = probs[:, 0].mean().item()
        avg_rank1 = probs[:, 1].mean().item()
        avg_min = probs[:, -1].mean().item()
        entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        avg_entropy = entropies.mean().item()
        max_entropy = np.log(topk_dist.shape[1])
        gt_dominant = "YES" if avg_gt > 0.8 else "no"

        print(
            f"  {temp:>6.2f}  {avg_gt:>11.6f}  {avg_rank1:>14.6f}  "
            f"{avg_min:>13.8f}  {avg_entropy:>7.3f}/{max_entropy:.3f}  {gt_dominant:>8}"
        )

    print("\n\n  Key insight:")
    print("  - Low temperature  → sharp distribution, GT gets most probability mass")
    print("  - High temperature → uniform distribution, all K entries share probability")
    print("  - The default temp=1.0 with these distances gives GT ~12-14% (quite flat!)")
    print("  - For GT to dominate, you likely need temp << 1.0 (e.g., 0.1 or lower)")
    print()


if __name__ == "__main__":
    main()
