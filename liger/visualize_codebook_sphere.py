"""
Visualize first-level RQ-VAE codebook on a 3D sphere.

For each of the 256 first-level SID codes, extract the learned codebook vector
(from the saved RQ-VAE state_dict), PCA-reduce to 3D, project onto a unit
sphere, and color each point by the number of items assigned to that code.

Falls back to centroid-of-assigned-items approximation if no state_dict is
available.

Produces a figure with two side-by-side subplots:
  Left  — pure-semantic quantization
  Right — fused (semantic + CF) quantization

Usage:
  python visualize_codebook_sphere.py \
      --semantic_sid  path/to/semantic_sid.pkl \
      --semantic_emb  path/to/semantic_emb.pt \
      --fused_sid     path/to/fused_sid.pkl \
      --fused_emb     path/to/fused_emb.pt \
      --output        codebook_sphere.png

  # Optional: pass saved RQ-VAE state_dict for exact codebook vectors
  --semantic_rqvae path/to/semantic_rqvae.pt
  --fused_rqvae    path/to/fused_rqvae.pt
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_sid(sid_path):
    """Load RQ-VAE codes from a .pkl file → np.ndarray [n_items, n_codebook]."""
    with open(sid_path, "rb") as f:
        sids = pickle.load(f)
    if isinstance(sids, torch.Tensor):
        sids = sids.numpy()
    sids = np.asarray(sids)
    print(f"  SID shape: {sids.shape}, dtype: {sids.dtype}")
    return sids


def load_embeddings(emb_path):
    """Load item embeddings from a .pt or .pkl file → np.ndarray [n_items, dim]."""
    if emb_path.endswith(".pt"):
        emb = torch.load(emb_path, weights_only=False, map_location="cpu")
    else:
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)
    if isinstance(emb, torch.Tensor):
        emb = emb.numpy()
    emb = np.asarray(emb, dtype=np.float32)
    print(f"  Embedding shape: {emb.shape}")
    return emb


def load_codebook_vectors(rqvae_path, latent_dim=128, codebook_size=256):
    """
    Extract first-level codebook vectors from a saved RQ-VAE state_dict.

    The RQ-VAE quantizer stores codebooks as ModuleList of VQEmbedding
    (nn.Embedding).  Each VQEmbedding has weight shape [codebook_size+1, dim]
    where the last row is the padding entry — we exclude it.

    Returns:
      vectors: [codebook_size, latent_dim] np.ndarray
    """
    state = torch.load(rqvae_path, weights_only=False, map_location="cpu")
    # The first codebook is at key "quantizer.codebooks.0.weight"
    key = "quantizer.codebooks.0.weight"
    if key not in state:
        # Try alternative key patterns
        candidates = [k for k in state if "codebooks" in k and k.endswith(".0.weight")]
        if candidates:
            key = candidates[0]
        else:
            raise KeyError(
                f"Cannot find first codebook in state_dict. "
                f"Available keys: {list(state.keys())[:20]}..."
            )
    weight = state[key]  # [codebook_size+1, latent_dim]
    vectors = weight[:-1].numpy().astype(np.float32)  # drop padding row
    print(f"  Codebook vectors: {vectors.shape} (from key '{key}')")
    return vectors


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_counts(sids, codebook_size=256):
    """Count items assigned to each first-level SID code."""
    first_level = sids[:, 0].astype(int)
    counts = np.zeros(codebook_size, dtype=int)
    for c in range(codebook_size):
        counts[c] = int((first_level == c).sum())
    nonempty = counts > 0
    print(f"  Non-empty codes: {nonempty.sum()}/{codebook_size}")
    print(f"  Items per code: min={counts[nonempty].min()}, "
          f"max={counts.max()}, mean={counts[nonempty].mean():.1f}")
    return counts, nonempty


def compute_centroids(sids, embeddings, codebook_size=256, use_scaler=False):
    """
    Fallback: approximate codebook vectors via per-code item embedding centroids.
    When use_scaler=True, apply StandardScaler (fit on all items) before
    computing centroids, matching the RQ-VAE training preprocessing.
    """
    if use_scaler:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        print("  StandardScaler applied to embeddings before centroid computation")
    first_level = sids[:, 0].astype(int)
    dim = embeddings.shape[1]
    centroids = np.zeros((codebook_size, dim), dtype=np.float32)
    for c in range(codebook_size):
        mask = first_level == c
        if mask.sum() > 0:
            centroids[c] = embeddings[mask].mean(axis=0)
    return centroids


def compute_dispersion_metrics(vectors, counts, nonempty_mask):
    """
    Compute dispersion metrics in the *original* high-dimensional space
    (before PCA), so they are not distorted by 3D projection.

    Returns a dict with:
      mean_cos_dist: mean pairwise cosine distance (1 - cos_sim) between
                     non-empty code vectors — higher = more spread out.
      mean_euc_dist: mean pairwise Euclidean distance between non-empty codes.
      count_cv: coefficient of variation of item counts (std/mean) —
                higher = more imbalanced (some codes overloaded, many empty).
    """
    vecs = vectors[nonempty_mask]  # [n_nonempty, dim]
    n = vecs.shape[0]

    # Pairwise cosine distance
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vecs / norms
    cos_sim = normalized @ normalized.T  # [n, n]
    # Mean of upper triangle (excluding diagonal)
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_cos_dist = (1.0 - cos_sim[triu_mask]).mean()

    # Pairwise Euclidean distance
    sq = np.sum(vecs ** 2, axis=1)  # [n]
    sq_dists = sq[:, None] + sq[None, :] - 2 * (vecs @ vecs.T)
    sq_dists = np.maximum(sq_dists, 0)  # numerical safety
    euc_dists = np.sqrt(sq_dists)
    mean_euc_dist = euc_dists[triu_mask].mean()

    # Count coefficient of variation
    cnts = counts[nonempty_mask]
    count_cv = cnts.std() / cnts.mean() if cnts.mean() > 0 else 0.0

    # Gini coefficient (inequality of item distribution across codes)
    # 0 = perfectly equal, 1 = maximally concentrated (one code has all items)
    sorted_cnts = np.sort(cnts)
    n_codes = len(sorted_cnts)
    cumsum = np.cumsum(sorted_cnts)
    gini = (2 * np.sum((np.arange(1, n_codes + 1)) * sorted_cnts) /
            (n_codes * cumsum[-1]) - (n_codes + 1) / n_codes) if cumsum[-1] > 0 else 0.0

    # Top-10% concentration: fraction of items in the top 10% of codes
    top_10pct_n = max(1, n_codes // 10)
    top_10pct_share = sorted_cnts[-top_10pct_n:].sum() / cnts.sum() if cnts.sum() > 0 else 0.0

    # --- Tail metrics: exclude top-K hottest codes, measure balance of the rest ---
    # This reveals whether the "long tail" of codes is balanced on its own,
    # without extreme hot codes skewing the mean.
    top_k = max(1, n_codes // 10)  # remove top 10% hottest codes
    tail_cnts = sorted_cnts[:-top_k] if n_codes > top_k else sorted_cnts
    tail_cv = tail_cnts.std() / tail_cnts.mean() if tail_cnts.mean() > 0 else 0.0

    tail_sorted = np.sort(tail_cnts)
    tail_n = len(tail_sorted)
    tail_cumsum = np.cumsum(tail_sorted)
    tail_gini = (2 * np.sum((np.arange(1, tail_n + 1)) * tail_sorted) /
                 (tail_n * tail_cumsum[-1]) - (tail_n + 1) / tail_n
                 ) if tail_cumsum[-1] > 0 else 0.0

    return {
        "mean_cos_dist": float(mean_cos_dist),
        "mean_euc_dist": float(mean_euc_dist),
        "count_cv": float(count_cv),
        "gini": float(gini),
        "top10_share": float(top_10pct_share),
        "tail_cv": float(tail_cv),
        "tail_gini": float(tail_gini),
    }


def pca_to_sphere(vectors, nonempty_mask):
    """
    PCA-reduce vectors to 3D, then project onto a unit sphere.
    Only non-empty codes participate in PCA fitting.

    Returns:
      coords_3d: [codebook_size, 3]  (zero rows for empty codes)
      explained_var: [3,] explained variance ratio
    """
    pca = PCA(n_components=3)
    pca.fit(vectors[nonempty_mask])

    coords_3d = np.zeros((vectors.shape[0], 3), dtype=np.float32)
    coords_3d[nonempty_mask] = pca.transform(vectors[nonempty_mask])

    # Normalize each non-empty point to the unit sphere
    norms = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    coords_3d = coords_3d / norms

    return coords_3d, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_sphere_wireframe(ax, n_lat=12, n_lon=18, color="gray", alpha=0.3, linewidth=0.5):
    """Draw a unit sphere as latitude/longitude wireframe (no surface fill)."""
    # Latitude lines (horizontal circles)
    for lat in np.linspace(-80, 80, n_lat):
        phi = np.deg2rad(lat)
        r = np.cos(phi)
        z = np.sin(phi)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), z * np.ones_like(theta),
                color=color, alpha=alpha, linewidth=linewidth)
    # Longitude lines (vertical meridians)
    for lon in np.linspace(0, 360, n_lon, endpoint=False):
        theta = np.deg2rad(lon)
        phi = np.linspace(0, np.pi, 100)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)


def plot_one_sphere(ax, coords_3d, counts, nonempty_mask, title, metrics=None):
    """Draw a wireframe sphere and scatter points colored by item count."""
    draw_sphere_wireframe(ax)

    pts = coords_3d[nonempty_mask]  # [n_nonempty, 3]
    cnts = counts[nonempty_mask]

    # Log-scale color mapping: compresses extreme values so the colorbar
    # isn't dominated by a few hot codes. Map count → color via LogNorm.
    from matplotlib.colors import LogNorm
    cnts_for_color = np.maximum(cnts, 1)  # avoid log(0)

    scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=cnts_for_color,
        cmap="YlOrRd",          # yellow → orange → red heatmap
        norm=LogNorm(vmin=max(cnts_for_color.min(), 1),
                     vmax=cnts_for_color.max()),
        s=40,                    # fixed point size
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
        depthshade=True,
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    # Remove axis ticks and labels for a clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    # Remove the default grid/background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])

    # Dispersion metrics text box (top-left, in 2D axes coordinates)
    if metrics is not None:
        text = (
            f"Mean Cos Dist:   {metrics['mean_cos_dist']:.4f}\n"
            f"Mean Euc Dist:   {metrics['mean_euc_dist']:.2f}\n"
            f"─── All codes ───\n"
            f"Count CV:        {metrics['count_cv']:.2f}\n"
            f"Gini:            {metrics['gini']:.3f}\n"
            f"Top-10% share:   {metrics['top10_share']:.1%}\n"
            f"─── Tail (excl top 10%) ───\n"
            f"Tail CV:         {metrics['tail_cv']:.2f}\n"
            f"Tail Gini:       {metrics['tail_gini']:.3f}"
        )
        ax.text2D(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=8.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )

    return scatter


# ---------------------------------------------------------------------------
# Interactive HTML (Plotly)
# ---------------------------------------------------------------------------

def _sphere_wireframe_traces(radius=1.0, n_lat=12, n_lon=18, color="lightgray", opacity=0.3):
    """Generate Plotly traces for latitude/longitude lines on a unit sphere."""
    traces = []
    # Latitude lines
    for lat in np.linspace(-80, 80, n_lat):
        phi = np.deg2rad(lat)
        r = radius * np.cos(phi)
        z = radius * np.sin(phi)
        theta = np.linspace(0, 2 * np.pi, 100)
        traces.append(go.Scatter3d(
            x=r * np.cos(theta), y=r * np.sin(theta), z=z * np.ones_like(theta),
            mode="lines",
            line=dict(color=color, width=1),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        ))
    # Longitude lines
    for lon in np.linspace(0, 360, n_lon, endpoint=False):
        theta = np.deg2rad(lon)
        phi = np.linspace(0, np.pi, 100)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=color, width=1),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        ))
    return traces


def _make_scatter_trace(coords_3d, counts, nonempty_mask, name, color_max):
    """Build a Plotly 3D scatter trace with hover info (code id + count)."""
    pts = coords_3d[nonempty_mask]
    cnts = counts[nonempty_mask]
    code_ids = np.where(nonempty_mask)[0]
    cnts_safe = np.maximum(cnts, 1)  # for log scale
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=np.log1p(cnts),   # log-scale color for better dynamic range
            colorscale="YlOrRd",
            cmin=0,
            cmax=np.log1p(max(color_max, 1)),
            line=dict(color="black", width=0.5),
            opacity=0.9,
        ),
        text=[f"Code {cid}<br>Count: {cnt}" for cid, cnt in zip(code_ids, cnts)],
        hoverinfo="text",
        name=name,
        showlegend=False,
    )


def save_interactive_html(
    sem_coords, sem_counts, sem_nonempty, sem_var, sem_source,
    fus_coords, fus_counts, fus_nonempty, fus_var, fus_source,
    fis_coords, fis_var, fis_source,
    codebook_size, html_path,
):
    """Generate an interactive HTML file with three side-by-side 3D sphere plots.

    Third plot (fis = fused-in-semantic): fused SID partition centroids
    computed in the semantic embedding space, showing how CF relocates
    clusters relative to the pure-semantic manifold.
    """
    if not _PLOTLY_AVAILABLE:
        print("Plotly not installed — skipping interactive HTML. "
              "Install with: pip install plotly")
        return

    global_max = max(sem_counts.max(), fus_counts.max())

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Pure Semantic ({sem_source}) — {sem_nonempty.sum()}/{codebook_size} codes, "
            f"PCA var={sem_var.sum():.1%}",
            f"Fused (Semantic+CF) ({fus_source}) — {fus_nonempty.sum()}/{codebook_size} codes, "
            f"PCA var={fus_var.sum():.1%}",
            f"Fused SID in Semantic Space ({fis_source}) — {fus_nonempty.sum()}/{codebook_size} codes, "
            f"PCA var={fis_var.sum():.1%}",
        ],
        horizontal_spacing=0.02,
    )

    # Col 1: semantic
    for t in _sphere_wireframe_traces():
        fig.add_trace(t, row=1, col=1)
    fig.add_trace(
        _make_scatter_trace(sem_coords, sem_counts, sem_nonempty, "Semantic", global_max),
        row=1, col=1,
    )

    # Col 2: fused
    for t in _sphere_wireframe_traces():
        fig.add_trace(t, row=1, col=2)
    fig.add_trace(
        _make_scatter_trace(fus_coords, fus_counts, fus_nonempty, "Fused", global_max),
        row=1, col=2,
    )

    # Col 3: fused SID in semantic embedding space
    for t in _sphere_wireframe_traces():
        fig.add_trace(t, row=1, col=3)
    fig.add_trace(
        _make_scatter_trace(fis_coords, fus_counts, fus_nonempty, "FusedInSemantic", global_max),
        row=1, col=3,
    )

    scene_cfg = dict(
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
        zaxis=dict(visible=False, range=[-1.3, 1.3]),
        aspectmode="cube",
        bgcolor="white",
    )
    fig.update_layout(
        scene=scene_cfg,
        scene2=scene_cfg,
        scene3=scene_cfg,
        title=dict(
            text="First-Level RQ-VAE Codebook on Unit Sphere<br>"
                 "<sub>Hover over points to see code id and item count</sub>",
            font_size=14,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        showlegend=False,
    )

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive HTML to {html_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize first-level RQ-VAE codebook on 3D sphere"
    )
    parser.add_argument("--semantic_sid", type=str, required=True,
                        help="Path to pure-semantic SID .pkl file")
    parser.add_argument("--semantic_emb", type=str, required=True,
                        help="Path to pure-semantic item embedding .pt/.pkl file")
    parser.add_argument("--fused_sid", type=str, required=True,
                        help="Path to fused (semantic+CF) SID .pkl file")
    parser.add_argument("--fused_emb", type=str, required=True,
                        help="Path to fused item embedding .pt/.pkl file")
    parser.add_argument("--semantic_rqvae", type=str, default=None,
                        help="Path to saved RQ-VAE state_dict .pt (semantic)")
    parser.add_argument("--fused_rqvae", type=str, default=None,
                        help="Path to saved RQ-VAE state_dict .pt (fused)")
    parser.add_argument("--semantic_use_scaler", action="store_true", default=False,
                        help="Apply StandardScaler to semantic embeddings (centroid mode)")
    parser.add_argument("--fused_use_scaler", action="store_true", default=False,
                        help="Apply StandardScaler to fused embeddings (centroid mode)")
    parser.add_argument("--output", type=str, default="codebook_sphere.png",
                        help="Output figure path")
    parser.add_argument("--codebook_size", type=int, default=256,
                        help="Number of first-level codes (default: 256)")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="RQ-VAE latent dimension (default: 128)")
    args = parser.parse_args()

    # --- Semantic quantization ---
    print("[1/4] Loading pure-semantic SID and embeddings...")
    sem_sids = load_sid(args.semantic_sid)
    sem_emb = load_embeddings(args.semantic_emb)
    n = min(sem_sids.shape[0], sem_emb.shape[0])
    sem_sids, sem_emb = sem_sids[:n], sem_emb[:n]
    sem_counts, sem_nonempty = compute_counts(sem_sids, args.codebook_size)

    if args.semantic_rqvae and os.path.exists(args.semantic_rqvae):
        print(f"  Loading codebook from {args.semantic_rqvae}")
        sem_vectors = load_codebook_vectors(
            args.semantic_rqvae, args.latent_dim, args.codebook_size
        )
        sem_source = "codebook"
    else:
        print("  No RQ-VAE state_dict found, using centroid approximation")
        sem_vectors = compute_centroids(sem_sids, sem_emb, args.codebook_size,
                                         use_scaler=args.semantic_use_scaler)
        sem_source = "centroid"

    # --- Fused quantization ---
    print("[2/4] Loading fused SID and embeddings...")
    fus_sids = load_sid(args.fused_sid)
    fus_emb = load_embeddings(args.fused_emb)
    n = min(fus_sids.shape[0], fus_emb.shape[0])
    fus_sids, fus_emb = fus_sids[:n], fus_emb[:n]
    fus_counts, fus_nonempty = compute_counts(fus_sids, args.codebook_size)

    if args.fused_rqvae and os.path.exists(args.fused_rqvae):
        print(f"  Loading codebook from {args.fused_rqvae}")
        fus_vectors = load_codebook_vectors(
            args.fused_rqvae, args.latent_dim, args.codebook_size
        )
        fus_source = "codebook"
    else:
        print("  No RQ-VAE state_dict found, using centroid approximation")
        fus_vectors = compute_centroids(fus_sids, fus_emb, args.codebook_size,
                                         use_scaler=args.fused_use_scaler)
        fus_source = "centroid"

    # --- PCA to 3D + sphere projection ---
    print("[3/5] PCA projection to 3D sphere...")
    sem_coords, sem_var = pca_to_sphere(sem_vectors, sem_nonempty)
    fus_coords, fus_var = pca_to_sphere(fus_vectors, fus_nonempty)
    print(f"  Semantic explained variance: {sem_var} (total {sem_var.sum():.2%})")
    print(f"  Fused    explained variance: {fus_var} (total {fus_var.sum():.2%})")

    # Third view: fused SID partition projected onto semantic embedding space.
    # This shows where the fused-quantization clusters live in the *semantic*
    # embedding manifold — revealing how CF signal relocates items relative to
    # the pure-semantic structure.
    print("[4/5] Computing fused-SID centroids in semantic embedding space...")
    fus_in_sem_vectors = compute_centroids(
        fus_sids, sem_emb, args.codebook_size, use_scaler=args.semantic_use_scaler
    )
    fus_in_sem_source = "centroid (in semantic space)"
    fus_in_sem_coords, fus_in_sem_var = pca_to_sphere(fus_in_sem_vectors, fus_nonempty)
    print(f"  Fused-in-semantic explained variance: {fus_in_sem_var} "
          f"(total {fus_in_sem_var.sum():.2%})")

    # --- Dispersion metrics (computed in original high-dim space) ---
    print("[5/6] Computing dispersion metrics...")
    sem_metrics = compute_dispersion_metrics(sem_vectors, sem_counts, sem_nonempty)
    fus_metrics = compute_dispersion_metrics(fus_vectors, fus_counts, fus_nonempty)
    fis_metrics = compute_dispersion_metrics(fus_in_sem_vectors, fus_counts, fus_nonempty)
    print(f"  Semantic:        cos_dist={sem_metrics['mean_cos_dist']:.4f}, "
          f"euc={sem_metrics['mean_euc_dist']:.2f}, "
          f"cv={sem_metrics['count_cv']:.2f}, gini={sem_metrics['gini']:.3f}, "
          f"top10%={sem_metrics['top10_share']:.1%}, "
          f"tail_cv={sem_metrics['tail_cv']:.2f}, tail_gini={sem_metrics['tail_gini']:.3f}")
    print(f"  Fused:           cos_dist={fus_metrics['mean_cos_dist']:.4f}, "
          f"euc={fus_metrics['mean_euc_dist']:.2f}, "
          f"cv={fus_metrics['count_cv']:.2f}, gini={fus_metrics['gini']:.3f}, "
          f"top10%={fus_metrics['top10_share']:.1%}, "
          f"tail_cv={fus_metrics['tail_cv']:.2f}, tail_gini={fus_metrics['tail_gini']:.3f}")
    print(f"  Fused-in-Sem:    cos_dist={fis_metrics['mean_cos_dist']:.4f}, "
          f"euc={fis_metrics['mean_euc_dist']:.2f}, "
          f"cv={fis_metrics['count_cv']:.2f}, gini={fis_metrics['gini']:.3f}, "
          f"top10%={fis_metrics['top10_share']:.1%}, "
          f"tail_cv={fis_metrics['tail_cv']:.2f}, tail_gini={fis_metrics['tail_gini']:.3f}")

    # --- Plot ---
    print("[6/6] Plotting sphere figure...")
    fig = plt.figure(figsize=(24, 8))

    ax1 = fig.add_subplot(131, projection="3d")
    scatter1 = plot_one_sphere(
        ax1, sem_coords, sem_counts, sem_nonempty,
        f"Pure Semantic Quantization ({sem_source})\n"
        f"({sem_nonempty.sum()}/{args.codebook_size} codes, "
        f"PCA var={sem_var.sum():.1%})",
        metrics=sem_metrics,
    )

    ax2 = fig.add_subplot(132, projection="3d")
    scatter2 = plot_one_sphere(
        ax2, fus_coords, fus_counts, fus_nonempty,
        f"Fused (Semantic + CF) Quantization ({fus_source})\n"
        f"({fus_nonempty.sum()}/{args.codebook_size} codes, "
        f"PCA var={fus_var.sum():.1%})",
        metrics=fus_metrics,
    )

    ax3 = fig.add_subplot(133, projection="3d")
    scatter3 = plot_one_sphere(
        ax3, fus_in_sem_coords, fus_counts, fus_nonempty,
        f"Fused SID in Semantic Space ({fus_in_sem_source})\n"
        f"({fus_nonempty.sum()}/{args.codebook_size} codes, "
        f"PCA var={fus_in_sem_var.sum():.1%})",
        metrics=fis_metrics,
    )

    # Shared colorbar — placed at the bottom to avoid overlapping the right 3D plot
    cbar = fig.colorbar(scatter3, ax=[ax1, ax2, ax3], shrink=0.5, pad=0.02,
                        orientation="horizontal", fraction=0.04)
    cbar.ax.set_xlabel("Items per first-level SID", fontsize=11)

    fig.suptitle(
        "First-Level RQ-VAE Codebook on Unit Sphere\n"
        "(point position = PCA of codebook vectors, "
        "color = item count)",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {args.output}")

    # --- Interactive HTML (Plotly) ---
    html_path = os.path.splitext(args.output)[0] + ".html"
    save_interactive_html(
        sem_coords, sem_counts, sem_nonempty, sem_var, sem_source,
        fus_coords, fus_counts, fus_nonempty, fus_var, fus_source,
        fus_in_sem_coords, fus_in_sem_var, fus_in_sem_source,
        args.codebook_size, html_path,
    )


if __name__ == "__main__":
    main()
