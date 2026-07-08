"""
Training script for LETTER RQ-VAE tokenizer.

This script trains the LETTER tokenizer which uses:
1. Sinkhorn-balanced assignment for balanced codebook usage
2. Diversity loss (within-cluster contrastive) for diverse codebook entries
3. CF alignment loss to align quantized representations with collaborative filtering embeddings

The trained tokenizer produces semantic IDs that are then consumed by the same
TIGER backbone model used by other tokenizers.

Performance optimizations over previous version:
- Matches original LETTER trainer pattern: cluster labels computed per epoch (not per batch)
- Uses DataLoader with num_workers and pin_memory for async data loading
- Uses .to(device, non_blocking=True) inside training loop for GPU async transfer
- Simplified return values (4 values from model/compute_loss, matching original)
"""

import os
import pickle
import collections
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from .letter.rqvae import RQVAELetter
from .letter.layers import constrained_kmeans


def check_collision(all_indices_str):
    """Check if there are any collisions (duplicate semantic IDs)."""
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    """Count occurrences of each semantic ID."""
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_items(all_indices_str):
    """Get groups of items that share the same semantic ID (collisions)."""
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups


def _compute_cluster_labels(model, n_clusters):
    """Compute cluster labels for each VQ layer using constrained k-means.

    This is called once per epoch (like the original LETTER trainer),
    using n_jobs=10 for parallel k-means.

    Args:
        model: RQVAELetter model
        n_clusters: number of clusters

    Returns:
        labels: dict mapping layer_idx (str) -> cluster labels list
    """
    labels = {}
    embs = [
        layer.embedding.weight.cpu().detach().numpy()
        for layer in model.rq.vq_layers
    ]
    for idx, emb in enumerate(embs):
        _, cluster_labels = constrained_kmeans(emb, n_clusters)
        labels[str(idx)] = cluster_labels
    return labels


def train_epoch(model, dataloader, optimizer, config, n_clusters, flag_eval=False):
    """Train or evaluate one epoch of the LETTER tokenizer.

    Matches the original LETTER trainer._train_epoch() pattern:
    - Cluster labels are computed at the start of each epoch (inside this function)
    - Model forward returns 4 values (out, rq_loss, indices, x_q)
    - compute_loss returns 4 values (total_loss, cf_loss, loss_recon, quant_loss)

    Args:
        model: RQVAELetter model
        dataloader: data loader
        optimizer: optimizer (None for eval)
        config: LETTER config dict
        n_clusters: number of clusters for constrained k-means
        flag_eval: whether this is evaluation mode

    Returns:
        total_loss, total_recon_loss, total_cf_loss, total_quant_loss (aggregated)
    """
    if flag_eval:
        model.eval()
    else:
        model.train()

    total_loss = 0
    total_recon_loss = 0
    total_cf_loss = 0
    total_quant_loss = 0

    # Compute cluster labels once per epoch (matching original LETTER trainer)
    # This uses n_jobs=10 for parallel k-means and is fast enough
    # to run every epoch as in the original implementation
    labels = _compute_cluster_labels(model, n_clusters)

    for batch_idx, batch in enumerate(dataloader):
        data, emb_idx = batch[0], batch[1]
        data = data.to(model.cf_embedding.device if model.cf_embedding is not None
                       else next(model.parameters()).device,
                       non_blocking=True)

        if not flag_eval:
            optimizer.zero_grad()

        out, rq_loss, indices, dense_out = model(data, labels)
        loss, cf_loss, loss_recon, quant_loss = model.compute_loss(
            out, rq_loss, emb_idx, dense_out, xs=data
        )

        if not flag_eval:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_recon_loss += loss_recon.item()
        total_cf_loss += cf_loss.item() if cf_loss != 0 else cf_loss
        total_quant_loss += quant_loss.item()

    return total_loss, total_recon_loss, total_cf_loss, total_quant_loss


def vq_init(model, item_embedding, device, batch_size=2048):
    """Initialize VQ codebooks using constrained k-means.

    Args:
        model: RQVAELetter model
        item_embedding: item embeddings [n_items, in_dim]
        device: torch device
        batch_size: batch size for initialization
    """
    model.eval()
    init_dataset = TensorDataset(item_embedding)
    init_loader = DataLoader(
        init_dataset,
        batch_size=len(item_embedding),
        shuffle=True,
        pin_memory=False,
    )
    print("Initializing VQ codebooks with constrained k-means...")
    for batch in init_loader:
        data = batch[0].to(device, non_blocking=True)
        model.vq_initialization(data)
    print("VQ initialization complete.")


def compute_collision_rate(model, item_embedding, device, n_clusters, batch_size=1024):
    """Compute collision rate of current semantic IDs.

    Args:
        model: RQVAELetter model
        item_embedding: item embeddings [n_items, in_dim]
        device: torch device
        n_clusters: number of clusters for constrained k-means
        batch_size: batch size

    Returns:
        collision_rate: fraction of items sharing a semantic ID
    """
    model.eval()
    dataset = TensorDataset(item_embedding, torch.arange(len(item_embedding)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    indices_set = set()
    num_sample = 0

    # Compute cluster labels for evaluation
    labels = _compute_cluster_labels(model, n_clusters)

    for batch in loader:
        data, emb_idx = batch[0].to(device, non_blocking=True), batch[1]
        num_sample += len(data)
        indices = model.get_indices(data, labels, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = "-".join([str(int(_)) for _ in index])
            indices_set.add(code)

    collision_rate = (num_sample - len(indices_set)) / num_sample
    return collision_rate


def train_letter_rqvae(model, x, device, writer, config):
    """Train the LETTER RQ-VAE model.

    Matches the original LETTER Trainer.fit() pattern:
    - DataLoader with num_workers and pin_memory
    - Keep data on CPU, transfer to GPU inside training loop with non_blocking
    - Cluster labels computed per epoch (inside train_epoch)
    - Evaluation on collision rate

    Args:
        model: RQVAELetter model
        x: training item embeddings [n_items, in_dim] (numpy or CPU tensor)
        device: torch device
        writer: logging writer
        config: LETTER-specific config dict
    """
    model.to(device)
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    lr = config["lr"]
    eval_step = config.get("eval_step", min(2000, num_epochs))
    n_clusters = config.get("n_clusters", 10)
    num_workers = config.get("num_workers", 4)

    # Build optimizer
    if config.get("optimizer", "AdamW") == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 1e-4)
        )
    elif config.get("optimizer", "Adam") == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 1e-4)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train/validation split — keep data on CPU for pin_memory transfer
    trainset, validationset = train_test_split(
        x, test_size=config.get("val_ratio", 0.05), random_state=42
    )
    train_indices = torch.arange(len(trainset))
    val_indices = torch.arange(len(validationset))

    # Keep data on CPU; transfer to GPU inside training loop with non_blocking
    trainset_tensor = torch.Tensor(trainset)
    valset_tensor = torch.Tensor(validationset)

    train_dataset = TensorDataset(trainset_tensor, train_indices)
    val_dataset = TensorDataset(valset_tensor, val_indices)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Step 1: Initialize VQ codebooks
    # For VQ init, we need to transfer data to GPU
    trainset_gpu = trainset_tensor.to(device)
    vq_init(model, trainset_gpu, device, batch_size)

    best_collision_rate = np.inf
    best_model_state = None
    patience = config.get("patience", 10)  # number of eval_steps to wait
    patience_counter = 0

    for epoch_idx in tqdm(range(num_epochs)):

        # Train one epoch (cluster labels computed inside train_epoch)
        training_start_time = time()
        train_loss, train_recon_loss, cf_loss, quant_loss = train_epoch(
            model, dataloader, optimizer, config, n_clusters, flag_eval=False
        )
        training_end_time = time()

        # Periodic evaluation
        if (epoch_idx + 1) % eval_step == 0:
            # Eval epoch
            train_epoch(
                model, val_dataloader, None, config, n_clusters, flag_eval=True
            )

            collision_rate = compute_collision_rate(
                model, trainset_gpu, device, n_clusters, batch_size=batch_size
            )

            if collision_rate < best_collision_rate:
                best_collision_rate = collision_rate
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            writer.log({"pretrain/collision_rate": collision_rate})

            eval_time = time() - training_end_time
            print(
                f"Epoch {epoch_idx + 1}: train_loss={train_loss:.4f}, "
                f"collision_rate={collision_rate:.4f}, "
                f"eval_time={eval_time:.1f}s"
            )

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch_idx + 1} "
                    f"(no improvement for {patience} eval steps)."
                )
                break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("LETTER RQ-VAE training complete.")
    print(f"Best collision rate: {best_collision_rate:.4f}")


def generate_indices(model, item_embedding, device, n_clusters=10, batch_size=1024, max_resolve_iters=20):
    """Generate semantic IDs for all items, resolving collisions via Sinkhorn.

    This follows LETTER's collision resolution strategy:
    1. First pass: assign IDs without Sinkhorn (greedy argmin)
    2. Check for collisions
    3. If collisions exist, re-assign colliding items using Sinkhorn
    4. Repeat until no collisions or max iterations reached

    Args:
        model: trained RQVAELetter model
        item_embedding: item embeddings [n_items, in_dim]
        device: torch device
        n_clusters: number of clusters for constrained k-means
        batch_size: batch size
        max_resolve_iters: max collision resolution iterations

    Returns:
        all_indices: numpy array of shape [n_items, num_quantizers]
    """
    model.eval()
    dataset = TensorDataset(item_embedding, torch.arange(len(item_embedding)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    all_indices = []
    all_indices_str = []

    # Compute cluster labels
    labels = _compute_cluster_labels(model, n_clusters)

    # First pass: generate all indices without Sinkhorn
    for batch in loader:
        data, emb_idx = batch[0].to(device, non_blocking=True), batch[1]
        indices = model.get_indices(data, labels, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = "-".join([str(int(_)) for _ in index])
            all_indices.append(index.tolist())
            all_indices_str.append(code)

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # Enable Sinkhorn on the last VQ layer for collision resolution
    # (only if it was set to 0 during training)
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    # Resolve collisions
    tt = 0
    while True:
        if tt >= max_resolve_iters or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_items(all_indices_str)
        print(
            f"Collision resolution iteration {tt + 1}: "
            f"{len(collision_item_groups)} collision groups"
        )

        for collision_items in collision_item_groups:
            data = item_embedding[collision_items]
            data = data.to(device, non_blocking=True)
            indices = model.get_indices(data, labels, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                all_indices[item] = index.tolist()
                all_indices_str[item] = "-".join([str(int(_)) for _ in index])
        tt += 1

    # Report collision statistics
    collision_count = get_indices_count(all_indices_str)
    max_count = max(collision_count.values()) if collision_count else 0
    tot_item = len(all_indices_str)
    tot_unique = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_unique) / tot_item
    print(f"All indices: {len(all_indices)}, Max collision count: {max_count}")
    print(f"Collision rate: {collision_rate:.4f}")

    return all_indices


def train(config, device, item_embedding, id_split, id_save_location, cf_embedding=None):
    """Main entry point for LETTER tokenizer training, matching the interface
    of ID_generation/train_rqvae.py:train().

    If the semantic ID file already exists, skip training.
    Otherwise, train the LETTER RQ-VAE, generate semantic IDs, and save them.

    Args:
        config: full config dict (must contain config["dataset"]["LETTER"])
        device: torch device
        item_embedding: item embeddings [n_items, embedding_dim]
        id_split: dict with 'seen', 'unseen_val', 'unseen_test' arrays
        id_save_location: path to save semantic IDs (.pkl)
        cf_embedding: collaborative filtering embeddings [n_items, cf_dim] (optional)
    """
    if os.path.exists(id_save_location):
        return

    print("LETTER Semantic ID file not found, Training LETTER RQ-VAE model...")
    from utils import setup_logging

    writer = setup_logging(config)
    model_config = config["dataset"]["LETTER"]

    input_size = model_config["input_dim"]
    hidden_sizes = model_config["hidden_dim"]
    e_dim = model_config["latent_dim"]
    num_emb_list = model_config["num_emb_list"]
    dropout = model_config["dropout"]
    sk_epsilons = model_config["sk_epsilons"]

    model = RQVAELetter(
        in_dim=input_size,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        layers=hidden_sizes,
        dropout_prob=dropout,
        bn=model_config.get("bn", False),
        loss_type=model_config.get("loss_type", "mse"),
        quant_loss_weight=model_config.get("quant_loss_weight", 1.0),
        kmeans_init=model_config.get("kmeans_init", True),
        kmeans_iters=model_config.get("kmeans_iters", 100),
        sk_epsilons=sk_epsilons,
        sk_iters=model_config.get("sk_iters", 50),
        alpha=model_config.get("alpha", 0.1),
        beta=model_config.get("beta", 0.1),
        n_clusters=model_config.get("n_clusters", 10),
        cf_embedding=cf_embedding,
    )

    train_letter_rqvae(
        model,
        item_embedding[id_split["seen"] - 1].cpu().numpy(),
        device,
        writer,
        model_config,
    )
    writer.finish()

    # Generate semantic IDs for all items
    model.to(device)
    model.eval()
    all_indices = generate_indices(
        model, item_embedding, device,
        n_clusters=model_config.get("n_clusters", 10),
        batch_size=model_config.get("batch_size", 1024),
    )

    with open(id_save_location, "wb") as f:
        pickle.dump(all_indices, f)

    # Also save the RQ-VAE codebook weights for potential residual decoder usage
    codebook_weights = [cb.weight.data.clone() for cb in model.rq.vq_layers]
    rqvae_save_dir = os.path.dirname(id_save_location)
    seed_suffix = id_save_location.rsplit("_", 1)[-1].replace(".pkl", "")
    codebook_path = os.path.join(
        rqvae_save_dir, f"letter_codebook_weights_{seed_suffix}.pt"
    )
    torch.save(codebook_weights, codebook_path)
    print(f"Saved LETTER codebook weights to {codebook_path}")
