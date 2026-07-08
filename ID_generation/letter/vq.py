import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import constrained_kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """Vector Quantizer with Sinkhorn-balanced assignment and diversity loss.

    Matches the original LETTER implementation:
    - forward() returns 3 values: x_q, loss, indices
    - diversity_loss uses random.choice (simple, original approach)
    - Combined loss = codebook_loss + mu * commitment_loss + beta * diversity_loss

    Args:
        n_e: number of embeddings (codebook size)
        e_dim: embedding dimension
        mu: commitment loss weight (default: 0.25)
        beta: diversity loss weight (default: 1.0)
        kmeans_init: whether to use constrained kmeans for codebook initialization
        kmeans_iters: max kmeans iterations
        sk_epsilon: Sinkhorn regularization (0 = disable Sinkhorn, >0 = enable)
        sk_iters: Sinkhorn iterations
    """

    def __init__(
        self,
        n_e,
        e_dim,
        mu=0.25,
        beta=1.0,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.01,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.mu = mu
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        """Initialize codebook embeddings using constrained k-means.
        
        Uses larger size_min (50) for initial codebook init, matching the original
        LETTER VQ init_emb which uses different parameters than diversity-loss clustering.
        """
        from k_means_constrained import KMeansConstrained
        x = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
        size_min = min(len(x) // (self.n_e * 2), 50)
        clf = KMeansConstrained(
            n_clusters=self.n_e,
            size_min=size_min,
            size_max=size_min * 4,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(x)
        centers = torch.from_numpy(clf.cluster_centers_)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        """Center distances to [-1, 1] range for Sinkhorn algorithm."""
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def diversity_loss(self, x_q, indices, indices_cluster, indices_list):
        """Within-cluster diversity loss (original LETTER implementation).

        For each item, randomly pick another codebook entry from the same cluster,
        and use contrastive loss to push different items within the same cluster
        to map to different codebook entries.

        Args:
            x_q: quantized representations [B, e_dim]
            indices: codebook indices [B]
            indices_cluster: cluster label for each index (list)
            indices_list: dict mapping cluster_label -> list of codebook indices in that cluster
        """
        emb = self.embedding.weight
        temp = 1

        pos_list = [indices_list[i] for i in indices_cluster]
        pos_sample = []
        for idx, pos in enumerate(pos_list):
            random_element = random.choice(pos)
            while random_element == indices[idx]:
                random_element = random.choice(pos)
            pos_sample.append(random_element)

        y_true = torch.tensor(pos_sample, device=x_q.device)

        sim = torch.matmul(x_q, emb.t())

        # Mask out self-similarity
        sim_self = torch.zeros_like(sim)
        for idx, row in enumerate(sim_self):
            sim_self[idx, indices[idx]] = 1e12
        sim = sim - sim_self
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)

        return loss

    def diversity_loss_main_entry(self, x, x_q, indices, labels):
        """Compute diversity loss using cluster labels.

        Args:
            x: input latent vectors
            x_q: quantized vectors
            indices: assigned codebook indices
            labels: list of cluster labels for each codebook entry
        """
        indices_cluster = [labels[idx.item()] for idx in indices]
        target_numbers = list(range(max(labels) + 1)) if labels else list(range(10))
        indices_list = {}
        for target_number in target_numbers:
            indices_list[target_number] = [
                index for index, num in enumerate(labels) if num == target_number
            ]

        diversity_loss = self.diversity_loss(x_q, indices, indices_cluster, indices_list)
        return diversity_loss

    def vq_init(self, x, use_sk=True):
        """Initialize VQ codebook (for kmeans_init mode)."""
        latent = x.view(-1, self.e_dim)

        if not self.initted:
            self.init_emb(latent)

        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        return x_q

    def forward(self, x, label, idx, use_sk=True):
        """Forward pass for vector quantization.

        Args:
            x: input latent vectors [B, e_dim]
            label: cluster labels for diversity loss computation
            idx: VQ layer index (-1 for inference/sampling mode)
            use_sk: whether to use Sinkhorn algorithm for balanced assignment

        Returns:
            x_q: quantized vectors (with straight-through gradient)
            loss: combined loss (codebook + commitment + diversity)
            indices: assigned codebook indices
        """
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Compute L2 distances to codebook entries
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )

        if not use_sk or self.sk_epsilon <= 0:
            if idx != -1:
                indices = torch.argmin(d, dim=-1)
            else:
                # Sampling mode: sample from softmax distribution
                temp = 1.0
                prob_dist = F.softmax(-d / temp, dim=1)
                indices = torch.multinomial(prob_dist, 1).squeeze()
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # Diversity loss (within-cluster contrastive)
        diversity_loss = self.diversity_loss_main_entry(x, x_q, indices, label)

        # Commitment loss + codebook loss + diversity loss (all combined)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

        # Straight-through estimator: gradient flows through x, not x_q
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices
