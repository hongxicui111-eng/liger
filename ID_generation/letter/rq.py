import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple VQ layers.

    Matches the original LETTER implementation:
    - forward() returns 3 values: x_q, mean_losses, all_indices
    - Each VQ layer quantizes the residual from the previous layer

    Args:
        n_e_list: list of codebook sizes for each VQ layer
        e_dim: embedding dimension
        sk_epsilons: list of Sinkhorn epsilons (0 = disabled) for each layer
        beta: diversity loss weight
        kmeans_init: whether to use constrained kmeans initialization
        kmeans_iters: max kmeans iterations
        sk_iters: Sinkhorn iterations
    """

    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta=1.0,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def get_codebook(self):
        """Return all codebook embeddings as a stacked tensor [num_quantizers, n_e, e_dim]."""
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_ini(self, x):
        """Initialize VQ codebooks using constrained k-means (for kmeans_init mode)."""
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x, labels, use_sk=True):
        """Forward pass through all VQ layers.

        Args:
            x: input latent vectors [B, e_dim]
            labels: dict mapping layer_idx (str) -> cluster labels list
            use_sk: whether to use Sinkhorn algorithm

        Returns:
            x_q: aggregated quantized vectors [B, e_dim]
            mean_losses: mean loss across all layers
            all_indices: stacked indices [B, num_quantizers]
        """
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)]
            x_res, loss, indices = quantizer(residual, label, idx, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
