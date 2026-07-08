import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import MLPLayers, constrained_kmeans
from .rq import ResidualVectorQuantizer


class RQVAELetter(nn.Module):
    """LETTER RQ-VAE tokenizer with Sinkhorn-balanced assignment, diversity loss,
    and collaborative filtering (CF) alignment loss.

    Matches the original LETTER RQ-VAE implementation:
    - forward() returns 4 values: out, rq_loss, indices, x_q
    - compute_loss() returns 4 values: total_loss, cf_loss, loss_recon, quant_loss
    - Cluster labels are computed per epoch in the trainer, not inside the model

    Args:
        in_dim: input embedding dimension (e.g., 768 for sentence-T5)
        num_emb_list: list of codebook sizes per VQ layer, e.g. [256, 256, 256, 256]
        e_dim: latent/codebook embedding dimension (e.g., 32)
        layers: MLP hidden layer sizes, e.g. [2048, 1024, 512, 256, 128, 64]
        dropout_prob: dropout probability
        bn: whether to use batch normalization
        loss_type: reconstruction loss type ("mse" or "l1")
        quant_loss_weight: weight for quantization loss
        kmeans_init: whether to use constrained k-means for codebook initialization
        kmeans_iters: max k-means iterations
        sk_epsilons: Sinkhorn epsilons per VQ layer (0 = disabled)
        sk_iters: Sinkhorn iterations
        alpha: CF loss weight
        beta: diversity loss weight
        n_clusters: number of clusters for constrained k-means (diversity loss)
        sample_strategy: sampling strategy
        cf_embedding: collaborative filtering embeddings [n_items, cf_dim] for CF loss
    """

    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        alpha=1.0,
        beta=0.001,
        n_clusters=10,
        sample_strategy="all",
        cf_embedding=None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy

        # Register CF embeddings as buffer (not a learnable parameter)
        if cf_embedding is not None:
            if isinstance(cf_embedding, np.ndarray):
                cf_embedding = torch.from_numpy(cf_embedding).float()
            self.register_buffer("cf_embedding", cf_embedding)
        else:
            self.cf_embedding = None

        # Encoder
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

        # Residual Vector Quantizer
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        # Decoder
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

    def forward(self, x, labels, use_sk=True):
        """Forward pass.

        Args:
            x: input embeddings [B, in_dim]
            labels: dict mapping layer_idx (str) -> cluster labels list
            use_sk: whether to use Sinkhorn algorithm

        Returns:
            out: reconstructed embeddings [B, in_dim]
            rq_loss: quantization loss
            indices: codebook indices [B, num_quantizers]
            x_q: quantized latent vectors [B, e_dim]
        """
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, labels, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices, x_q

    def CF_loss(self, quantized_rep, encoded_rep):
        """Collaborative filtering alignment loss.

        Aligns the quantized representation with CF embeddings via
        a contrastive (InfoNCE) loss: for each item, the quantized rep
        should be most similar to its own CF embedding.

        Args:
            quantized_rep: quantized latent vectors [B, e_dim]
            encoded_rep: CF embeddings for this batch [B, cf_dim]
        """
        batch_size = quantized_rep.size(0)
        labels = torch.arange(
            batch_size, dtype=torch.long, device=quantized_rep.device
        )
        similarities = torch.matmul(
            quantized_rep, encoded_rep.transpose(0, 1)
        )
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss

    def vq_initialization(self, x, use_sk=True):
        """Initialize VQ codebooks using constrained k-means."""
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        """Get codebook indices for input embeddings.

        Args:
            xs: input embeddings [B, in_dim]
            labels: cluster labels dict
            use_sk: whether to use Sinkhorn

        Returns:
            indices: codebook indices [B, num_quantizers]
        """
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, emb_idx, dense_out, xs=None):
        """Compute total loss: reconstruction + quantization + CF alignment.

        Matches the original LETTER compute_loss interface.

        Args:
            out: reconstructed embeddings
            quant_loss: quantization loss (includes diversity loss inside VQ)
            emb_idx: item indices for fetching CF embeddings
            dense_out: quantized latent vectors (for CF loss)
            xs: original input embeddings

        Returns:
            total_loss: combined loss
            cf_loss: CF alignment loss
            loss_recon: reconstruction loss
            quant_loss: quantization loss
        """
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # CF Loss
        cf_loss = 0
        if self.cf_embedding is not None:
            cf_embedding_in_batch = self.cf_embedding[emb_idx]
            cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)

        total_loss = rqvae_n_diversity_loss + self.alpha * cf_loss

        return total_loss, cf_loss, loss_recon, quant_loss
