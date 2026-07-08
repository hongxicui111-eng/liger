import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans


class MLPLayers(nn.Module):
    """MLP layers used in LETTER's encoder and decoder.

    Unlike the existing ID_generation/rqvae/layers.py (which uses residual connections),
    LETTER uses a simpler sequential MLP without residual connections.
    """

    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        activation_name = activation_name.lower()
        if activation_name == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name == "tanh":
            activation = nn.Tanh()
        elif activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            f"activation function {activation_name} is not implemented"
        )
    return activation


def kmeans(samples, num_clusters, num_iters=10):
    """K-means clustering on GPU tensors (scikit-learn based)."""
    B, dim, dtype, device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )
    x = samples.cpu().detach().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)
    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)
    return tensor_centers


def constrained_kmeans(data, n_clusters=10):
    """Constrained k-means clustering with size limits.

    Uses k_means_constrained package to enforce balanced clusters.
    Matches the original LETTER trainer's constrained_km parameters.
    Falls back to regular KMeans if the package is not available.
    """
    try:
        from k_means_constrained import KMeansConstrained

        x = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
        size_min = min(len(x) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=n_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
    except ImportError:
        import warnings
        warnings.warn(
            "k_means_constrained not installed, falling back to regular KMeans. "
            "Install with: pip install k-means-constrained"
        )
        x = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
        from sklearn.cluster import KMeans as SKLearnKMeans
        clf = SKLearnKMeans(n_clusters=n_clusters, max_iter=10, n_init=10)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

    return t_centers, t_labels


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """Sinkhorn-Knopp algorithm for balanced assignment.

    Produces a doubly-stochastic matrix Q such that:
    - Each row sums to 1 (each sample gets assigned uniformly)
    - Each column sums to B/K (each codebook entry gets balanced usage)

    Args:
        distances: [B, K] distance matrix (already centered)
        epsilon: regularization parameter (smaller = more balanced, but harder convergence)
        sinkhorn_iterations: number of Sinkhorn iterations

    Returns:
        Q: [B, K] assignment matrix
    """
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples
    K = Q.shape[1]  # codebook size

    # Normalize to make matrix sum to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # Normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # Normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # columns must sum to 1 so that Q is an assignment
    return Q
