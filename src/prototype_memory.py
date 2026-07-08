"""
MA-LIGER: Memory-Augmented LIGER
=================================

VQ-style Prototype Memory module for generative recommendation.

V2 Design (Unified Interest Representation + Confidence Gate):
  - Each Prototype stores L latent interest tokens (NOT SID embeddings)
  - Retrieval uses the mean of each prototype's tokens as cluster center
  - Cross-Attention + Confidence Gate for adaptive fusion
  - Gated Residual Fusion: prototype features participate in generation

Core change from V1:
  - Removed prototype_vectors + prototype_sids split
  - Single prototype_tokens [K, L, d_model] — unified interest representation
  - Added Confidence Gate (g = sigmoid(MLP(E_current)))
  - Residual fusion: encoder_hidden + g * attn_output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class PrototypeMemory(nn.Module):
    """VQ-style Prototype Memory with unified interest tokens.

    Each Prototype is represented by L latent interest tokens.
    No separate SID embeddings — semantic unity.

    Args:
        d_model: dimension of encoder hidden states (T5 d_model)
        K: number of prototypes
        p: number of nearest prototypes to retrieve (Top-p)
        L: number of interest tokens per prototype
        beta: weight for commitment loss
        delta: weight for diversity (entropy) regularization
        ema_decay: EMA decay for prototype token updates (0.99 = slow update)
    """

    def __init__(
        self,
        d_model: int,
        K: int = 256,
        p: int = 3,
        L: int = 8,
        beta: float = 0.25,
        delta: float = 0.1,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.p = p
        self.L = L
        self.beta = beta
        self.delta = delta
        self.ema_decay = ema_decay

        # Learnable prototype interest tokens — [K, L, d_model]
        # Each prototype is a SEQUENCE of L interest tokens
        self.prototype_tokens = nn.Parameter(
            torch.randn(K, L, d_model) * 0.02
        )

        self._log_step = 0

    @property
    def prototype_centers(self):
        """Mean of each prototype's tokens — used as retrieval centers [K, d]."""
        return self.prototype_tokens.mean(dim=1)  # [K, d_model]

    def forward(self, E_current: torch.Tensor):
        """VQ lookup: find nearest prototypes and retrieve interest tokens.

        Args:
            E_current: [B, d_model] user interest embedding (predicted_embedding)

        Returns:
            interest_context: [B, p*L, d_model] concatenated interest tokens from Top-p prototypes
            proto_losses: dict with 'commitment_loss', 'diversity_loss', 'prototype_usage', 'gate_value_placeholder'
        """
        B = E_current.shape[0]
        device = E_current.device

        # --- Step 1: Compute distances using prototype centers ---
        centers = self.prototype_centers  # [K, d_model]
        distances = torch.cdist(
            E_current.unsqueeze(1),           # [B, 1, d]
            centers.unsqueeze(0).expand(B, -1, -1),  # [B, K, d]
        ).squeeze(1)  # [B, K]

        # Top-p nearest prototypes (smallest distance)
        idx = distances.topk(self.p, dim=-1, largest=False).indices  # [B, p]

        # --- Step 2: Retrieve interest tokens from selected prototypes ---
        # prototype_tokens[idx] → [B, p, L, d_model]
        selected_tokens = self.prototype_tokens[idx]
        # Flatten p * L into a single sequence: [B, p*L, d_model]
        interest_context = selected_tokens.reshape(B, self.p * self.L, self.d_model)

        # --- Step 3: Commitment Loss (STE) ---
        # z_q: centers of the assigned prototypes → [B, p, d_model]
        z_q = centers[idx]  # [B, p, d_model]

        # STE: forward uses z_q, gradient flows through E_current
        z_q_sg = E_current.unsqueeze(1) + (z_q - E_current.unsqueeze(1)).detach()

        # Commitment loss: encourage E_current to stay close to assigned prototype centers
        commitment_loss = F.mse_loss(
            z_q.detach(),
            E_current.unsqueeze(1).expand_as(z_q),
        )

        # --- Step 4: Diversity Loss (entropy regularization) ---
        usage = torch.zeros(self.K, device=device)
        usage.scatter_add_(0, idx[:, 0], torch.ones(B, device=device))
        usage_prob = usage / (usage.sum() + 1e-8)
        entropy = -torch.sum(usage_prob * torch.log(usage_prob + 1e-8))
        diversity_loss = -entropy

        # --- Step 5: EMA update prototype tokens (detach, no gradient) ---
        if self.training:
            with torch.no_grad():
                centers_data = self.prototype_centers.data  # [K, d]
                for k_idx in range(self.K):
                    mask = (idx[:, 0] == k_idx)
                    if mask.sum() > 0:
                        new_center = E_current[mask].mean(0)
                        # Update all L tokens of this prototype toward new center
                        old_center = centers_data[k_idx]
                        shift = new_center - old_center
                        # Move all tokens by the same shift (preserves internal structure)
                        self.prototype_tokens.data[k_idx] = (
                            self.prototype_tokens.data[k_idx]
                            + (1.0 - self.ema_decay) * shift.unsqueeze(0)
                        )

        # --- Logging ---
        self._log_step += 1
        active_count = (usage > 0).sum().item()
        proto_losses = {
            "commitment_loss": commitment_loss,
            "diversity_loss": diversity_loss,
            "prototype_active_count": active_count,
            "prototype_max_usage": usage.max().item(),
        }

        return interest_context, proto_losses


class PrototypeCrossAttention(nn.Module):
    """Cross-Attention + Confidence Gate for adaptive prototype fusion.

    V2 design:
    1. Cross-Attention: Q=encoder_hidden, K/V=interest_context
    2. Confidence Gate: g = sigmoid(MLP(E_current))
    3. Gated Residual Fusion: encoder_hidden + g * attn_output

    The gate controls how much the prototype influences each user:
    - Cold start users: g large (rely more on group knowledge)
    - Rich history users: g small (rely more on individual history)

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        gate_hidden: hidden dimension for the confidence gate MLP
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        gate_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # Confidence Gate — multi-signal input:
        #   E_current [d_model]        — user's compressed interest (what they like)
        #   encoder_hidden mean [d_model] — global sequence quality
        #   attn_out mean [d_model]    — prototype relevance signal
        #   seq_len [1]                 — explicit history length signal
        # Total input dim = 2 * d_model + d_model + 1 = 3 * d_model + 1
        gate_input_dim = 3 * d_model + 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        interest_context: torch.Tensor,
        E_current: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """Fuse prototype interest context into encoder hidden states.

        Args:
            encoder_hidden: [B, seq_len, d_model] encoder last hidden states
            interest_context: [B, p*L, d_model] prototype interest tokens
            E_current: [B, d_model] user interest embedding
            attention_mask: [B, seq_len] 1 for valid positions, 0 for padding

        Returns:
            encoder_hidden_aug: [B, seq_len, d_model] augmented encoder hidden
            gate_values: [B] confidence gate values (for logging/analysis)
        """
        # Cross-Attention: encoder_hidden attends to prototype interest tokens
        attn_out, _ = self.attn(
            query=encoder_hidden,
            key=interest_context,
            value=interest_context,
            need_weights=False,
        )

        # ---- Confidence Gate with multi-signal input ----
        # Signal 1: E_current — user's compressed interest
        # Signal 2: mean(encoder_hidden) — global sequence representation quality
        # Signal 3: mean(attn_out) — how relevant the prototypes are
        # Signal 4: seq_len — explicit history length (cold start signal)

        if attention_mask is not None:
            # Masked mean pooling over valid positions
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            enc_mean = (encoder_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)  # [B, d]
            attn_mean = (attn_out * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)  # [B, d]
            seq_len = attention_mask.sum(1, keepdim=True).float()  # [B, 1]
        else:
            enc_mean = encoder_hidden.mean(1)  # [B, d]
            attn_mean = attn_out.mean(1)  # [B, d]
            seq_len = torch.full((encoder_hidden.shape[0], 1), encoder_hidden.shape[1],
                                 device=encoder_hidden.device, dtype=encoder_hidden.dtype)  # [B, 1]

        gate_input = torch.cat([E_current, enc_mean, attn_mean, seq_len], dim=-1)
        g = torch.sigmoid(self.gate_mlp(gate_input))  # [B, 1]
        g = g.unsqueeze(1)  # [B, 1, 1] — broadcast over seq_len and d_model

        # Gated Residual Fusion
        augmented = encoder_hidden + g * attn_out
        return self.norm(augmented), g.squeeze(1)  # [B], gate values for logging


def build_prototype_init(
    model,
    train_dataloader,
    device: torch.device,
    K: int,
    L: int,
    max_users: int = 50000,
) -> dict:
    """Initialize prototype_tokens from K-Means clustering.

    Stage 2 initialization:
    1. Run trained encoder on training data to collect predicted_embeddings
    2. K-Means cluster them into K groups
    3. Each cluster center is replicated L times to form prototype_tokens

    Args:
        model: trained TIGER_Residual model (from Stage 1)
        train_dataloader: training DataLoader
        device: torch device
        K: number of prototypes
        L: number of interest tokens per prototype
        max_users: maximum number of user embeddings to collect

    Returns:
        init_dict with key:
            'prototype_tokens': [K, L, d_model] numpy array
    """
    model.eval()

    # --- Collect predicted_embeddings from training set ---
    # Use model_forward_residual to handle the correct batch keys
    # (input_sids / attention_mask_sids / labels_sids, NOT input_ids / attention_mask)
    from .evaluation import model_forward_residual

    # We need method_config to know use_id and flag_add_input_embedding
    # Derive n_codebook from model config
    n_codebook = model.config.num_layers  # T5 num_layers = RQ-VAE levels

    all_embeddings = []

    with torch.no_grad():
        for batch in train_dataloader:
            # Use model_forward_residual to get encoder outputs correctly
            # We pass skip_forward=True to avoid running the full forward_residual
            # But we still need the encoder outputs, so we use a different approach:
            # directly call the encoder with the prepared inputs

            # SID mode: get input_sids and attention_mask_sids from batch
            input_sids = batch["input_sids"].to(device)
            attention_mask_sids = batch["attention_mask_sids"].to(device)

            # Prepare encoder inputs (same as model_forward_residual does)
            inputs_embeds = model.shared(input_sids)
            encoder_outputs = model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_sids,
                return_dict=True,
            )
            item_seq_len = attention_mask_sids.sum(-1)
            predicted_embedding = model.gather_indexes(
                encoder_outputs.last_hidden_state, item_seq_len - 1
            )
            all_embeddings.append(predicted_embedding.cpu().numpy())

            if len(all_embeddings) * input_sids.shape[0] >= max_users:
                break

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # [N, d_model]
    n_collected = all_embeddings.shape[0]
    print(f"[Prototype Init] Collected {n_collected} user embeddings, running K-Means with K={K}")

    # --- K-Means clustering ---
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=1024,
        max_iter=100,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(all_embeddings)  # [N]
    centers = kmeans.cluster_centers_  # [K, d_model]

    # --- Replicate each center L times to form prototype_tokens ---
    d_model = all_embeddings.shape[1]
    prototype_tokens = np.zeros((K, L, d_model), dtype=np.float32)
    for k_idx in range(K):
        for l_idx in range(L):
            # Add small noise to each replica so they differentiate during training
            prototype_tokens[k_idx, l_idx] = centers[k_idx] + np.random.randn(d_model) * 0.01

    print(f"[Prototype Init] Done. prototype_tokens: {prototype_tokens.shape}")

    return {
        "prototype_tokens": prototype_tokens,
    }
