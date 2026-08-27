"""
SASRec: Self-Attentive Sequential Recommendation

Reference: ICLR 2018 - "Self-Attentive Sequential Recommendation" (Kang & McAuley)
Adapted from: https://github.com/kang205/SASRec

This model learns item embeddings from user-item interaction sequences using
self-attention. The learned item_embeddings capture collaborative filtering
signals and can be extracted for use in LETTER's CF alignment loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Args:
        num_items: number of unique items (items are 1-indexed, so embedding table has num_items+1)
        hidden_units: embedding and hidden dimension (also the CF embedding dim for LETTER)
        num_heads: number of attention heads
        num_blocks: number of self-attention blocks
        max_len: maximum sequence length
        dropout: dropout rate
        semantic_embeddings: optional frozen semantic embeddings [num_items+1, hidden_units].
            If provided, each item's representation = semantic_emb + learned CF emb
            (bold fusion version — no projection). Index 0 should be zeros (padding).
    """

    def __init__(
        self,
        num_items,
        hidden_units=32,
        num_heads=2,
        num_blocks=2,
        max_len=50,
        dropout=0.2,
        semantic_embeddings=None,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_len = max_len

        # CF embedding (learned) — 0 is padding, items are 1-indexed
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)

        # Frozen semantic embeddings (bold fusion: item_repr = semantic + CF, no projection)
        if semantic_embeddings is not None:
            assert semantic_embeddings.shape == (
                num_items + 1,
                hidden_units,
            ), f"semantic_embeddings must be [num_items+1, hidden_units], got {semantic_embeddings.shape}"
            self.register_buffer("semantic_embeddings", semantic_embeddings)
        else:
            self.semantic_embeddings = None
        # Positional embedding
        self.positional_embeddings = nn.Embedding(max_len, hidden_units)

        # Self-attention blocks
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(hidden_units, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.feedforward_layers = nn.ModuleList(
            [FeedForward(hidden_units, dropout) for _ in range(num_blocks)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_units) for _ in range(num_blocks * 2)]
        )

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _item_emb(self, ids):
        """Fused item embedding: semantic (frozen) + CF (learned)."""
        emb = self.item_embeddings(ids)
        if self.semantic_embeddings is not None:
            emb = emb + self.semantic_embeddings[ids]
        return emb

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def log2feats(self, seq_ids):
        """Forward pass: sequence of item IDs -> sequence of hidden states.

        Args:
            seq_ids: [B, seq_len] item ID sequence (1-indexed, 0 for padding)

        Returns:
            hidden: [B, seq_len, hidden_units]
        """
        batch_size, seq_len = seq_ids.shape
        assert seq_len <= self.max_len

        # Item embeddings + positional embeddings (fused: semantic + CF)
        x = self._item_emb(seq_ids)  # [B, seq_len, hidden_units]
        positions = torch.arange(seq_len, device=seq_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.positional_embeddings(positions)
        x = self.dropout(x)

        # Causal mask: each position can only attend to itself and earlier positions
        # Upper triangle = True (masked out), Lower triangle = False (visible)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq_ids.device), diagonal=1
        ).bool()

        # Padding mask: positions where seq_ids == 0 should be masked
        padding_mask = (seq_ids == 0)  # [B, seq_len]

        # Self-attention blocks with residual connections
        for i in range(self.num_blocks):
            # Self-attention + residual + layer norm
            normed = self.layer_norms[2 * i](x)
            attn_out = self.attention_layers[i](
                normed, causal_mask=causal_mask, padding_mask=padding_mask
            )
            x = x + attn_out

            # Feed-forward + residual + layer norm
            normed = self.layer_norms[2 * i + 1](x)
            ff_out = self.feedforward_layers[i](normed, padding_mask=padding_mask)
            x = x + ff_out

        return x

    def forward(self, seq_ids, pos_ids, neg_ids=None):
        """Forward pass for training with batch-internal softmax loss.

        Instead of BPR (1 random negative), each position uses all other
        batch items' positive embeddings as negatives. For each [b, t]:
        - positive = pos_ids[b, t]
        - negatives = pos_ids[b', t] for all b' != b in the batch
        - loss = -log(softmax(scores)[b, b, t])

        Args:
            seq_ids: [B, seq_len] input sequence (with last item removed)
            pos_ids: [B, seq_len] positive (next) items
            neg_ids: unused (kept for backward compat with the dataset)

        Returns:
            loss: cross-entropy loss scalar
        """
        hidden = self.log2feats(seq_ids)  # [B, seq_len, hidden_units]
        pos_emb = self._item_emb(pos_ids)    # [B, seq_len, hidden_units]

        # Cross-score: scores[b, c, t] = hidden[b, t] · pos_emb[c, t]
        # b = query user, c = candidate item (from batch), t = position
        scores = torch.einsum('btd,ctd->bct', hidden, pos_emb)  # [B, B, seq_len]

        # Mask padding negatives: if pos_ids[c, t] == 0, candidate c is padding
        neg_mask = (pos_ids == 0)  # [B, seq_len]
        scores = scores.masked_fill(neg_mask.unsqueeze(0), float('-inf'))

        # log_softmax over the candidate dimension (dim=1)
        log_probs = F.log_softmax(scores, dim=1)  # [B, B, seq_len]

        # Positive = diagonal: log_probs[b, b, t]
        pos_log_probs = torch.diagonal(log_probs, dim1=0, dim2=1).t()  # [B, seq_len]

        # Only count valid positions (pos_ids > 0)
        mask = (pos_ids > 0).float()  # [B, seq_len]
        loss = -pos_log_probs
        # Padding positions may produce NaN/inf (all candidates masked);
        # torch.where avoids propagating them (unlike multiplication).
        loss = torch.where(mask.bool(), loss, torch.zeros_like(loss))
        loss = torch.nan_to_num(loss, nan=0.0)  # safety: clean stray NaN
        loss = loss.sum() / mask.sum().clamp(min=1)

        return loss

    def predict(self, seq_ids, item_ids):
        """Score items given a sequence.

        Args:
            seq_ids: [B, seq_len] input sequence
            item_ids: [B, num_candidates] candidate item IDs

        Returns:
            scores: [B, num_candidates] dot-product scores
        """
        hidden = self.log2feats(seq_ids)  # [B, seq_len, hidden_units]

        # Use the last non-padding hidden state as user representation
        mask = (seq_ids > 0).float()  # [B, seq_len]
        seq_lengths = mask.sum(dim=-1).long()  # [B]
        last_indices = (seq_lengths - 1).clamp(min=0)  # [B]
        user_repr = hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]
        # user_repr: [B, hidden_units]

        # Get candidate item embeddings (fused: semantic + CF)
        item_emb = self._item_emb(item_ids)  # [B, num_candidates, hidden_units]

        # Compute scores
        scores = (user_repr.unsqueeze(1) * item_emb).sum(dim=-1)  # [B, num_candidates]
        return scores


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, hidden_units, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.head_dim = hidden_units // num_heads
        assert hidden_units % num_heads == 0

        self.W_Q = nn.Linear(hidden_units, hidden_units)
        self.W_K = nn.Linear(hidden_units, hidden_units)
        self.W_V = nn.Linear(hidden_units, hidden_units)
        self.W_O = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, padding_mask=None):
        """x: [B, seq_len, hidden_units]"""
        B, T, D = x.shape

        Q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, d]
        K = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, T, T]

        # Apply causal mask
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask: key positions that are padding should be masked
        if padding_mask is not None:
            # padding_mask: [B, T] -> [B, 1, 1, T]
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(scores, dim=-1)
        # Replace NaN from all-masked rows (padding queries with all keys masked)
        # with 0 so they contribute nothing to the output
        attn = attn.nan_to_num(nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, h, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        out = self.W_O(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, hidden_units, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        return self.ff(x)
