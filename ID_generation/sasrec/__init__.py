"""
SASRec model and training utilities for generating CF (Collaborative Filtering) embeddings.

The CF embeddings are used by the LETTER tokenizer's alignment loss to ensure
that the quantized representations capture collaborative filtering signals
(i.e., items with similar user behavior patterns should have similar representations).

Reference: ICLR 2018 - "Self-Attentive Sequential Recommendation" (Kang & McAuley)
"""

from .model import SASRec
from .train_sasrec import train_sasrec, extract_cf_embeddings
