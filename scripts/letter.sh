#!/bin/bash
# LETTER tokenizer + TIGER backbone
# Key difference from tiger.sh: uses LETTER tokenizer (Sinkhorn+Div+CF) instead of EMA RQ-VAE
# The backbone model is still TIGER; only the tokenizer changes.

# ──────────────────────────────────────────────────────────────
# CF Embedding Generation:
#   LETTER requires CF (collaborative filtering) embeddings for its alignment loss.
#   Two ways to provide them:
#     1. Auto: Just run with test_method=letter. If CF embeddings don't exist,
#        SASRec will be automatically trained on user-item sequences and CF embeddings
#        will be saved to ID_generation/sasrec/ckpt/<Dataset>-32d-sasrec.pt
#     2. Manual: Pre-train SASRec separately and specify the path:
#        method.letter_cf_embedding_path=./path/to/cf_embeddings.pt
# ──────────────────────────────────────────────────────────────

# Basic LETTER + TIGER (hard label, no residual)
# CF embeddings will be auto-generated if not found.
python run.py test_method=letter \
    method.use_letter_tokenizer=True

# LETTER + TIGER with explicit CF embedding path
# python run.py test_method=letter \
#     method.use_letter_tokenizer=True \
#     method.letter_cf_embedding_path=./ID_generation/sasrec/ckpt/Beauty-32d-sasrec.pt

# LETTER + TIGER with residual decoder (soft label + residual interleaving)
# python run.py test_method=letter \
#     method.use_letter_tokenizer=True \
#     method.use_residual_decoder=True \
#     method.codebook_loss_weight=1.0

# LETTER + TIGER with soft label SID (no residual)
# python run.py test_method=letter \
#     method.use_letter_tokenizer=True \
#     method.use_softlabel_sid=True
