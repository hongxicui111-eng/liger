# Copyright (c) Meta Platforms, Inc. and affiliates.

# =============================================================================
# Residual Decoder Experiment
# 
# This experiment adds residual information from RQ-VAE codebooks
# into the decoder during semantic ID generation.
#
# Key config entries to add/override:
#   method.use_residual_decoder=True        # Enable residual decoder mode
#   method.codebook_loss_weight=1.0         # Weight for codebook selection CE loss term
#   method.num_residual_levels=2            # Number of residual positions (default: n_sem_codebook - 1)
#
# NEW: Mixed Teacher-Forcing / Non-Teacher-Forcing training
#   method.resid_nontf_ratio=0.2           # At each residual step, each sample has 20%
#                                            # probability of using predicted sid_k (instead of
#                                            # GT sid_k) for codebook index selection.
#                                            # This addresses the train-test mismatch where
#                                            # training uses GT sid_k but inference uses predicted.
#                                            # 0.0 = always GT (original), 0.1~0.3 recommended.
#   method.ntp_nontf_ratio=0.0             # At each NTP step k>0, each sample has this
#                                            # probability of using predicted sid_{k-1} as
#                                            # decoder input instead of GT. More aggressive.
#                                            # Recommended: start with 0 (or very small 0.05)
#                                            # and increase only if resid_nontf alone isn't enough.
# =============================================================================

dataset_name=Beauty

# ── Baseline Residual Decoder (original, full teacher forcing) ──
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.use_residual_decoder=True \
    method.codebook_loss_weight=1.0 \
    method.resid_beams=null \
    method.resid_score_weight=1.0 \
    method.resid_nontf_ratio=0.0 \
    method.ntp_nontf_ratio=0.0 \
    experiment_id="residual_$dataset_name"


# ── Residual + Mixed Non-TF for residual (recommended first experiment) ──
# Only the residual computation uses predicted sid_k with 20% probability.
# NTP decoder inputs still use full teacher forcing (safe, stable).
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.use_residual_decoder=True \
    method.codebook_loss_weight=1.0 \
    method.resid_beams=null \
    method.resid_score_weight=1.0 \
    method.resid_nontf_ratio=0.2 \
    method.ntp_nontf_ratio=0.0 \
    experiment_id="residual_nontf20_$dataset_name"


# ── Residual + Mixed Non-TF for both residual AND NTP (aggressive) ──
# Both residual and NTP paths use predicted tokens with some probability.
# More realistic (simulates inference conditions) but riskier for training stability.
# Recommended only if resid_nontf alone shows improvement but not enough.
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.use_residual_decoder=True \
    method.codebook_loss_weight=1.0 \
    method.resid_beams=null \
    method.resid_score_weight=1.0 \
    method.resid_nontf_ratio=0.2 \
    method.ntp_nontf_ratio=0.1 \
    experiment_id="residual_nontf20_ntp10_$dataset_name"


# ── Residual Decoder (with text embedding) ── 
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.use_residual_decoder=True \
    method.codebook_loss_weight=1.0 \
    method.flag_use_output_embedding=False \
    method.embedding_loss_weight=0 \
    method.resid_beams=null \
    method.resid_score_weight=1.0 \
    method.resid_nontf_ratio=0.2 \
    method.ntp_nontf_ratio=0.0 \
    experiment_id="residual_text_nontf20_$dataset_name"


# ── Sweep: different resid_nontf_ratio values ──
# Try 0.1, 0.2, 0.3, 0.5 to find the optimal ratio
for ratio in 0.1 0.2 0.3 0.5; do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        method.use_residual_decoder=True \
        method.codebook_loss_weight=1.0 \
        method.resid_beams=null \
        method.resid_score_weight=1.0 \
        method.resid_nontf_ratio=$ratio \
        method.ntp_nontf_ratio=0.0 \
        experiment_id="residual_nontf${ratio}_$dataset_name"
done