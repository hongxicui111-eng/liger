# Copyright (c) Meta Platforms, Inc. and affiliates.

# =============================================================================
# Residual Decoder Experiment
# 
# This experiment adds residual information from RQ-VAE codebooks
# into the decoder during semantic ID generation.
#
# Key config entries to add/override:
#   method.use_residual_decoder=True        # Enable residual decoder mode
#   method.mse_loss_weight=1.0              # Weight for MSE loss term
#   method.num_residual_levels=2            # Number of residual positions (default: n_sem_codebook - 1)
# =============================================================================

dataset_name=Beauty

# Residual Decoder (TIGER base)
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.use_residual_decoder=True \
    method.mse_loss_weight=1.0 \
    experiment_id="residual_$dataset_name"


# Residual Decoder (with text embedding) 
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.use_residual_decoder=True \
    method.mse_loss_weight=1.0 \
    method.flag_use_output_embedding=False \
    method.embedding_loss_weight=0 \
    experiment_id="residual_text_$dataset_name"


# Residual Decoder (liger-style: both SID + embedding outputs)
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.use_residual_decoder=True \
    method.mse_loss_weight=1.0 \
    experiment_id="residual_liger_$dataset_name"