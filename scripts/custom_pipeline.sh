#!/bin/bash
# =============================================================================
# TIGER Custom Dataset — End-to-End Pipeline
#
# Two-step workflow:
#   Step 0:  build_embedding.py     →  item_embedding.pt  (one-time, external)
#   Step 1:  run_custom.py          →  RQ-VAE (auto if SID missing) + TIGER training + eval
#
# RQ-VAE training is now INSIDE run_custom.py — if --sid_file doesn't exist,
# it automatically trains RQ-VAE with wandb logging and saves SID + codebook
# weights. This matches how run.py works for Amazon/Steam datasets.
#
# Usage:
#   bash scripts/custom_pipeline.sh
#   bash scripts/custom_pipeline.sh --skip-embedding    # skip Step 0
# =============================================================================
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

# Paths — adjust these to your environment
CAPTION_DIR="/llm_reco_ssd/huangrui06/llmrec_data/reco_log_csv_1w_processed/qwen3_infer_output"
PID_MAPPING_FILE="/code/get_caption/10W_datacaption/results/1Wdata/7days/compare/two_qwen/rec_data/pid_mapping.json"
DATA_DIR="/code/get_caption/10W_datacaption/results/1Wdata/7days/compare/two_qwen/rec_data"
EMBEDDING_FILE="/code/get_caption/pid_embeddings/item_embedding.pt"
SID_DIR="./ID_generation/ID/"
SID_FILE="./ID_generation/ID/custom_semantic_0.pkl"
CODEBOOK_WEIGHTS_FILE="./ID_generation/ID/custom_codebook_weights_0.pt"

# RQ-VAE hyperparameters (used only when SID file doesn't exist and auto-training kicks in)
# IMPORTANT: Adjust hidden_dim and latent_dim based on your embedding dimension.
#   - embedding_dim=768 → hidden_dim=[768,512,256], latent_dim=128  (original)
#   - embedding_dim=256 → hidden_dim=[256,128,64],  latent_dim=64
#   - embedding_dim=64  → hidden_dim=[256,128,64],  latent_dim=32
# Rule: hidden_dim starts >= input_dim, latent_dim < final hidden_dim
#
# Epochs: Your dataset is ~60x larger than Amazon Beauty (815K vs 12K items).
# With 355 batches/epoch, 300 epochs ≈ 106K gradient updates ≈ 2x Beauty's 48K.
# 300 epochs is MORE than sufficient. Don't use 8000 (that's for 6-batch datasets).
CODEBOOK_SIZE=256
NUM_LAYERS=3
LATENT_DIM=128
HIDDEN_DIM="768 512 256"
DROPOUT=0.1
BETA=0.25
RQVAE_EPOCHS=300
RQVAE_BATCH_SIZE=4096
RQVAE_LR=0.001

# Training hyperparameters
SEED=42
DEVICE_ID=1
EXPERIMENT_ID="custom_exp"
TRAIN_FILE="train_all.txt"

# Wandb logging
WANDB_MODE="disabled"   # change to "online" or "offline" to enable wandb

# ── Parse flags ────────────────────────────────────────────────────────────────

SKIP_EMBEDDING=true

for arg in "$@"; do
    case $arg in
        --skip-embedding)  SKIP_EMBEDDING=true;  shift ;;
        --no-skip-embedding)  SKIP_EMBEDDING=false;  shift ;;
        --help)
            echo "Usage: bash scripts/custom_pipeline.sh [--skip-embedding] [--no-skip-embedding]"
            echo ""
            echo "  --skip-embedding      Skip Step 0 (build_embedding.py), assume .pt file exists (default)"
            echo "  --no-skip-embedding   Run Step 0 (build_embedding.py)"
            echo ""
            echo "RQ-VAE is auto-trained inside run_custom.py if --sid_file doesn't exist."
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo " TIGER Custom Dataset Pipeline"
echo "============================================================"

# =============================================================================
# Step 0: Build Item Embedding (.pt) from Caption Files (one-time)
# =============================================================================
if [ "$SKIP_EMBEDDING" = false ]; then
    echo ""
    echo "── Step 0: Building item embedding tensor from caption files ──"
    echo "  Caption dir:    $CAPTION_DIR"
    echo "  PID mapping:    $PID_MAPPING_FILE"
    echo "  Output:         $EMBEDDING_FILE"
    echo ""

    python build_embedding.py \
        --caption_dir "$CAPTION_DIR" \
        --pid_mapping_file "$PID_MAPPING_FILE" \
        --output_file "$EMBEDDING_FILE"

    echo "✓ Step 0 complete: $EMBEDDING_FILE"
else
    echo ""
    echo "── Step 0: SKIPPED (using existing $EMBEDDING_FILE) ──"
fi

# =============================================================================
# Step 1: Train TIGER (+ auto-train RQ-VAE if SID file missing)
# =============================================================================
echo ""
echo "── Step 1: TIGER training (+ RQ-VAE if needed) ──"
echo "  Data dir:       $DATA_DIR"
echo "  Embedding:      $EMBEDDING_FILE"
echo "  SID file:       $SID_FILE"
echo "  RQVAE epochs:   $RQVAE_EPOCHS"
echo "  Experiment:     $EXPERIMENT_ID"
echo "  Wandb:          $WANDB_MODE"
echo ""

# ── Option A: TIGER base model ──
echo ">>> Training TIGER base model ..."
python run_custom.py \
    --data_dir "$DATA_DIR" \
    --embedding_file "$EMBEDDING_FILE" \
    --sid_file "$SID_FILE" \
    --seed "$SEED" \
    --device_id "$DEVICE_ID" \
    --experiment_id "${EXPERIMENT_ID}_tiger" \
    --rqvae_epochs "$RQVAE_EPOCHS" \
    --rqvae_batch_size "$RQVAE_BATCH_SIZE" \
    --rqvae_lr "$RQVAE_LR" \
    --rqvae_beta "$BETA" \
    --rqvae_codebook_size "$CODEBOOK_SIZE" \
    --rqvae_num_layers "$NUM_LAYERS" \
    --rqvae_latent_dim "$LATENT_DIM" \
    --rqvae_hidden_dim $HIDDEN_DIM \
    --rqvae_dropout "$DROPOUT" \
    --wandb_mode "$WANDB_MODE"

echo "✓ TIGER base model training complete"

# ── Option B: TIGER_Residual model (uncomment to run) ──
# echo ">>> Training TIGER_Residual model ..."
# python run_custom.py \
#     --data_dir "$DATA_DIR" \
#     --embedding_file "$EMBEDDING_FILE" \
#     --sid_file "$SID_FILE" \
#     --use_residual_decoder True \
#     --codebook_loss_weight 1.0 \
#     --resid_nontf_ratio 0.2 \
#     --ntp_nontf_ratio 0.0 \
#     --seed "$SEED" \
#     --device_id "$DEVICE_ID" \
#     --experiment_id "${EXPERIMENT_ID}_residual" \
#     --rqvae_epochs "$RQVAE_EPOCHS" \
#     --wandb_mode "$WANDB_MODE"
#
# echo "✓ TIGER_Residual training complete"

# ── Option C: TIGER + Soft Label SID (uncomment to run) ──
# echo ">>> Training TIGER_SoftLabel model ..."
# python run_custom.py \
#     --data_dir "$DATA_DIR" \
#     --embedding_file "$EMBEDDING_FILE" \
#     --sid_file "$SID_FILE" \
#     --use_softlabel_sid True \
#     --soft_label_K 3 \
#     --soft_label_temperature 1.0 \
#     --seed "$SEED" \
#     --device_id "$DEVICE_ID" \
#     --experiment_id "${EXPERIMENT_ID}_softlabel" \
#     --rqvae_epochs "$RQVAE_EPOCHS" \
#     --wandb_mode "$WANDB_MODE"
#
# echo "✓ TIGER_SoftLabel training complete"

echo ""
echo "============================================================"
echo " Pipeline complete!"
echo "============================================================"
