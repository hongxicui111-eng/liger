# =============================================================================
# test_only.sh — Evaluate a trained model checkpoint on the test set
#
# This script runs test_only.py, which loads a saved model checkpoint
# and evaluates it on the test set without any training.
#
# Usage:
#   bash scripts/test_only.sh
#
# Or manually (use checkpoint_path= for structured config, or +checkpoint_path= 
# if you haven't added checkpoint_path to main.yaml):
#   python test_only.py \
#       dataset=amazon \
#       dataset.name=Beauty \
#       seed=42 \
#       device_id=0 \
#       method=base \
#       test_method=tiger \
#       checkpoint_path="<path_to_ckpt_best.pt>" \
#       experiment_id="test_Beauty"
# =============================================================================

dataset_name=Sports_and_Outdoors

# ── Example 1: Standard TIGER model ──
python test_only.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    experiment_id="tiger_${dataset_name}_weightloss_base01_bs1024_lr1e-3_rankloss_explore"


# ── Example 2: TIGER with residual decoder ──
# Use checkpoint_path= to explicitly specify the checkpoint.
# The path must match where training saved ckpt_best.pt.
python test_only.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.use_residual_decoder=True \
    method.codebook_loss_weight=0.5 \
    method.resid_beams=null \
    method.resid_score_weight=1.0 \
    experiment_id="residual_${dataset_name}_codeLoss05_K10T1_MSEloss01"


# ── Example 3: Liger (TIGER + text embedding + dense retrieval) ──
# python test_only.py \
#     dataset=amazon \
#     dataset.name=$dataset_name \
#     seed=42 \
#     device_id=0 \
#     method=setting \
#     test_method=liger \
#     experiment_id="liger_${dataset_name}"


# ── Example 4: Residual Liger ──
# python test_only.py \
#     dataset=amazon \
#     dataset.name=$dataset_name \
#     seed=42 \
#     device_id=0 \
#     method=setting \
#     test_method=liger \
#     method.use_residual_decoder=True \
#     method.codebook_loss_weight=1.0 \
#     method.resid_beams=null \
#     method.resid_score_weight=1.0 \
#     experiment_id="residual_liger_${dataset_name}"


# ── Example 5: With explicit checkpoint path ──
# python test_only.py \
#     dataset=amazon \
#     dataset.name=$dataset_name \
#     seed=42 \
#     device_id=0 \
#     method=base \
#     test_method=tiger \
#     checkpoint_path="/home/sunyijia/cuihongxi/liger/results/tiger/Amazon_Sports_and_Outdoors/residual_Sports_and_Outdoors_codeLoss05_K10T1_MSEloss01_seed_42/results/ckpt_best.pt" \
#     experiment_id="test_custom_ckpt_${dataset_name}"


# ── Example 6: Steam dataset ──
# python test_only.py \
#     dataset=steam \
#     dataset.name=steam \
#     seed=42 \
#     device_id=0 \
#     method=base \
#     test_method=tiger \
#     experiment_id="tiger_steam"