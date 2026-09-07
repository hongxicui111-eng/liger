#!/bin/bash
# 针对已训练好的模型直接进行测试
# 用法：
#   bash scripts/test.sh <DATASET> <EXPERIMENT_ID> <CKPT_PATH>
#
# 示例：
#   bash scripts/test.sh Beauty tiger_Beauty_seed42 \
#       ./results/tiger/Amazon_Beauty/tiger_Beauty_seed42_seed_42/results/ckpt_best.pt

DATASET=${1:-"Beauty"}
EXPERIMENT_ID=${2:-"tiger_${DATASET}"}
CKPT_PATH=${3:-"./results/tiger/Amazon_${DATASET}/${EXPERIMENT_ID}_seed_42/results/ckpt_best.pt"}

echo "数据集     : ${DATASET}"
echo "Experiment : ${EXPERIMENT_ID}"
echo "Checkpoint : ${CKPT_PATH}"
echo ""

python test.py \
    dataset=amazon \
    dataset.name="${DATASET}" \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    experiment_id="${EXPERIMENT_ID}" \
    +test.ckpt_path="${CKPT_PATH}"
