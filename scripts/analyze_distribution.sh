#!/bin/bash
# 并发统计 Amazon 三个数据集的物品流行度 & 用户活跃度分布

DATASETS=("Beauty" "Toys_and_Games" "Sports_and_Outdoors")
PIDS=()

for DATASET in "${DATASETS[@]}"; do
    LOG_DIR="./results/analysis/Amazon_${DATASET}"
    mkdir -p "${LOG_DIR}"
    echo "▶ 启动 ${DATASET} ..."
    python analyze_distribution.py \
        dataset=amazon \
        dataset.name="${DATASET}" \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        experiment_id="analyze_dist" \
        > "${LOG_DIR}/run.log" 2>&1 &
    PIDS+=($!)
done

echo "三个任务已并发启动，PID: ${PIDS[*]}"
echo "等待全部完成..."

ALL_OK=true
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "✗ ${DATASETS[$i]} 失败（exit $CODE），查看日志: ./results/analysis/Amazon_${DATASETS[$i]}/run.log"
        ALL_OK=false
    else
        echo "✓ ${DATASETS[$i]} 完成"
    fi
done

if $ALL_OK; then
    echo ""
    echo "全部完成，统计结果保存在 ./results/analysis/"
fi
