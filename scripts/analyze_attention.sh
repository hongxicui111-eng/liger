#!/bin/bash
# =============================================================================
# Attention Analysis for TIGER_Residual Model
#
# 诊断问题：生成后续 SID 时，模型是否关注了前面的残差位置？
#
# 用法：
#   1. 修改下面「路径配置」区的变量为你服务器上的实际路径
#   2. bash scripts/analyze_attention.sh
# =============================================================================

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        路径配置（按需修改）                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── ⭐ 最重要：wandb 训练配置 YAML ──
# 包含所有模型/数据/训练参数，不需要再猜任何值
config_yaml="./results/tiger/Amazon_Sports_and_Outdoors/residual_Sports_and_Outdoors_lookupTable_1_TrueNTPloss_seed_42/config.yaml"

# ── 训练输出路径 ──
ckpt_path="./results/tiger/Amazon_Sports_and_Outdoors/residual_Sports_and_Outdoors_lookupTable_1_TrueNTPloss_seed_42/results/ckpt_best.pt"
codebook_path="./ID_generation/ID/codebook_weights_42.pt"

# ── 数据文件路径（留空则从 config_yaml 或自动搜索） ──
data_file=""
id2meta_file=""
embedding_path=""
sid_path=""
raw_data_dir=""
processed_dir=""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        实验参数（按需修改）                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

dataset_name="Sports_and_Outdoors"
seed=42
device_id=0
num_samples=5
content_model="sentence-t5-xxl"
output_dir="./attention_analysis"
plot_flag=""               # 留空 = 不生成图表；设为 "--plot" 则生成


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         执行（不需要改）                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

echo "=============================================="
echo "  Attention Analysis: TIGER_Residual"
echo "  Dataset:     $dataset_name"
echo "  Config YAML: $config_yaml"
echo "  Checkpoint:  $ckpt_path"
echo "  Samples:     $num_samples"
echo "=============================================="

extra_args=""
[ -n "$data_file" ]       && extra_args="$extra_args --data_file $data_file"
[ -n "$id2meta_file" ]    && extra_args="$extra_args --id2meta_file $id2meta_file"
[ -n "$embedding_path" ]  && extra_args="$extra_args --embedding_path $embedding_path"
[ -n "$sid_path" ]        && extra_args="$extra_args --sid_path $sid_path"
[ -n "$codebook_path" ]   && extra_args="$extra_args --codebook_path $codebook_path"
[ -n "$raw_data_dir" ]    && extra_args="$extra_args --raw_data_dir $raw_data_dir"
[ -n "$processed_dir" ]   && extra_args="$extra_args --processed_dir $processed_dir"

python scripts/analyze_attention.py \
    --ckpt_path         "$ckpt_path"        \
    --config_yaml       "$config_yaml"      \
    --dataset           amazon              \
    --dataset_name      "$dataset_name"     \
    --content_model     "$content_model"    \
    --seed              "$seed"             \
    --device_id         "$device_id"        \
    --num_samples       "$num_samples"      \
    --output_dir        "$output_dir"       \
    $extra_args                             \
    $plot_flag

echo ""
echo "Done."