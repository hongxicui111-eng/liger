import pickle
import numpy as np
import os
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ===================== 【核心配置项 · 仅需修改这里】 =====================
# 1. RQ-VAE 3级码本pkl文件
PKL_FILE = "/home/sunyijia/cuihongxi/liger/ID_generation/ID/Sports_and_Outdoors_sentence-t5-xxl_42.pkl"
# 2. 用户交互训练文件路径
TRAIN_SEQ_FILE = "/home/sunyijia/cuihongxi/liger/ID_generation/preprocessing/processed/Sports_and_Outdoors.txt"
# 3. 输出目录
OUTPUT_DIR = "/home/sunyijia/cuihongxi/liger/ID_generation/preprocessing/processed/Sports/spearman_consistency_result"
# 4. 过滤阈值：过滤掉频次低于该值的ID/SID，消除低频噪声干扰
MIN_COUNT = 5
# ==========================================================================

# 自动创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# 1. 加载3级码本，构建物品ID→三级SID映射
# ----------------------
print("🔄 加载3级码本映射...")
with open(PKL_FILE, "rb") as f:
    item2sid = pickle.load(f)  # shape: [总物品数, 3]
num_items = item2sid.shape[0]
item_sid_mapping = {
    item_id: (int(s1), int(s2), int(s3))
    for item_id, (s1, s2, s3) in enumerate(item2sid)
}
print(f"✅ 加载完成，共 {num_items} 个物品，3级码本")

# ----------------------
# 2. 解析用户序列，拆分训练/验证集，收集所有数据
# ----------------------
print("🔄 解析用户交互序列...")
# 原始物品ID收集
train_items = []      # 训练集：序列[:-2]
val1_items = []       # 验证集1：序列[-2]
val2_items = []       # 验证集2：序列[-1]

with open(TRAIN_SEQ_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        seq = list(map(int, line.split()))
        if len(seq) < 3:
            print(f"⚠️  跳过无效行 {line_idx}：序列过短")
            continue
        # 严格按规则拆分
        user_seq = seq[1:]
        train_part = user_seq[:-2]
        v1 = user_seq[-2]
        v2 = user_seq[-1]
        # 收集物品ID
        train_items.extend(train_part)
        val1_items.append(v1)
        val2_items.append(v2)

print(f"✅ 序列解析完成")
print(f"训练集物品总数: {len(train_items)} | 唯一物品数: {len(set(train_items))}")
print(f"验证集1物品总数: {len(val1_items)} | 唯一物品数: {len(set(val1_items))}")
print(f"验证集2物品总数: {len(val2_items)} | 唯一物品数: {len(set(val2_items))}")

# ----------------------
# 3. 映射为三级SID，统计所有维度的频次Counter
# ----------------------
def map_to_sid(item_list):
    """批量将物品ID映射为三级SID"""
    s1_list, s2_list, s3_list = [], [], []
    for item in item_list:
        item-=1
        if item not in item_sid_mapping:
            continue
        s1, s2, s3 = item_sid_mapping[item]
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)
    return s1_list, s2_list, s3_list

# 映射三级SID
train_s1, train_s2, train_s3 = map_to_sid(train_items)
val1_s1, val1_s2, val1_s3 = map_to_sid(val1_items)
val2_s1, val2_s2, val2_s3 = map_to_sid(val2_items)

# 构建所有维度的统计字典
data_dict = {
    "ori ID": {
        "train": Counter(train_items),
        "val1": Counter(val1_items),
        "val2": Counter(val2_items)
    },
    "first SID": {
        "train": Counter(train_s1),
        "val1": Counter(val1_s1),
        "val2": Counter(val2_s1)
    },
    "second SID": {
        "train": Counter(train_s2),
        "val1": Counter(val1_s2),
        "val2": Counter(val2_s2)
    },
    "third SID": {
        "train": Counter(train_s3),
        "val1": Counter(val1_s3),
        "val2": Counter(val2_s3)
    }
}

# ----------------------
# 4. 核心：斯皮尔曼等级相关系数计算函数
# ----------------------
def calc_spearman_consistency(
    train_counter: Counter,
    val_counter: Counter,
    min_count: int = MIN_COUNT
):
    """
    计算训练集和验证集分布的斯皮尔曼等级相关系数
    :param train_counter: 训练集频次统计
    :param val_counter: 验证集频次统计
    :param min_count: 过滤低频阈值
    :return: 斯皮尔曼系数rho, 显著性p值, 有效类别数
    """
    # 1. 过滤低频，消除噪声
    train_filtered = {k: v for k, v in train_counter.items() if v >= min_count}
    val_filtered = {k: v for k, v in val_counter.items() if v >= min_count}

    # 2. 取两个集合的交集（仅看两边都出现的类别，无交集则无意义）
    common_cats = list(set(train_filtered.keys()).intersection(set(val_filtered.keys())))
    if len(common_cats) < 5:
        return np.nan, np.nan, len(common_cats)

    # 3. 构建排序等级：频次越高，排名越靠前（rank=1为最热门）
    train_rank = {cat: rank+1 for rank, cat in enumerate(sorted(train_filtered.keys(), key=lambda x: -train_filtered[x]))}
    val_rank = {cat: rank+1 for rank, cat in enumerate(sorted(val_filtered.keys(), key=lambda x: -val_filtered[x]))}

    # 4. 提取对齐的排名序列
    train_rank_list = [train_rank[cat] for cat in common_cats]
    val_rank_list = [val_rank[cat] for cat in common_cats]

    # 5. 计算斯皮尔曼系数和p值
    rho, p_value = spearmanr(train_rank_list, val_rank_list)
    return rho, p_value, len(common_cats)

# ----------------------
# 5. 批量计算所有维度的一致性
# ----------------------
print("\n" + "="*80)
print("📊 斯皮尔曼等级相关系数 · 分布一致性评估结果")
print("="*80)
print(f"过滤阈值：频次 ≥ {MIN_COUNT}")
print(f"系数解读：越接近1，分布一致性越强 | p<0.05 结果显著可靠")
print("-"*80)
print(f"{'维度':<10} | {'对比组':<12} | {'斯皮尔曼系数':<12} | {'p值':<10} | {'有效类别数':<8} | {'一致性结论':<10}")
print("-"*80)

# 存储结果用于保存
result_list = []
compare_pairs = [
    ("train vs val1", "train", "val1"),
    ("train vs val2", "train", "val2")
]

for dim_name, dim_data in data_dict.items():
    for pair_name, train_key, val_key in compare_pairs:
        rho, p_val, n_cats = calc_spearman_consistency(
            dim_data[train_key], dim_data[val_key]
        )
        # 一致性结论
        if np.isnan(rho):
            conclusion = "无效"
        elif p_val >= 0.05:
            conclusion = "不显著"
        elif rho >= 0.8:
            conclusion = "极强一致"
        elif rho >= 0.6:
            conclusion = "高度一致"
        elif rho >= 0.4:
            conclusion = "中度一致"
        elif rho >= 0.2:
            conclusion = "弱一致"
        else:
            conclusion = "几乎无一致"
        
        # 打印
        print(f"{dim_name:<10} | {pair_name:<12} | {rho:<12.4f} | {p_val:<10.4f} | {n_cats:<8} | {conclusion:<10}")
        # 保存结果
        result_list.append({
            "维度": dim_name,
            "对比组": pair_name,
            "斯皮尔曼系数": round(rho, 4) if not np.isnan(rho) else np.nan,
            "p值": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "有效类别数": n_cats,
            "一致性结论": conclusion
        })

print("="*80)

# 保存结果到CSV
result_df = pd.DataFrame(result_list)
result_df.to_csv(os.path.join(OUTPUT_DIR, "spearman_consistency_result.csv"), index=False, encoding="utf-8-sig")
print(f"✅ 完整结果已保存到：{os.path.join(OUTPUT_DIR, 'spearman_consistency_result.csv')}")

# ----------------------
# 6. 可视化：Top50热门排序对比（以原始物品ID为例）
# ----------------------
def plot_topk_rank_compare(train_counter, val_counter, k=50, name="ori item ID", save_path=""):
    """绘制TopK热门类别排序对比图"""
    # 取训练集TopK热门
    topk_train = sorted(train_counter.keys(), key=lambda x: -train_counter[x])[:k]
    # 构建排名映射
    train_rank = {cat: i+1 for i, cat in enumerate(sorted(train_counter.keys(), key=lambda x: -train_counter[x]))}
    val_rank = {cat: i+1 for i, cat in enumerate(sorted(val_counter.keys(), key=lambda x: -val_counter[x]))}
    
    # 提取排名
    train_ranks = [train_rank[cat] for cat in topk_train]
    val_ranks = [val_rank.get(cat, len(val_rank)+1) for cat in topk_train]  # 验证集没有的补最低排名
    
    # 画图
    plt.figure(figsize=(16, 8))
    x = np.arange(k)
    plt.scatter(x, train_ranks, label="Train Rank", s=80, color="#2E86AB", zorder=5)
    plt.scatter(x, val_ranks, label="Valid Rank", s=80, color="#F24C4C", zorder=5)
    # 连线看差异
    for i in range(k):
        plt.plot([i, i], [train_ranks[i], val_ranks[i]], color="gray", linestyle="--", alpha=0.5)
    
    plt.gca().invert_yaxis()  # 排名1在顶部，更符合直觉
    plt.title(f"{name} Top{k} Hot item Train vs Valid Rank compare", fontsize=14)
    plt.xlabel("Train TopK Hot type", fontsize=12)
    plt.ylabel("Rank（smaller hoter）", fontsize=12)
    plt.xticks(x, [f"ID{i}" for i in topk_train], rotation=90, fontsize=6)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# 生成所有维度的对比图
for dim_name, dim_data in data_dict.items():
    plot_topk_rank_compare(
        dim_data["train"], dim_data["val1"],
        k=50, name=dim_name,
        save_path=os.path.join(OUTPUT_DIR, f"{dim_name}_top50_rank_compare.png")
    )
print(f"✅ 排名对比图已保存到：{OUTPUT_DIR}")