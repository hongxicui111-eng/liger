
import pickle
import numpy as np
import os
from collections import Counter
import pandas as pd

# ===================== 【核心配置项 · 请修改为你的路径】 =====================
# 1. RQ-VAE 3级码本pkl文件 (物品ID -> 3级SID)
PKL_FILE = "/home/sunyijia/cuihongxi/liger/ID_generation/ID/Sports_and_Outdoors_sentence-t5-xxl_42.pkl"
# 2. 用户交互训练文件路径
TRAIN_SEQ_FILE = "/home/sunyijia/cuihongxi/liger/ID_generation/preprocessing/processed/Sports_and_Outdoors.txt"  # 👈 改成你的训练文件路径
# 3. 统一输出目录
OUTPUT_DIR = "/home/sunyijia/cuihongxi/liger/ID_generation/preprocessing/processed/Sports/user_sequence_statistics"
# ==========================================================================

# 自动创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# 1. 加载3级码本，构建 物品ID → 三级SID 映射字典
# ----------------------
print("🔄 Loading 3-level codebook...")
with open(PKL_FILE, "rb") as f:
    item2sid = pickle.load(f)  # shape: [num_items, 3]

# 物品ID = 数组索引，直接映射
num_items = item2sid.shape[0]
item_sid_mapping = {
    item_id: [int(s1), int(s2), int(s3)]
    for item_id, (s1, s2, s3) in enumerate(item2sid)
}
print(f"✅ Loaded {num_items} items with 3-level SID mapping")

# ----------------------
# 2. 解析用户序列，拆分训练/验证集，收集物品ID
# ----------------------
print("🔄 Parsing user interaction sequences...")
train_items = []      # 序列[:-2] 训练数据物品
val_target1 = []      # 序列[-2] 验证target1
val_target2 = []      # 序列[-1] 验证target2

with open(TRAIN_SEQ_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        # 拆分每行数据：user_id + 交互序列
        seq = list(map(int, line.split()))
        if len(seq) < 3:
            print(f"⚠️  Skip invalid line {line_idx}: too short")
            continue
        
        # 严格按规则拆分
        user_seq = seq[1:]  # 去掉第一个user_id，剩余为物品交互序列
        train_part = user_seq[:-2]   # 训练序列
        v1 = user_seq[-2]            # 验证target1
        v2 = user_seq[-1]            # 验证target2

        # 收集物品ID
        train_items.extend(train_part)
        val_target1.append(v1)
        val_target2.append(v2)

print(f"✅ Parsing finished!")
print(f"Train items count: {len(train_items)}")
print(f"Val target1 count: {len(val_target1)}")
print(f"Val target2 count: {len(val_target2)}")

# ----------------------
# 3. 统计函数：保存频次为CSV
# ----------------------
def save_count(data: list, name: str, subdir: str = ""):
    """统计频次并保存为CSV"""
    cnt = Counter(data)
    df = pd.DataFrame(list(cnt.items()), columns=["ID", "count"])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    df["ratio (%)"] = (df["count"] / sum(cnt.values()) * 100).round(2)
    save_path = os.path.join(OUTPUT_DIR, subdir, f"{name}.csv")
    df.to_csv(save_path, index=False)
    return df

# 创建子文件夹
os.makedirs(os.path.join(OUTPUT_DIR, "item_id"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "sid_level1"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "sid_level2"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "sid_level3"), exist_ok=True)

# ----------------------
# 4. 统计 原始物品ID 频次
# ----------------------
print("🔄 Counting original item IDs...")
save_count(train_items, "train_item_counts", "item_id")
save_count(val_target1, "val_target1_item_counts", "item_id")
save_count(val_target2, "val_target2_item_counts", "item_id")

# ----------------------
# 5. 映射为三级SID，并统计每一级频次
# ----------------------
print("🔄 Counting 3-level SIDs...")
def map_to_sid(items):
    """批量将物品ID映射为三级SID"""
    s1_list, s2_list, s3_list = [], [], []
    for item in items:
        if item not in item_sid_mapping:
            continue
        s1, s2, s3 = item_sid_mapping[item-1] #码本里，id从0开始
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)
    return s1_list, s2_list, s3_list

# 训练集 SID
train_s1, train_s2, train_s3 = map_to_sid(train_items)
save_count(train_s1, "train_sid1_counts", "sid_level1")
save_count(train_s2, "train_sid2_counts", "sid_level2")
save_count(train_s3, "train_sid3_counts", "sid_level3")

# 验证target1 SID
v1_s1, v1_s2, v1_s3 = map_to_sid(val_target1)
save_count(v1_s1, "val1_sid1_counts", "sid_level1")
save_count(v1_s2, "val1_sid2_counts", "sid_level2")
save_count(v1_s3, "val1_sid3_counts", "sid_level3")

# 验证target2 SID
v2_s1, v2_s2, v2_s3 = map_to_sid(val_target2)
save_count(v2_s1, "val2_sid1_counts", "sid_level1")
save_count(v2_s2, "val2_sid2_counts", "sid_level2")
save_count(v2_s3, "val2_sid3_counts", "sid_level3")

# ----------------------
# 完成提示
# ----------------------
print("\n🎉 All statistics finished!")
print(f"📂 All results saved to: {os.path.abspath(OUTPUT_DIR)}")
print("\n📊 Output structure:")
print("├── item_id/              # 原始物品ID频次")
print("├── sid_level1/           # 第一级SID频次")
print("├── sid_level2/           # 第二级SID频次")
print("└── sid_level3/           # 第三级SID频次")