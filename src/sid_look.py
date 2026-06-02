import pickle
import numpy as np
import pandas as pd
import matplotlib
# 服务器无GUI，强制保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import os

# ===================== 【配置项】 =====================
# 你的pkl文件路径
PKL_FILE = "/home/sunyijia/cuihongxi/liger/ID_generation/ID/Sports_and_Outdoors_sentence-t5-xxl_42.pkl"
# 统一输出目录（自动创建）
OUTPUT_DIR = "/home/sunyijia/cuihongxi/liger/ID_generation/ID/Sports/sid_distribution_output"
# ======================================================

# 自动创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载RQ-VAE 3级编码数据
with open(PKL_FILE, "rb") as f:
    codes = pickle.load(f)

# 2. 基础信息打印
print("="*60)
print("📦 RQ-VAE 3-Tier Codebook SID Overview")
print("="*60)
print(f"Total Items: {codes.shape[0]}")
print(f"Codebook Levels: {codes.shape[1]} (3 levels ✅)")
print("ID Range per level:")
for i in range(3):
    print(f"  Level {i+1}: min={codes[:,i].min()}, max={codes[:,i].max()}")
print("="*60)

# 3. 统计函数
def stat_level_distribution(level_data):
    counter = Counter(level_data)
    df = pd.DataFrame(list(counter.items()), columns=["SID", "count"])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    total = len(level_data)
    df["ratio (%)"] = (df["count"] / total * 100).round(2)
    return df, counter

# 4. 逐3级分析 + 输出到指定目录
level_names = ["Level_1", "Level_2", "Level_3"]
for idx, name in enumerate(level_names):
    print(f"\n🔥 {name} SID Distribution")
    df, cnt = stat_level_distribution(codes[:, idx])
    
    # 打印统计信息
    print(f"Unique SIDs: {len(cnt)}")
    print("Top 20 High-Frequency SIDs:")
    print(df.head(20).to_string(index=False))
    
    # 保存分布图 → 输出目录
    png_path = os.path.join(OUTPUT_DIR, f"{name}_sid_dist.png")
    plt.figure(figsize=(14, 5))
    top50 = df.head(50)
    plt.bar(top50["SID"].astype(str), top50["count"], color='#2E86AB')
    plt.title(f"{name} - Top50 SID Distribution", fontsize=14)
    plt.xticks(rotation=60, fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    # 保存CSV → 输出目录
    csv_path = os.path.join(OUTPUT_DIR, f"{name}_sid_distribution.csv")
    df.to_csv(csv_path, index=False)

# 最终提示
print("\n✅ Analysis finished! All files saved to:", os.path.abspath(OUTPUT_DIR))
print("📊 Generated files:")
print("   - 3x PNG distribution charts")
print("   - 3x CSV distribution tables")