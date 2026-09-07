"""
数据集分布统计：物品流行度 & 用户活跃度
用法：python analyze_distribution.py（与 run.py 同目录执行，复用相同 hydra 配置）
"""

import os
from collections import Counter

import hydra
import numpy as np
from omegaconf import DictConfig

from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.utils import process_data_split
from run import set_dir


def print_distribution(name: str, values: np.ndarray) -> dict:
    q1  = np.percentile(values, 25)
    q2  = np.percentile(values, 50)
    q3  = np.percentile(values, 75)
    p90 = np.percentile(values, 90)
    p95 = np.percentile(values, 95)

    print(f"\n{'='*50}")
    print(f"  {name} 分布统计（共 {len(values)} 个）")
    print(f"{'='*50}")
    print(f"  最小值 : {values.min():.0f}")
    print(f"  Q1(25%): {q1:.0f}")
    print(f"  中位数  : {q2:.0f}")
    print(f"  Q3(75%): {q3:.0f}")
    print(f"  P90    : {p90:.0f}")
    print(f"  P95    : {p95:.0f}")
    print(f"  最大值 : {values.max():.0f}")
    print(f"  均值   : {values.mean():.1f}")
    print(f"  标准差 : {values.std():.1f}")

    # 按四分位数切 4 组
    bins  = [values.min() - 1, q1, q2, q3, values.max() + 1]
    labels = ["Q1(cold)", "Q2(low)", "Q3(medium)", "Q4(hot)"]
    print(f"\n  四分位分组（推荐评测分组）:")
    for i in range(4):
        lo, hi = int(bins[i]) + 1, int(bins[i + 1])
        mask = (values >= lo) & (values <= hi)
        cnt  = mask.sum()
        print(f"    {labels[i]:12s}: [{lo:5d}, {hi:5d}]  共 {cnt:6d} 个  ({cnt/len(values)*100:.1f}%)")

    return {"q1": q1, "q2": q2, "q3": q3}


def analyze(user_sequence):
    # ── 用户活跃度：用训练序列长度表示（去掉最后 2 个 val/test item）
    user_train_lens = []
    for seq in user_sequence:
        train_len = max(len(seq) - 2, 0)   # 去掉 val + test
        user_train_lens.append(train_len)
    user_train_lens = np.array(user_train_lens)

    # ── 物品流行度：在训练集中出现的次数（只统计训练 item，去掉最后 2 个）
    item_counts = Counter()
    for seq in user_sequence:
        for item in seq[:-2]:
            item_counts[item] += 1

    item_freq = np.array(list(item_counts.values()))

    user_thresholds = print_distribution("用户活跃度（训练序列长度）", user_train_lens)
    item_thresholds = print_distribution("物品流行度（训练集出现次数）", item_freq)

    # ── 额外：展示不同固定阈值下的分组情况（方便和论文常用标准对比）
    # 边界含义：[lo, hi] 左闭右闭
    print("\n\n  ── 物品流行度：常用固定阈值分组参考 ──")
    item_edges = [1, 5, 10, 20, 50, 100, int(item_freq.max()) + 1]
    for i in range(len(item_edges) - 1):
        lo, hi = item_edges[i], item_edges[i + 1] - 1
        cnt = ((item_freq >= lo) & (item_freq <= hi)).sum()
        print(f"    [{lo:4d},{hi:6d}] : {cnt:6d} 个  ({cnt/len(item_freq)*100:.1f}%)")

    print("\n\n  ── 用户活跃度：常用固定阈值分组参考 ──")
    user_edges = [3, 4, 6, 11, int(user_train_lens.max()) + 1]
    user_labels = ["cold(=3)", "low(4-5)", "medium(6-10)", "heavy(>=11)"]
    for i in range(len(user_edges) - 1):
        lo, hi = user_edges[i], user_edges[i + 1] - 1
        cnt = ((user_train_lens >= lo) & (user_train_lens <= hi)).sum()
        print(f"    {user_labels[i]:15s} [{lo:2d},{hi:3d}] : {cnt:6d} 个  ({cnt/len(user_train_lens)*100:.1f}%)")

    print("\n")
    return user_thresholds, item_thresholds, user_train_lens, item_counts


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:
    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    is_steam = config["dataset"]["type"] == "steam"

    data_file, id2meta_file, _ = preprocessing(config["dataset"])

    _, user_sequence = process_data_split(
        config, data_file, id2meta_file, is_steam=is_steam
    )

    out_dir = os.path.join(
        "./results/analysis",
        f"{config['dataset']['type']}_{config['dataset']['name']}",
    )
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "distribution_stats.log")

    # 同时输出到终端和日志文件
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"\n数据集: {config['dataset']['name']}")
        print(f"用户总数: {len(user_sequence)}")
        user_thresholds, item_thresholds, user_lens, item_counts = analyze(user_sequence)

    output = buf.getvalue()
    print(output, end="")                        # 输出到终端
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)                          # 写入日志文件

    save_path = os.path.join(out_dir, "distribution_thresholds.npz")
    np.savez(
        save_path,
        # 四分位阈值（动态，基于数据分布）
        user_q1=user_thresholds["q1"],
        user_q2=user_thresholds["q2"],
        user_q3=user_thresholds["q3"],
        item_q1=item_thresholds["q1"],
        item_q2=item_thresholds["q2"],
        item_q3=item_thresholds["q3"],
        # 固定阈值（用于评测分组，三数据集通用）
        user_fixed=np.array([3, 4, 6, 11]),    # cold/low/medium/heavy 的左边界
        item_fixed=np.array([1, 5, 10, 20]),   # cold/low/medium/hot 的左边界
    )
    msg = f"分位数阈值已保存到: {save_path}\n统计日志已保存到  : {log_path}\n"
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg)


if __name__ == "__main__":
    main()
