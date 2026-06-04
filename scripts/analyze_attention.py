"""
Attention Analysis for TIGER_Residual Model
============================================
诊断问题：在生成后续 SID 时，模型是否关注了前面的残差位置？

分析方式：
  - 在 forward_residual 中开启 output_attentions=True
  - 对每个 NTP 预测步，将其 self-attention 中所有历史 KV cache 位置分为三类：
      BOS, SID（正常 SID token 位置）, RESID（残差位置）
  - 统计每层每头对这三类位置的注意力分布
  - 输出文本报告 + 可选 matplotlib 图表

用法:
    python scripts/analyze_attention.py \
        --ckpt_path results/tiger/Amazon_Beauty/residual_Beauty_seed_42/results/ckpt_best.pt \
        --dataset amazon --dataset_name Beauty --seed 42 --device_id 0 \
        [--output_dir ./attention_analysis] [--num_samples 10] [--plot]
"""

import os
import sys
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Allow importing from parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import T5Config
from src.tiger_residual import TIGER_Residual
from src.load_data import load_data
from utils import set_seed, CustomDataset
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    # fallback: pip install pyyaml
    print("pyyaml not installed. Run: pip install pyyaml"); sys.exit(1)


# ═════════════════════════════════════════════════════════════════════
# 1. 带注意力输出的前向传播（复制 forward_residual 逻辑 + output_attentions）
# ═════════════════════════════════════════════════════════════════════

def forward_residual_with_attentions(model, input_ids, attention_mask, labels_sids,
                                      inputs_embeds=None, encoder_outputs=None):
    """
    与 TIGER_Residual.forward_residual 逻辑完全相同，但每个 decoder 调用都
    加入 output_attentions=True，收集所有 self-attention 权重。

    Returns:
        outputs: dict (同原始 forward_residual)
        attn_records: list of dict — 每次 decoder forward 的注意力
    """
    B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    n_codebook = labels_sids.shape[1]
    n_sem = model.n_semantic_codebook
    num_resid = model.num_residual_levels

    # ── Encoder ──
    if encoder_outputs is None:
        if inputs_embeds is not None:
            encoder_outputs = model.encoder(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            encoder_outputs = model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                return_dict=True,
            )
    encoder_hidden = encoder_outputs.last_hidden_state

    if model.flag_use_output_embedding:
        item_seq_len = attention_mask.sum(-1)
        model.predicted_embedding = model.gather_indexes(
            encoder_outputs.last_hidden_state, item_seq_len - 1)
    else:
        model.predicted_embedding = None

    # ── Cumulative residual targets (if enabled) ──
    cumulative_residual_loss = 0.0
    num_cumul_steps = 0
    cumsum_gt = None
    if model.cumulative_residual_loss_weight > 0:
        gt_cb_embs = []
        for t in range(n_sem):
            cb_idx_t = model._get_codebook_idx(labels_sids[:, t], t)
            gt_cb_embs.append(model.codebook_embs[t][cb_idx_t].float())
        cumsum_gt = [gt_cb_embs[-1]]
        for t in range(n_sem - 2, -1, -1):
            cumsum_gt.insert(0, gt_cb_embs[t] + cumsum_gt[0])

    # ── Step-by-step decode ──
    past_key_values = None
    all_ntp_logits = []
    sid_loss = 0.0
    codebook_loss = 0.0
    num_sid_steps = 0
    num_codebook_steps = 0

    attn_records = []          # 每次 decoder forward 的注意力
    kv_position_types = []     # 跟踪 KV cache 中每个位置的类型: BOS / SID / RESID

    for k in range(n_codebook):
        # === NTP step: predict sid_k ===
        if k == 0:
            dec_input_ids = torch.full(
                (B, 1), model.config.decoder_start_token_id,
                dtype=torch.long, device=device)
            step_type = "BOS"
        else:
            dec_input_ids = labels_sids[:, k - 1:k]
            step_type = "SID"

        decoder_out = model.decoder(
            input_ids=dec_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            output_attentions=True,        # ← 关键：开启注意力输出
        )
        hidden = decoder_out.last_hidden_state[:, -1, :]
        past_key_values = decoder_out.past_key_values
        kv_position_types.append(step_type)

        attn_records.append({
            "k": k,
            "step_type": step_type,
            "self_attentions": decoder_out.attentions,
            "cross_attentions": decoder_out.cross_attentions,
            "kv_len": decoder_out.attentions[0].shape[-1],
        })

        # NTP loss
        logits = model.lm_head(hidden).float()
        all_ntp_logits.append(logits)
        target_k = labels_sids[:, k]
        sid_loss += F.cross_entropy(logits, target_k, reduction="sum") / B
        num_sid_steps += 1

        # === Residual step ===
        if k < num_resid and k + 1 < n_codebook:
            cb_idx_k = model._get_codebook_idx(labels_sids[:, k], k)
            cb_emb = model.codebook_embs[k][cb_idx_k]
            h_latent = model.output_adapters[k](hidden.float())
            residual = (h_latent - cb_emb.float())
            residual_max = model.latent_size * 2.0
            residual = residual.clamp(-residual_max, residual_max)

            residual_adapted = model.input_adapters[k](residual.to(hidden.dtype))
            residual_input = residual_adapted.unsqueeze(1)

            decoder_out_resid = model.decoder(
                inputs_embeds=residual_input,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=True,
            )
            hidden_resid = decoder_out_resid.last_hidden_state[:, -1, :]

            if not torch.isnan(hidden_resid).any():
                past_key_values = decoder_out_resid.past_key_values
            kv_position_types.append("RESID")

            attn_records.append({
                "k": k,
                "step_type": "RESID",
                "self_attentions": decoder_out_resid.attentions,
                "cross_attentions": decoder_out_resid.cross_attentions,
                "kv_len": decoder_out_resid.attentions[0].shape[-1],
            })

            # Codebook loss
            hidden_resid_latent = model.output_adapters[k + 1](hidden_resid.float())
            cb_weights = model.codebook_embs[k + 1]
            logits_cb = hidden_resid_latent @ cb_weights.T.float()
            cb_idx_next = model._get_codebook_idx(labels_sids[:, k + 1], k + 1)
            codebook_loss += F.cross_entropy(logits_cb, cb_idx_next, reduction="sum") / B
            num_codebook_steps += 1

            if model.cumulative_residual_loss_weight > 0:
                target_cumul = cumsum_gt[k + 1]
                if model.cumulative_residual_loss_type == "cosine":
                    cos_sim = F.cosine_similarity(hidden_resid_latent, target_cumul, dim=-1)
                    cumulative_residual_loss += (1.0 - cos_sim).sum() / B
                else:
                    cumulative_residual_loss += F.mse_loss(
                        hidden_resid_latent, target_cumul, reduction="sum") / B
                num_cumul_steps += 1

    # ── Combine losses ──
    sid_loss = sid_loss / num_sid_steps if num_sid_steps > 0 else 0.0
    codebook_loss = codebook_loss / num_codebook_steps if num_codebook_steps > 0 else 0.0
    cumulative_residual_loss = cumulative_residual_loss / num_cumul_steps if num_cumul_steps > 0 else 0.0
    total_loss = (sid_loss + model.codebook_loss_weight * codebook_loss
                  + model.cumulative_residual_loss_weight * cumulative_residual_loss)
    stacked_logits = torch.stack(all_ntp_logits, dim=1)

    outputs = {
        "loss": total_loss,
        "sid_loss": sid_loss,
        "codebook_loss": codebook_loss,
        "cumulative_residual_loss": cumulative_residual_loss,
        "logits": stacked_logits,
        "predicted_embedding": model.predicted_embedding,
    }
    return outputs, attn_records, kv_position_types


# ═════════════════════════════════════════════════════════════════════
# 2. 注意力分析核心
# ═════════════════════════════════════════════════════════════════════

def analyze_attention_per_step(attn_records, kv_position_types):
    """
    对每个 NTP 步，统计 self-attention 中分配给 BOS / SID / RESID 位置的比例。

    self-attention 矩阵结构: [B, n_heads, query_len, kv_len]
    其中 query_len = 1（逐 token 解码），kv_len = 当前 KV cache 长度

    Returns:
        per_ntp_step[k] = {
            "kv_positions": [str],           # 当前 KV cache 中每个位置的类型
            "per_layer": {layer_idx: {"bos_attn":, "sid_attn":, "resid_attn":}},
            "per_head_per_layer": {(layer_idx, head_idx): {同上}},
        }
    """
    result = {}

    for record in attn_records:
        if record["step_type"] not in ("BOS", "SID"):
            continue

        k = record["k"]
        self_attns = record["self_attentions"]  # tuple[L] of [B, nH, 1, kv_len]
        n_layers = len(self_attns)
        kv_len = record["kv_len"]

        # 当前 KV cache 中的位置类型（应该是前 kv_len 个）
        kv_pos = kv_position_types[-kv_len:] if kv_len <= len(kv_position_types) else kv_position_types[:]

        per_layer = {}
        per_head = {}

        for layer_idx in range(n_layers):
            attn = self_attns[layer_idx]                        # [B, nH, 1, kv_len]
            attn_avg = attn.mean(dim=0).mean(dim=1)            # [nH, kv_len]
            n_heads = attn_avg.shape[0]

            # 构建 mask
            bos_mask   = torch.tensor([t == "BOS"   for t in kv_pos], dtype=torch.float32, device=attn_avg.device)
            sid_mask   = torch.tensor([t == "SID"   for t in kv_pos], dtype=torch.float32, device=attn_avg.device)
            resid_mask = torch.tensor([t == "RESID" for t in kv_pos], dtype=torch.float32, device=attn_avg.device)

            # 层级别（所有头平均）
            layer_avg = attn_avg.mean(dim=0)  # [kv_len]
            per_layer[layer_idx] = {
                "bos_attn":   (layer_avg * bos_mask).sum().item(),
                "sid_attn":   (layer_avg * sid_mask).sum().item(),
                "resid_attn": (layer_avg * resid_mask).sum().item(),
            }

            # 每个头
            for h in range(n_heads):
                head_avg = attn_avg[h]  # [kv_len]
                per_head[(layer_idx, h)] = {
                    "bos_attn":   (head_avg * bos_mask).sum().item(),
                    "sid_attn":   (head_avg * sid_mask).sum().item(),
                    "resid_attn": (head_avg * resid_mask).sum().item(),
                }

        result[k] = {
            "kv_positions": kv_pos,
            "per_layer": per_layer,
            "per_head_per_layer": per_head,
        }

    return result


# ═════════════════════════════════════════════════════════════════════
# 3. 文本报告
# ═════════════════════════════════════════════════════════════════════

def print_attention_report(analysis_result, n_layers, n_heads):
    """打印格式化的注意力分析报告。"""
    print("\n" + "=" * 80)
    print("  TIGER_Residual  Self-Attention Analysis Report")
    print("=" * 80)
    print(f"  Decoder layers: {n_layers},  Attention heads per layer: {n_heads}")
    print()

    if not analysis_result:
        print("  No NTP steps found. Check input.")
        return

    # ── 1. Per-step summary ──
    print("-" * 80)
    print("  Section 1 — Per-Step Summary: avg attention to RESIDUAL positions")
    print("            (across all layers & heads)")
    print("-" * 80)
    header = f"  {'Step':>6s}  {'#KV':>5s}  {'#SID':>5s}  {'#RESID':>7s}  {'→SID':>8s}  {'→RESID':>10s}  {'→BOS':>8s}"
    print(header)
    print(f"  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*8}")

    for k in sorted(analysis_result.keys()):
        info = analysis_result[k]
        kv_pos = info["kv_positions"]
        n_sid   = kv_pos.count("SID")
        n_resid = kv_pos.count("RESID")
        n_bos   = kv_pos.count("BOS")

        sid_sum = resid_sum = bos_sum = count = 0.0
        for v in info["per_layer"].values():
            sid_sum   += v["sid_attn"]
            resid_sum += v["resid_attn"]
            bos_sum   += v["bos_attn"]
            count     += 1

        sid_avg   = sid_sum / count if count > 0 else 0
        resid_avg = resid_sum / count if count > 0 else 0
        bos_avg   = bos_sum / count if count > 0 else 0

        print(f"  {k:>6d}  {len(kv_pos):>5d}  {n_sid:>5d}  {n_resid:>7d}  "
              f"{sid_avg:>8.4f}  {resid_avg:>10.4f}  {bos_avg:>8.4f}")

    # ── 2. Per-layer breakdown ──
    print("\n" + "-" * 80)
    print("  Section 2 — Per-Layer Breakdown (averaged over all NTP steps)")
    print("-" * 80)
    print(f"  {'Layer':>6s}  {'→SID':>8s}  {'→RESID':>10s}  {'→BOS':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")

    for layer_idx in range(n_layers):
        sid_sum = resid_sum = bos_sum = 0.0
        step_count = 0
        for k, info in analysis_result.items():
            if layer_idx in info["per_layer"]:
                sid_sum   += info["per_layer"][layer_idx]["sid_attn"]
                resid_sum += info["per_layer"][layer_idx]["resid_attn"]
                bos_sum   += info["per_layer"][layer_idx]["bos_attn"]
                step_count += 1
        if step_count > 0:
            print(f"  {layer_idx:>6d}  {sid_sum/step_count:>8.4f}  "
                  f"{resid_sum/step_count:>10.4f}  {bos_sum/step_count:>8.4f}")

    # ── 3. Top-10 heads attending to RESID ──
    print("\n" + "-" * 80)
    print("  Section 3 — Top-10 Heads Attending to RESIDUAL")
    print("-" * 80)
    head_scores = defaultdict(float)
    head_counts = defaultdict(int)
    for k, info in analysis_result.items():
        for (ly, hd), vals in info["per_head_per_layer"].items():
            head_scores[(ly, hd)] += vals["resid_attn"]
            head_counts[(ly, hd)] += 1
    head_avg = {key: total / head_counts[key] for key, total in head_scores.items()}
    sorted_heads = sorted(head_avg.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Layer':>6s}  {'Head':>5s}  {'→RESID':>10s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*10}")
    for (ly, hd), score in sorted_heads[:10]:
        print(f"  {ly:>6d}  {hd:>5d}  {score:>10.4f}")

    # ── 4. Diagnosis ──
    print("\n" + "-" * 80)
    print("  Section 4 — DIAGNOSIS")
    print("-" * 80)
    total_resid = total_attn = 0.0
    for k, info in analysis_result.items():
        if k == 0:
            continue
        for vals in info["per_layer"].values():
            total_resid += vals["resid_attn"]
            total_attn  += (vals["bos_attn"] + vals["sid_attn"] + vals["resid_attn"])
    avg_resid_pct = (total_resid / total_attn * 100) if total_attn > 0 else 0

    print(f"  Average self-attention to RESIDUAL positions: {avg_resid_pct:.2f}%")
    print()
    if avg_resid_pct < 1.0:
        print("  ⚠️  WARNING: Residual attention is VERY LOW (< 1%).")
        print("      → 模型基本忽略了残差位置的信息！")
        print("      → 可能原因：")
        print("        1. 残差信号太弱（检查 adapter 初始化）")
        print("        2. 残差位置对 decoder 来说是 out-of-distribution")
        print("        3. 模型学会了仅依赖 NTP teacher forcing 而不利用残差")
        print("      → 建议：增大 codebook_loss_weight、调整 adapter 学习率")
    elif avg_resid_pct < 5.0:
        print("  ⚡ NOTE: Residual attention is relatively low (< 5%).")
        print("      → 残差信息可能没有被充分利用。")
    else:
        print("  ✓  Residual attention is healthy (> 5%).")
        print("      → 模型有效地关注了残差位置。")
    print("=" * 80 + "\n")


# ═════════════════════════════════════════════════════════════════════
# 4. 可选图表 (matplotlib)
# ═════════════════════════════════════════════════════════════════════

def plot_attention_heatmap(analysis_result, n_layers, n_heads, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skip plots.")
        return
    os.makedirs(output_dir, exist_ok=True)

    steps = sorted(analysis_result.keys())
    layers = list(range(n_layers))

    # ── Heatmap: per-step × per-layer residual attention ──
    data = np.zeros((len(layers), len(steps)))
    for j, k in enumerate(steps):
        for i, ly in enumerate(layers):
            if ly in analysis_result[k]["per_layer"]:
                data[i, j] = analysis_result[k]["per_layer"][ly]["resid_attn"]

    fig, ax = plt.subplots(figsize=(max(8, len(steps)*1.5), max(4, len(layers)*0.4)))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"sid_{k}" for k in steps])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{i}" for i in layers])
    ax.set_xlabel("NTP Step"); ax.set_ylabel("Decoder Layer")
    ax.set_title("Self-Attention → RESIDUAL Positions (per step × layer)")
    plt.colorbar(im, ax=ax, label="Residual Attention")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "resid_attn_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── Per-head heatmap ──
    hdata = np.zeros((n_layers, n_heads))
    for ly in range(n_layers):
        for hd in range(n_heads):
            total = cnt = 0.0
            for k, info in analysis_result.items():
                if (ly, hd) in info["per_head_per_layer"]:
                    total += info["per_head_per_layer"][(ly, hd)]["resid_attn"]
                    cnt += 1
            hdata[ly, hd] = total / cnt if cnt > 0 else 0

    fig, ax = plt.subplots(figsize=(max(6, n_heads), max(4, n_layers*0.5)))
    im = ax.imshow(hdata, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f"L{i}" for i in layers])
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("Self-Attention → RESIDUAL (avg over all NTP steps)")
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f"{hdata[i,j]:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if hdata[i,j] > hdata.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, label="Residual Attention")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "resid_attn_per_head.png"), dpi=150)
    plt.close(fig)

    # ── Stacked bar: attention distribution per step ──
    data_bos, data_sid, data_resid = [], [], []
    for k in steps:
        bs = ss = rs = cnt = 0.0
        for v in analysis_result[k]["per_layer"].values():
            bs += v["bos_attn"]; ss += v["sid_attn"]; rs += v["resid_attn"]; cnt += 1
        data_bos.append(bs/cnt if cnt else 0)
        data_sid.append(ss/cnt if cnt else 0)
        data_resid.append(rs/cnt if cnt else 0)

    fig, ax = plt.subplots(figsize=(max(8, len(steps)*1.2), 5))
    x = np.arange(len(steps)); w = 0.6
    ax.bar(x, data_bos, w, label="BOS", color="#3498db")
    ax.bar(x, data_sid, w, bottom=data_bos, label="SID", color="#2ecc71")
    ax.bar(x, data_resid, w, bottom=[a+b for a,b in zip(data_bos,data_sid)],
           label="RESID", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sid_{k}" for k in steps])
    ax.set_ylabel("Avg Attention"); ax.set_title("Self-Attention Distribution per NTP Step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attn_distribution_per_step.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


# ═════════════════════════════════════════════════════════════════════
# 5. Main
# ═════════════════════════════════════════════════════════════════════

def build_from_checkpoint(args):
    """从 checkpoint + config_yaml 重建模型 + 数据"""
    import glob as _glob

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # ── 读取 config YAML ──
    if args.config_yaml and os.path.exists(args.config_yaml):
        with open(args.config_yaml, "r") as f:
            wandb_cfg = yaml.safe_load(f)
        # wandb YAML 的顶层 key 带 .value 嵌套，提取实际值
        def _extract(cfg):
            """wandb 格式: key -> {value: ...}，提取实际内容"""
            out = {}
            for k, v in cfg.items():
                if isinstance(v, dict) and "value" in v:
                    out[k] = v["value"]
                else:
                    out[k] = v
            return out
        flat_cfg = _extract(wandb_cfg)
        dataset_cfg = flat_cfg.get("dataset", {})
        method_cfg  = flat_cfg.get("method", {})
        tiger_cfg   = dataset_cfg.get("TIGER", {})
        t5_cfg      = tiger_cfg.get("T5", {})
        rqvae_cfg   = dataset_cfg.get("RQ-VAE", {})
        print(f"  Loaded config from YAML: {args.config_yaml}")
    else:
        print("  ⚠ No config_yaml provided, using defaults — may cause mismatches!")
        dataset_cfg = {"type": "Amazon", "name": args.dataset_name,
                       "content_model": args.content_model or "sentence-t5-xxl",
                       "max_items_per_seq": 20,
                       "features_needed": ["title", "price", "brand", "categories"],
                       "prompt_format": "amazon",
                       "raw_data_path": args.raw_data_dir or "./ID_generation/preprocessing/raw_data/",
                       "processed_data_path": args.processed_dir or "./ID_generation/preprocessing/processed/"}
        method_cfg  = {"include_user_id": False, "use_id": "sid",
                       "flag_add_input_embedding": False, "flag_use_output_embedding": False,
                       "embedding_loss_weight": 0, "sid_loss_weight": 1,
                       "use_residual_decoder": True, "codebook_loss_weight": 1.0,
                       "num_residual_levels": None,
                       "embedding_head_dict": {"use_new_init": False, "embed_target": "ground_truth",
                                              "embed_proj_type": "mlp"}}
        tiger_cfg   = {}
        t5_cfg      = {"encoder_layers": 6, "decoder_layers": 6, "d_model": 128,
                       "d_ff": 1024, "num_heads": 6, "d_kv": 64, "dropout_rate": 0.2,
                       "feed_forward_proj": "relu", "initializer_factor": 0.02}
        rqvae_cfg   = {"code_book_size": 256, "latent_dim": 128,
                       "input_dim": 768, "hidden_dim": [768, 512, 256]}

    # ── 数据路径 ──
    raw_data_dir  = dataset_cfg.get("raw_data_path", args.raw_data_dir or "./ID_generation/preprocessing/raw_data/")
    processed_dir = dataset_cfg.get("processed_data_path", args.processed_dir or "./ID_generation/preprocessing/processed/")
    content_model_name = os.path.basename((dataset_cfg.get("content_model", args.content_model or "sentence-t5-xxl")).rstrip("/"))

    data_file = args.data_file
    if not data_file:
        ds_name = dataset_cfg.get("name", args.dataset_name)
        ds_lower = ds_name.lower().replace(" ", "_")
        for c in [os.path.join(raw_data_dir, ds_lower, f"{ds_name}.inter"),
                  os.path.join(raw_data_dir, ds_lower, f"{ds_lower}.inter")]:
            if os.path.exists(c): data_file = c; break
        if not data_file:
            for pat in [os.path.join(raw_data_dir, "**", "*.inter"),
                        os.path.join(raw_data_dir, "**", "*.csv"),
                        os.path.join(raw_data_dir, "**", "*.txt")]:
                hits = _glob.glob(pat, recursive=True)
                if hits: data_file = hits[0]; break
    if not data_file or not os.path.exists(data_file):
        print(f"  ⚠ data_file not found! Set --data_file."); sys.exit(1)

    id2meta_file = args.id2meta_file
    if not id2meta_file:
        ds_name = dataset_cfg.get("name", args.dataset_name)
        for pat in [os.path.join(processed_dir, f"{ds_name}*{content_model_name}*.json"),
                    os.path.join(processed_dir, f"{ds_name}*.json")]:
            hits = _glob.glob(pat)
            if hits: id2meta_file = hits[0]; break
    if not id2meta_file or not os.path.exists(id2meta_file):
        print(f"  ⚠ id2meta_file not found! Set --id2meta_file."); sys.exit(1)

    embedding_path = args.embedding_path or os.path.join(
        processed_dir, f"{dataset_cfg.get('name', args.dataset_name)}_{content_model_name}_embeddings.pt")
    id_save_location = args.sid_path or f"./ID_generation/ID/{dataset_cfg.get('name', args.dataset_name)}_{content_model_name}_{args.seed}.pkl"

    print(f"  data_file:     {data_file}")
    print(f"  id2meta_file:  {id2meta_file}")
    print(f"  embedding:     {embedding_path}")
    print(f"  sid_path:      {id_save_location}")

    # ── 构建 full_config (给 process_data_split / process_embeddings 用) ──
    full_config = {
        "dataset": dataset_cfg, "seed": args.seed, "device_id": args.device_id,
        "n_positions": tiger_cfg.get("n_positions", 102),
        "TIGER": {"n_positions": tiger_cfg.get("n_positions", 102),
                  "save_every_n_epochs": 10, "T5": t5_cfg, "RQ-VAE": rqvae_cfg,
                  "trainer": tiger_cfg.get("trainer", {"batch_size": 1024})},
        "method": method_cfg, "RQ-VAE": rqvae_cfg,
    }

    # ── Process data split ──
    from ID_generation.utils import process_data_split, process_embeddings
    is_steam = (dataset_cfg.get("type", "").lower() == "steam")
    id_split, user_sequence = process_data_split(full_config, data_file, id2meta_file, is_steam=is_steam)

    # ── Load embeddings ──
    item_embedding = process_embeddings(full_config, device, id2meta_file, embedding_path)

    # ── SID file ──
    if not os.path.exists(id_save_location):
        print(f"  ⚠ SID file not found at {id_save_location}, re-training RQ-VAE...")
        from ID_generation.train_rqvae import train as train_sid_fn
        train_sid_fn(full_config, device, item_embedding, id_split, id_save_location)

    # ── 加载 checkpoint ──
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # ── 从 state_dict shape 自动推断 n_sem / vocab_size（与 YAML 配置交叉验证） ──
    n_sem_auto   = state_dict["semantic_pos.weight"].shape[0] - 1
    vocab_auto   = state_dict["shared.weight"].shape[0]
    d_model_auto = state_dict["shared.weight"].shape[1]
    print(f"  [Auto-detect from ckpt] n_sem={n_sem_auto}, vocab_size={vocab_auto}, d_model={d_model_auto}")

    # ── 从 config 提取所有模型参数 ──
    n_sem       = n_sem_auto          # 强制用 checkpoint 的实际值
    vocab_size  = vocab_auto
    d_model     = d_model_auto
    cb_size     = rqvae_cfg.get("code_book_size", 256)
    latent_sz   = rqvae_cfg.get("latent_dim", rqvae_cfg.get("latent_size", d_model))
    max_items   = dataset_cfg.get("max_items_per_seq", 20)
    n_pos       = tiger_cfg.get("n_positions", 102)
    encoder_l   = t5_cfg.get("encoder_layers", 6)
    decoder_l   = t5_cfg.get("decoder_layers", 6)
    d_ff        = t5_cfg.get("d_ff", 1024)
    n_heads     = t5_cfg.get("num_heads", 6)
    d_kv        = t5_cfg.get("d_kv", 64)
    dropout     = t5_cfg.get("dropout_rate", 0.2)
    ff_proj     = t5_cfg.get("feed_forward_proj", "relu")
    init_f      = t5_cfg.get("initializer_factor", 0.02)
    flag_emb    = method_cfg.get("flag_use_output_embedding", False)
    flag_text   = method_cfg.get("flag_add_input_embedding", False)
    emb_head    = method_cfg.get("embedding_head_dict") or {
        "use_new_init": False, "embed_target": "ground_truth", "embed_proj_type": "mlp"}
    cb_loss_w   = method_cfg.get("codebook_loss_weight", 1.0)
    # num_residual_levels 在 YAML 中可能是 null，此时 fallback 为 n_sem-1
    num_resid   = method_cfg.get("num_residual_levels") or (n_sem - 1)
    soft_k      = method_cfg.get("soft_label_K", 0)
    soft_temp   = method_cfg.get("soft_label_temperature", 1.0)
    cumul_w     = method_cfg.get("cumulative_residual_loss_weight", 0.0)
    cumul_t     = method_cfg.get("cumulative_residual_loss_type", "mse")
    include_user = method_cfg.get("include_user_id", False)

    print(f"  Config summary: decoder={decoder_l}L×{n_heads}H, d_model={d_model}, "
          f"cb_size={cb_size}, n_sem={n_sem}, num_resid={num_resid}")

    # ── 构建模型 ──
    model_config = T5Config(
        num_layers=encoder_l, num_decoder_layers=decoder_l,
        d_model=d_model, d_ff=d_ff, num_heads=n_heads, d_kv=d_kv,
        dropout_rate=dropout, vocab_size=vocab_size,
        pad_token_id=0, eos_token_id=vocab_size - 1, decoder_start_token_id=0,
        feed_forward_proj=ff_proj, n_positions=n_pos,
        layer_norm_epsilon=1e-8, initializer_factor=init_f,
    )

    rqvae_cb_weights = None
    for try_p in [f"./ID_generation/ID/codebook_weights_{args.seed}.pt",
                  os.path.join(os.path.dirname(id_save_location), f"codebook_weights_{args.seed}.pt"),
                  args.codebook_path]:
        if try_p and os.path.exists(try_p):
            try:
                rqvae_cb_weights = torch.load(try_p, map_location=device)
                print(f"  Loaded codebook weights from {try_p}"); break
            except Exception: pass

    model = TIGER_Residual(
        config=model_config, n_semantic_codebook=n_sem,
        max_items_per_seq=max_items,
        flag_use_output_embedding=flag_emb,
        flag_use_learnable_text_embed=flag_text,
        embedding_head_dict=emb_head,
        rqvae_codebook_weights=rqvae_cb_weights,
        codebook_size=cb_size, latent_size=latent_sz,
        codebook_loss_weight=cb_loss_w, num_residual_levels=num_resid,
        soft_label_K=soft_k, soft_label_temperature=soft_temp,
        cumulative_residual_loss_weight=cumul_w,
        cumulative_residual_loss_type=cumul_t,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [INFO] Missing keys: {len(missing)}")
    if unexpected:
        print(f"  [INFO] Unexpected keys: {len(unexpected)}")
    model.eval()

    # ── 构建 DataLoader ──
    unseen_val, unseen_test, seen = id_split["unseen_val"], id_split["unseen_test"], id_split["seen"]
    method_config_load = {
        "include_user_id": include_user,
        "flag_add_input_embedding": flag_text,
        "flag_use_output_embedding": flag_emb,
        "embedding_head_dict": emb_head or {},
    }
    (training_data, val_data, test_data, _, _,
     _, _, _, _, _, _, _) = load_data(
        path=id_save_location, user_sequence=user_sequence,
        unseen_val=unseen_val, unseen_test=unseen_test, seen=seen,
        item_embedding=item_embedding, method_config=method_config_load,
        max_length=n_pos, codebook_size=cb_size, max_items_per_seq=max_items,
    )
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.num_samples,
                             shuffle=False, num_workers=0)
    print(f"  Model loaded. Test samples: {len(test_dataset)}")
    return model, test_loader, device, decoder_l, n_heads, flag_text


def main():
    parser = argparse.ArgumentParser(description="TIGER_Residual Attention Analysis")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, default=None,
                        help="wandb config YAML (contains all training config)")
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--dataset_name", type=str, default="Beauty")
    parser.add_argument("--content_model", type=str, default="sentence-t5-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./attention_analysis")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="# test samples to analyze")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    # 数据路径（可从 sh 脆指定）
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to user-item interaction file")
    parser.add_argument("--id2meta_file", type=str, default=None,
                        help="Path to item-to-metadata JSON file")
    parser.add_argument("--embedding_path", type=str, default=None,
                        help="Path to item embeddings .pt file")
    parser.add_argument("--sid_path", type=str, default=None,
                        help="Path to semantic ID .pkl file")
    parser.add_argument("--codebook_path", type=str, default=None,
                        help="Path to codebook_weights .pt file")
    parser.add_argument("--raw_data_dir", type=str, default=None,
                        help="Raw data directory")
    parser.add_argument("--processed_dir", type=str, default=None,
                        help="Processed data directory")
    args = parser.parse_args()

    model, test_loader, device, n_layers, n_heads, flag_text = build_from_checkpoint(args)

    print(f"\n{'='*60}")
    print(f"  Running attention analysis on {args.num_samples} samples...")
    print(f"{'='*60}")

    n_analyzed = 0
    for batch in test_loader:
        if n_analyzed >= args.num_samples:
            break

        remain = args.num_samples - n_analyzed
        # batch is a dict from CustomDataset:
        #   input_sids, attention_mask_sids, labels_sids, input_embeddings, ...
        # forward_residual expects: input_ids, attention_mask, labels_sids, inputs_embeds
        input_ids = batch["input_sids"][:remain].to(device)
        attention_mask = batch["attention_mask_sids"][:remain].to(device)
        labels_sids = batch["labels_sids"][:remain].to(device)
        # Only use input_embeddings if flag_add_input_embedding is True
        if flag_text and "input_embeddings" in batch and batch["input_embeddings"].numel() > 0:
            inputs_embeds = batch["input_embeddings"][:remain].to(device)
        else:
            inputs_embeds = None

        with torch.no_grad():
            outputs, attn_records, kv_types = forward_residual_with_attentions(
                model, input_ids, attention_mask, labels_sids,
                inputs_embeds=inputs_embeds,
            )

        analysis = analyze_attention_per_step(attn_records, kv_types)
        print_attention_report(analysis, n_layers, n_heads)

        n_analyzed += input_ids.shape[0]

    if args.plot and n_analyzed > 0:
        print("Generating plots...")
        plot_attention_heatmap(analysis, n_layers, n_heads, args.output_dir)

    print(f"\nDone. Analyzed {n_analyzed} samples.")


if __name__ == "__main__":
    main()