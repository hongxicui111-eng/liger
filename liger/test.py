"""
Standalone test program for TIGER / Liger / Hybrid models.

Loads a trained checkpoint and runs the **standard evaluation metrics**
(genret / dense / uni for Liger, prefix-then-dense for Hybrid) plus the
**per-layer SID accuracy** analysis:

  - Gen{n}_Recall@10 / Gen{n}_NDCG@10 : generative / hybrid retrieval metrics
  - Recall@10 / NDCG@10               : dense retrieval metrics
  - Layer_k_Recall@10 : cumulative recall after generating k SID layers
  - Layer_k_NDCG@10   : cumulative NDCG after generating k SID layers
  - Layer_k_Acc       : conditional accuracy of the k-th SID given the
                        first (k-1) SIDs are correct

Usage (run from the ``liger/`` directory):

    python test.py \
        dataset=amazon dataset.name=Beauty \
        method=setting test_method=liger \
        seed=42 device_id=0 \
        experiment_id="liger_Beauty" \
        checkpoint_path="results/liger/Amazon_Beauty/liger_Beauty_seed_42/results/ckpt_best.pt"

If ``checkpoint_path`` is omitted, the program looks for the best
checkpoint at the default output path:

    results/{test_method}/{type}_{name}/{experiment_id}_seed_{seed}/results/ckpt_best.pt
"""

import os
import sys
import json
import traceback

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import T5Config

from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.utils import process_data_split, process_embeddings
from src.evaluation import (
    evaluate,
    evaluate_dense_ids,
    evaluate_dense_sids,
    evaluate_layer_accuracy,
    evaluate_prefix_then_dense,
    generate_then_dense,
)
from src.load_data import build_prefix2items, load_data
from src.tiger import TIGER
from utils import CustomDataset, set_seed


class set_dir:
    """Mirrors the path setup in ``run.py`` so the SID / embedding paths
    resolve identically."""

    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        if config["test_method"] in ["tiger", "liger", "hybrid"]:
            self.rqvae_save_dir = "./ID_generation/ID/"
            os.makedirs(self.rqvae_save_dir, exist_ok=True)

            id_filename = (
                f"{config['dataset']['name']}_{config['dataset']['content_model']}"
            )

            sasrec_cfg = config["dataset"].get("SASRec", {})
            fusion_tag = ""
            if sasrec_cfg.get("enabled", False):
                fm = sasrec_cfg.get("fusion_mode", "fused")
                fusion_tag = "_fused" if fm == "fused" else "_cf"
            self.id_save_location = os.path.join(
                self.rqvae_save_dir,
                id_filename + f"{fusion_tag}_{config['seed']}.pkl",
            )

            self.sasrec_save_dir = "./ID_generation/sasrec/ckpt/"
            os.makedirs(self.sasrec_save_dir, exist_ok=True)
            self.sasrec_fused_path = os.path.join(
                self.sasrec_save_dir, id_filename + f"{fusion_tag}_{config['seed']}.pt"
            )

        self.embedding_save_name = f"_{config['dataset']['content_model']}"
        self.embedding_save_path = os.path.join(
            self.directory_processed, id_filename + "_embeddings.pt"
        )

        self.result_save_dir = f"./results/{config['test_method']}/"
        os.makedirs(self.result_save_dir, exist_ok=True)

    def set_config(self, config):
        config["dataset"]["raw_data_path"] = self.directory
        config["dataset"]["processed_data_path"] = self.directory_processed
        config["output_path"] = os.path.join(
            self.result_save_dir,
            f"{config['dataset']['type']}_{config['dataset']['name']}",
            f"{config['experiment_id']}_seed_{config['seed']}",
        )
        os.makedirs(config["output_path"], exist_ok=True)

        if config.get("force_rerun", False):
            run_dir = config["output_path"]
            sasrec_cfg = config["dataset"].get("SASRec", {})
            fusion_tag = ""
            if sasrec_cfg.get("enabled", False):
                fm = sasrec_cfg.get("fusion_mode", "fused")
                fusion_tag = "_fused" if fm == "fused" else "_cf"
            self.sasrec_fused_path = os.path.join(
                run_dir, f"sasrec{fusion_tag}_{config['seed']}.pt"
            )
            self.id_save_location = os.path.join(
                run_dir, f"sid{fusion_tag}_{config['seed']}.pkl"
            )

        return config


# ---------------------------------------------------------------------------
# Standard evaluation helpers (mirrors evaluate_helper from training.py)
# ---------------------------------------------------------------------------

KEYS = [10]  # Recall@k / NDCG@k cutoffs
RETRIEVE_KEY = [20, 40, 60, 80, 100]  # retrieve-then-rank candidate counts (Liger)


def _add_log(logs, result_dict, result_name, prefix):
    logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
    return logs


def _run_genret_eval(model, dataloader, all_semantic_ids, device,
                     method_config, name, item2sid):
    """Pure generative retrieval (beam search → SID match)."""
    if method_config["sid_loss_weight"] > 0:
        recall_dict, ndcg_dict, _, _ = evaluate(
            model, dataloader, all_semantic_ids, device,
            method_config=method_config, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
        )
        for key in recall_dict.keys():
            logs_entry = f"Recall@{key}"
            logs_ndcg = f"NDCG@{key}"
            print(f"  {name}/{logs_entry}: {np.mean(recall_dict[key]):.6f}")
            print(f"  {name}/{logs_ndcg}: {np.mean(ndcg_dict[key]):.6f}")
        return recall_dict, ndcg_dict
    return None, None


def _run_dense_eval(model, dataloader, device, method_config, name,
                    item2sid, item_embedding):
    """Pure dense retrieval (embedding dot product)."""
    if method_config["embedding_loss_weight"] > 0:
        if method_config["use_id"] == "item_id":
            recall_dict, ndcg_dict = evaluate_dense_ids(
                model, dataloader, device, item2sid,
                item_embedding=item_embedding, method_config=method_config, KEYS=KEYS,
            )
        else:
            recall_dict, ndcg_dict = evaluate_dense_sids(
                model, dataloader, device, item2sid,
                item_embedding=item_embedding, method_config=method_config, KEYS=KEYS,
            )
        for key in recall_dict.keys():
            print(f"  {name}/Recall@{key}: {np.mean(recall_dict[key]):.6f}")
            print(f"  {name}/NDCG@{key}: {np.mean(ndcg_dict[key]):.6f}")
        return recall_dict, ndcg_dict
    return None, None


def _run_unified_eval(model, dataloader, unseen_semantic_ids, device,
                      method_config, name, returned_cand, returned_embd,
                      item2sid, item_embedding, include_cold=True):
    """Generate-then-dense (Liger unified).

    When ``include_cold=False``, computes the NoCold variant — the candidate
    pool is only the generative (beam-search) SIDs, without appending the
    cold-start SIDs.
    """
    if (method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0):
        recall_dict, ndcg_dict = generate_then_dense(
            model, dataloader, unseen_semantic_ids, device,
            method_config=method_config, returned_cand=returned_cand,
            returned_embd=returned_embd, item2sid=item2sid,
            item_embedding=item_embedding, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
            include_cold=include_cold,
        )
        suffix = "" if include_cold else "_NoCold"
        for _retrieve_key in recall_dict.keys():
            for key in recall_dict[_retrieve_key].keys():
                r = np.mean(recall_dict[_retrieve_key][key])
                n = np.mean(ndcg_dict[_retrieve_key][key])
                print(f"  {name}{suffix}/Gen{_retrieve_key}_Recall@{key}: {r:.6f}")
                print(f"  {name}{suffix}/Gen{_retrieve_key}_NDCG@{key}: {n:.6f}")
        return recall_dict, ndcg_dict
    return None, None


def _run_hybrid_eval(model, dataloader, device, method_config, name,
                     item2sid, item_embedding, prefix2items, add_cold_items=None):
    """Hybrid: prefix generation then dense retrieval within prefix bucket.

    When ``add_cold_items`` is provided, also computes the AddCold variant
    (prefix items ∪ cold-start items, deduplicated) and returns it as the
    3rd/4th return values.
    """
    if (method_config["evaluation_method"] == "hybrid"
            and method_config["embedding_loss_weight"] > 0
            and prefix2items is not None):
        num_candidates_list = method_config.get(
            "hybrid_num_candidates", [20, 40, 60, 80, 100]
        )
        if isinstance(num_candidates_list, int):
            num_candidates_list = [num_candidates_list]
        recall_dict, ndcg_dict, recall_cold, ndcg_cold = evaluate_prefix_then_dense(
            model, dataloader, device, item2sid,
            item_embedding=item_embedding, method_config=method_config,
            prefix2items=prefix2items, KEYS=KEYS,
            num_candidates_list=num_candidates_list,
            add_cold_items=add_cold_items,
        )
        for n in recall_dict.keys():
            for key in recall_dict[n].keys():
                r = np.mean(recall_dict[n][key])
                nd = np.mean(ndcg_dict[n][key])
                print(f"  {name}/Gen{n}_Recall@{key}: {r:.6f}")
                print(f"  {name}/Gen{n}_NDCG@{key}: {nd:.6f}")
        if add_cold_items is not None:
            for n in recall_cold.keys():
                for key in recall_cold[n].keys():
                    r = np.mean(recall_cold[n][key])
                    nd = np.mean(ndcg_cold[n][key])
                    print(f"  {name}_AddCold/Gen{n}_Recall@{key}: {r:.6f}")
                    print(f"  {name}_AddCold/Gen{n}_NDCG@{key}: {nd:.6f}")
        return recall_dict, ndcg_dict, recall_cold, ndcg_cold
    return None, None, None, None


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_standard_results(all_results, split_names):
    """Print a compact table of all standard evaluation metrics."""
    print("\n" + "=" * 70)
    print("Standard Evaluation Metrics Summary")
    print("=" * 70)
    for split in split_names:
        print(f"\n--- {split} ---")
        results = all_results.get(split, {})
        for metric_name in sorted(results.keys()):
            print(f"  {metric_name}: {results[metric_name]:.6f}")


def _gf(x):
    """Format FLOPs per query as GFLOPs (or 'n/a')."""
    return "n/a" if x is None else f"{x / 1e9:.4f} GFLOPs"


def _print_flops_results(results):
    """Print the stage-wise per-query FLOPs and latency summary."""
    g = results["generation"]
    n_lat = results.get("n_samples_latency", results.get("n_samples", "?"))
    print(
        f"  mode={results['mode']}, beams={results['num_beams']}, "
        f"steps={results['steps']}, seq_len={results['seq_len']}, "
        f"FLOPs from {results.get('n_samples_flops', results.get('n_samples','?'))} samples "
        f"({results['num_batches_measured']} batches), "
        f"latency from {n_lat} samples"
    )
    print(f"  [Beam search]  FLOPs:   {_gf(g['measured_flops_per_query'])}"
          f"  |  latency: {g['latency_ms_per_query']:.2f} ms/query")
    if g.get("analytical_flops_per_query") is not None:
        print(f"                 analytic FLOPs: {_gf(g['analytical_flops_per_query'])}")
    print("  [Re-ranking]   cost per Gen{N} (FLOPs depends on C_N, latency includes scoring):")
    for n in sorted(results["rerank"].keys()):
        r = results["rerank"][n]
        print(
            f"    Gen{n}: FLOPs {_gf(r['measured_flops_per_query'])}"
            f"  latency {r['latency_ms_per_query']:.2f} ms/query"
            f"  cand mean/p95/max = {r['candidate_count_mean']:.0f}"
            f"/{r['candidate_count_p95']:.0f}/{r['candidate_count_max']:.0f}"
        )
        print(
            f"           total  FLOPs {_gf(r['total_flops_per_query'])}"
            f"  latency {r['total_latency_ms_per_query']:.2f} ms/query"
        )


def _print_layer_results(results, n_codebook):
    """Print per-layer metrics for a single split."""
    print(f"  {'Layer':<8} {'Recall@10':<14} {'NDCG@10':<14} {'Acc':<14}")
    print(f"  {'─' * 50}")
    for k in range(1, n_codebook + 1):
        recall = results.get(f"Layer_{k}_Recall@10", 0.0)
        ndcg = results.get(f"Layer_{k}_NDCG@10", 0.0)
        acc = results.get(f"Layer_{k}_Acc", 0.0)
        print(f"  Layer {k}  {recall:<14.6f} {ndcg:<14.6f} {acc:<14.6f}")


def _print_summary_table(all_results, n_codebook):
    """Print a compact summary comparing in-set vs cold-start."""
    splits = list(all_results.keys())
    if not splits:
        print("  (no results)")
        return

    # Header
    header = f"  {'Layer':<8}"
    for split in splits:
        header += f" {split+'/Recall':<18} {split+'/NDCG':<18} {split+'/Acc':<18}"
    print(header)
    print(f"  {'─' * (8 + 54 * len(splits))}")

    for k in range(1, n_codebook + 1):
        row = f"  Layer {k} "
        for split in splits:
            r = all_results[split].get(f"Layer_{k}_Recall@10", 0.0)
            n = all_results[split].get(f"Layer_{k}_NDCG@10", 0.0)
            a = all_results[split].get(f"Layer_{k}_Acc", 0.0)
            row += f" {r:<18.6f} {n:<18.6f} {a:<18.6f}"
        print(row)

    # Bottleneck analysis
    print()
    for split in splits:
        print(f"  [{split}] Bottleneck analysis:")
        max_drop = 0.0
        bottleneck_layer = 1
        prev_acc = 1.0
        for k in range(1, n_codebook + 1):
            acc = all_results[split].get(f"Layer_{k}_Acc", 0.0)
            drop = prev_acc - acc
            if drop > max_drop:
                max_drop = drop
                bottleneck_layer = k
            prev_acc = acc
            print(
                f"    Layer {k}: Acc={acc:.4f}"
                f"  (drop from previous: {drop:.4f})"
            )
        print(
            f"    → Biggest drop at Layer {bottleneck_layer} "
            f"(drop={max_drop:.4f})"
        )
        print()


def _compute_all_test_standard(standard_results, n_in, n_cold):
    """Weighted-average in_set and cold_start metrics into all_test.

    Automatically handles variants (AddCold, NoCold, etc.): any key starting
    with ``in_set`` (e.g. ``in_set``, ``in_set_AddCold``, ``in_set_NoCold``)
    is paired with its ``cold_start*`` counterpart to produce the
    corresponding ``all_test*`` aggregate.
    """
    total = n_in + n_cold
    if total == 0:
        return {}

    # Discover all "in_set"-style prefixes (e.g. "in_set", "in_set_AddCold")
    in_prefixes = set()
    for k in standard_results:
        if k.startswith("in_set/") or (k.startswith("in_set_") and "/" in k):
            in_prefixes.add(k.rsplit("/", 1)[0])

    all_test = {}
    for in_prefix in sorted(in_prefixes):
        cold_prefix = in_prefix.replace("in_set", "cold_start")
        all_prefix = in_prefix.replace("in_set", "all_test")

        in_suffixes = {
            k[len(in_prefix) + 1:] for k in standard_results
            if k.startswith(in_prefix + "/")
        }
        cold_suffixes = {
            k[len(cold_prefix) + 1:] for k in standard_results
            if k.startswith(cold_prefix + "/")
        }
        for suffix in sorted(in_suffixes & cold_suffixes):
            v_in = standard_results[f"{in_prefix}/{suffix}"]
            v_cold = standard_results[f"{cold_prefix}/{suffix}"]
            all_test[f"{all_prefix}/{suffix}"] = (
                v_in * n_in + v_cold * n_cold
            ) / total
    return all_test


def _compute_all_test_layer(raw_in, raw_cold, n_codebook, key=10):
    """Merge per-sample arrays and recompute per-layer metrics for all_test.

    Layer_k_Acc is a conditional probability (N_correct_k / N_correct_{k-1}),
    so it must be recomputed from merged per-sample counts — it cannot be
    obtained by weighting the per-split Acc values.
    """
    if not raw_in or not raw_cold:
        return {}

    def _merged(name):
        return np.concatenate([
            np.asarray(raw_in[name], dtype=float),
            np.asarray(raw_cold[name], dtype=float),
        ])

    all_results = {}
    for k in range(1, n_codebook + 1):
        mk = _merged(f"Layer_{k}_matches")
        nk = _merged(f"Layer_{k}_ndcg")
        all_results[f"Layer_{k}_Recall@{key}"] = float(mk.mean())
        all_results[f"Layer_{k}_NDCG@{key}"] = float(nk.mean())
        if k == 1:
            acc = float(mk.mean())
        else:
            prev = _merged(f"Layer_{k - 1}_matches")
            n_prev = int(prev.sum())
            n_k = int(mk.sum())
            acc = (n_k / n_prev) if n_prev > 0 else 0.0
        all_results[f"Layer_{k}_Acc"] = acc
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:
    # Disable wandb — this is a test-only run, no training metrics to log.
    os.environ["WANDB_DISABLED"] = "true"

    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    is_steam = config["dataset"]["type"] == "steam"

    try:
        data_file, id2meta_file, item2attribute_file = preprocessing(config["dataset"])

        train_config = {
            **config["dataset"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }
        method_config = {
            **config["method"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }

        # load id split
        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )

        # Allow explicit SID file override
        sid_path_override = config["dataset"].get("sid_path", None)
        if sid_path_override:
            PATH_CONFIG.id_save_location = sid_path_override
            print(f"Using explicit SID file: {PATH_CONFIG.id_save_location}")

        # Allow explicit semantic embedding file override
        semantic_emb_override = config["dataset"].get("semantic_embedding_path", None)
        if semantic_emb_override:
            embedding_save_path = semantic_emb_override
            print(f"Using explicit semantic embedding file: {embedding_save_path}")
        else:
            embedding_save_path = PATH_CONFIG.embedding_save_path

        # Load item embeddings
        item_embedding = process_embeddings(
            config, device, id2meta_file, embedding_save_path
        )

        # The supplied SID file is an immutable test artifact. Test does not
        # retrain or re-load RQ-VAE, so it must not require the embedding that
        # was used only for SID quantization.
        if not os.path.exists(PATH_CONFIG.id_save_location):
            raise FileNotFoundError(
                f"SID file not found: {PATH_CONFIG.id_save_location}. "
                "Provide the training-time SID file via dataset.sid_path."
            )
        print(f"Loading SID file: {PATH_CONFIG.id_save_location}")

        # The T5 checkpoint and all dense metrics must use the same embedding
        # source as training. Fused embeddings are therefore required only for
        # a model trained with dataset.model_embedding=fused; semantic models
        # do not need SASRec/fused artifacts at test time.
        model_emb_source = config["dataset"].get("model_embedding", "fused")
        sasrec_cfg = config["dataset"].get("SASRec", {})
        if model_emb_source == "semantic":
            model_embedding = item_embedding
            print(f"Model uses PURE SEMANTIC embedding: {model_embedding.shape}")
        elif model_emb_source == "fused":
            if not sasrec_cfg.get("enabled", False):
                raise ValueError(
                    "dataset.model_embedding=fused requires dataset.SASRec.enabled=true."
                )
            explicit_fused = sasrec_cfg.get("fused_embedding_path", None)
            fused_path = (
                explicit_fused if explicit_fused else PATH_CONFIG.sasrec_fused_path
            )
            if not os.path.exists(fused_path):
                raise FileNotFoundError(
                    f"Fused embedding required by model_embedding=fused was not found: "
                    f"{fused_path}. Provide the exact training artifact with "
                    "dataset.SASRec.fused_embedding_path=/path/to/fused.pt, or "
                    "run test with dataset.model_embedding=semantic only if that "
                    "matches the checkpoint's training configuration."
                )
            print(f"Loading fused embeddings for model reproduction: {fused_path}")
            model_embedding = torch.load(fused_path, weights_only=False).to(device)
            print(f"Model uses FUSED embedding: {model_embedding.shape}")
        else:
            raise ValueError(
                f"Unsupported dataset.model_embedding={model_emb_source!r}; "
                "expected 'semantic' or 'fused'."
            )

        print(
            "Test reproduction config: "
            f"model_embedding={model_emb_source}, "
            f"sid_path={PATH_CONFIG.id_save_location}, "
            f"semantic_embedding_path={embedding_save_path}"
        )

        # ------------------------------------------------------------------
        # Load data splits (same as train_tiger)
        # ------------------------------------------------------------------
        output_path = config["output_path"]
        codebook_size = train_config["RQ-VAE"]["code_book_size"]
        max_items_per_seq = train_config["max_items_per_seq"]

        config_tiger = train_config["TIGER"]
        trainer_config = config_tiger["trainer"]

        unseen_val, unseen_test, seen = (
            id_split["unseen_val"],
            id_split["unseen_test"],
            id_split["seen"],
        )

        (
            training_data,
            val_data,
            test_data,
            unseen_val_data,
            unseen_test_data,
            seen_semantic_ids,
            val_unseen_semantic_ids,
            test_unseen_semantic_ids,
            max_last_semantic_ids,
            n_semantic_codebook,
            n_codebook,
            item2sid,
        ) = load_data(
            PATH_CONFIG.id_save_location,
            user_sequence,
            unseen_val,
            unseen_test,
            seen,
            model_embedding,
            method_config,
            max_length=config_tiger["n_positions"],
            codebook_size=codebook_size,
            max_items_per_seq=max_items_per_seq,
        )

        # Build all_semantic_ids and unseen_semantic_ids (same as training.py)
        all_semantic_ids = np.unique(
            np.concatenate(
                [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
                axis=0,
            ),
            axis=0,
        )
        unseen_semantic_ids = np.unique(
            np.concatenate(
                [val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0
            ),
            axis=0,
        )
        all_semantic_ids = torch.from_numpy(all_semantic_ids)
        unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

        # Build prefix2items for hybrid method
        prefix2items = None
        if method_config["evaluation_method"] == "hybrid":
            prefix2items = build_prefix2items(item2sid, method_config["prefix_depth"])
            print(
                f"[hybrid] Built prefix2items mapping: {len(prefix2items)} unique "
                f"prefixes (prefix_depth={method_config['prefix_depth']}, "
                f"n_codebook={n_codebook})"
            )

        # Build cold-start item indices (0-based) for Hybrid AddCold variant.
        # unseen_val / unseen_test are 1-based item IDs; subtract 1 for 0-based.
        all_unseen_ids = np.unique(np.concatenate([unseen_val, unseen_test]))
        cold_item_indices = torch.tensor(
            all_unseen_ids - 1, dtype=torch.long, device=device
        )
        print(
            f"[AddCold] {len(all_unseen_ids)} cold-start items available "
            f"for Hybrid AddCold evaluation"
        )

        if method_config.get("use_extra_sid", True):
            last_codebook_size = max(max_last_semantic_ids, codebook_size)
        else:
            last_codebook_size = 0

        if method_config["include_user_id"]:
            this_vocab_size = (
                2000 + codebook_size * n_semantic_codebook + last_codebook_size + 2
            )
        else:
            this_vocab_size = (
                codebook_size * n_semantic_codebook + last_codebook_size + 2
            )

        if method_config["use_id"] == "item_id":
            this_vocab_size = model_embedding.shape[0] + 2

        t5_config = config_tiger["T5"]
        model_config = T5Config(
            num_layers=t5_config["encoder_layers"],
            num_decoder_layers=t5_config["decoder_layers"],
            d_model=t5_config["d_model"],
            d_ff=t5_config["d_ff"],
            num_heads=t5_config["num_heads"],
            d_kv=t5_config["d_kv"],
            dropout_rate=t5_config["dropout_rate"],
            vocab_size=this_vocab_size,
            pad_token_id=0,
            eos_token_id=int(this_vocab_size - 1),
            decoder_start_token_id=0,
            feed_forward_proj=t5_config["feed_forward_proj"],
            n_positions=config_tiger["n_positions"],
            layer_norm_epsilon=1e-8,
            initializer_factor=t5_config["initializer_factor"],
        )

        # ------------------------------------------------------------------
        # Initialize model and load checkpoint
        # ------------------------------------------------------------------
        model = TIGER(
            config=model_config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=max_items_per_seq,
            flag_use_output_embedding=method_config["flag_use_output_embedding"],
            flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
            embedding_head_dict=method_config["embedding_head_dict"],
        ).to(device)

        # Resolve checkpoint path
        checkpoint_path = config.get("checkpoint_path", None)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(output_path, "results", "ckpt_best.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Provide the path via checkpoint_path=... or run training first."
            )

        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            # Training checkpoint (ckpt.pt format)
            model.load_state_dict(state_dict["model_state_dict"], strict=True)
        else:
            # Best model (state_dict only, from ckpt_best.pt)
            model.load_state_dict(state_dict, strict=True)
        model.eval()

        # ------------------------------------------------------------------
        # Build test dataloaders
        # ------------------------------------------------------------------
        eval_batch_size = trainer_config.get("eval_batch_size", 32)
        test_dataset = CustomDataset(test_data)
        unseen_test_dataset = CustomDataset(unseen_test_data)
        test_dataloader = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=False
        )
        unseen_test_dataloader = DataLoader(
            unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
        )

        # The retrieval corpus must match train_tiger's item_embedding argument:
        # semantic for model_embedding=semantic, fused for model_embedding=fused.
        retrieval_embedding = model_embedding
        if method_config["flag_use_output_embedding"]:
            retrieval_embedding = retrieval_embedding.to(device)

        # ==================================================================
        # Part 1: Standard Evaluation Metrics
        # ==================================================================
        print("\n" + "=" * 70)
        print("Standard Evaluation Metrics")
        print("=" * 70)
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  KEYS (Recall@K): {KEYS}")

        is_hybrid = method_config["evaluation_method"] == "hybrid"
        standard_results = {}

        if is_hybrid:
            # ---- Hybrid: prefix-then-dense (standard + AddCold) ----
            print("\n--- In-Set (seen items) ---")
            r_in, n_in, r_in_cold, n_in_cold = _run_hybrid_eval(
                model, test_dataloader, device, method_config,
                "hybrid_in_test", item2sid, retrieval_embedding, prefix2items,
                add_cold_items=cold_item_indices,
            )
            if r_in is not None:
                for n_cand in r_in.keys():
                    for key in r_in[n_cand].keys():
                        standard_results[f"in_set/Gen{n_cand}_Recall@{key}"] = float(np.mean(r_in[n_cand][key]))
                        standard_results[f"in_set/Gen{n_cand}_NDCG@{key}"] = float(np.mean(n_in[n_cand][key]))
            if r_in_cold is not None:
                for n_cand in r_in_cold.keys():
                    for key in r_in_cold[n_cand].keys():
                        standard_results[f"in_set_AddCold/Gen{n_cand}_Recall@{key}"] = float(np.mean(r_in_cold[n_cand][key]))
                        standard_results[f"in_set_AddCold/Gen{n_cand}_NDCG@{key}"] = float(np.mean(n_in_cold[n_cand][key]))

            print("\n--- Cold-Start (unseen items) ---")
            r_cold, n_cold, r_cold_cold, n_cold_cold = _run_hybrid_eval(
                model, unseen_test_dataloader, device, method_config,
                "hybrid_cold_test", item2sid, retrieval_embedding, prefix2items,
                add_cold_items=cold_item_indices,
            )
            if r_cold is not None:
                for n_cand in r_cold.keys():
                    for key in r_cold[n_cand].keys():
                        standard_results[f"cold_start/Gen{n_cand}_Recall@{key}"] = float(np.mean(r_cold[n_cand][key]))
                        standard_results[f"cold_start/Gen{n_cand}_NDCG@{key}"] = float(np.mean(n_cold[n_cand][key]))
            if r_cold_cold is not None:
                for n_cand in r_cold_cold.keys():
                    for key in r_cold_cold[n_cand].keys():
                        standard_results[f"cold_start_AddCold/Gen{n_cand}_Recall@{key}"] = float(np.mean(r_cold_cold[n_cand][key]))
                        standard_results[f"cold_start_AddCold/Gen{n_cand}_NDCG@{key}"] = float(np.mean(n_cold_cold[n_cand][key]))

        else:
            # ---- Liger / TIGER: genret + dense + unified ----
            print("\n--- Genret In-Set (seen items) ---")
            r_gen_in, n_gen_in = _run_genret_eval(
                model, test_dataloader, all_semantic_ids, device,
                method_config, "genret_in_test", item2sid,
            )
            if r_gen_in is not None:
                for key in r_gen_in.keys():
                    standard_results[f"in_set/Recall@{key}"] = float(np.mean(r_gen_in[key]))
                    standard_results[f"in_set/NDCG@{key}"] = float(np.mean(n_gen_in[key]))

            print("\n--- Genret Cold-Start (unseen items) ---")
            r_gen_cold, n_gen_cold = _run_genret_eval(
                model, unseen_test_dataloader, all_semantic_ids, device,
                method_config, "genret_cold_test", item2sid,
            )
            if r_gen_cold is not None:
                for key in r_gen_cold.keys():
                    standard_results[f"cold_start/Recall@{key}"] = float(np.mean(r_gen_cold[key]))
                    standard_results[f"cold_start/NDCG@{key}"] = float(np.mean(n_gen_cold[key]))

            # Dense retrieval
            if method_config["flag_use_output_embedding"]:
                print("\n--- Dense In-Set ---")
                r_d_in, n_d_in = _run_dense_eval(
                    model, test_dataloader, device, method_config,
                    "dense_in_test", item2sid, retrieval_embedding,
                )
                if r_d_in is not None:
                    for key in r_d_in.keys():
                        standard_results[f"in_set/Dense_Recall@{key}"] = float(np.mean(r_d_in[key]))
                        standard_results[f"in_set/Dense_NDCG@{key}"] = float(np.mean(n_d_in[key]))

                print("\n--- Dense Cold-Start ---")
                r_d_cold, n_d_cold = _run_dense_eval(
                    model, unseen_test_dataloader, device, method_config,
                    "dense_cold_test", item2sid, retrieval_embedding,
                )
                if r_d_cold is not None:
                    for key in r_d_cold.keys():
                        standard_results[f"cold_start/Dense_Recall@{key}"] = float(np.mean(r_d_cold[key]))
                        standard_results[f"cold_start/Dense_NDCG@{key}"] = float(np.mean(n_d_cold[key]))

            # Unified (generate-then-dense) — requires returned candidates from genret
            if (method_config["flag_use_output_embedding"]
                    and method_config["embedding_loss_weight"] > 0
                    and method_config["sid_loss_weight"] > 0):
                print("\n--- Unified In-Set (generate-then-dense) ---")
                # Re-run genret to collect returned candidates
                _, _, returned_cand_in, returned_embd_in = evaluate(
                    model, test_dataloader, all_semantic_ids, device,
                    method_config=method_config, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
                )
                # Standard (with cold-start SIDs in candidate pool)
                r_u_in, n_u_in = _run_unified_eval(
                    model, test_dataloader, unseen_semantic_ids, device,
                    method_config, "uni_in_test", returned_cand_in, returned_embd_in,
                    item2sid, retrieval_embedding, include_cold=True,
                )
                if r_u_in is not None:
                    for n_ret in r_u_in.keys():
                        for key in r_u_in[n_ret].keys():
                            standard_results[f"in_set/Gen{n_ret}_Recall@{key}"] = float(np.mean(r_u_in[n_ret][key]))
                            standard_results[f"in_set/Gen{n_ret}_NDCG@{key}"] = float(np.mean(n_u_in[n_ret][key]))
                # NoCold (genret candidates only, no cold-start SIDs)
                print("\n--- Unified In-Set NoCold (genret candidates only) ---")
                r_u_in_nc, n_u_in_nc = _run_unified_eval(
                    model, test_dataloader, unseen_semantic_ids, device,
                    method_config, "uni_in_test", returned_cand_in, returned_embd_in,
                    item2sid, retrieval_embedding, include_cold=False,
                )
                if r_u_in_nc is not None:
                    for n_ret in r_u_in_nc.keys():
                        for key in r_u_in_nc[n_ret].keys():
                            standard_results[f"in_set_NoCold/Gen{n_ret}_Recall@{key}"] = float(np.mean(r_u_in_nc[n_ret][key]))
                            standard_results[f"in_set_NoCold/Gen{n_ret}_NDCG@{key}"] = float(np.mean(n_u_in_nc[n_ret][key]))

                print("\n--- Unified Cold-Start (generate-then-dense) ---")
                _, _, returned_cand_cold, returned_embd_cold = evaluate(
                    model, unseen_test_dataloader, all_semantic_ids, device,
                    method_config=method_config, KEYS=KEYS, RETRIEVE_KEY=RETRIEVE_KEY,
                )
                # Standard (with cold-start SIDs)
                r_u_cold, n_u_cold = _run_unified_eval(
                    model, unseen_test_dataloader, unseen_semantic_ids, device,
                    method_config, "uni_cold_test", returned_cand_cold, returned_embd_cold,
                    item2sid, retrieval_embedding, include_cold=True,
                )
                if r_u_cold is not None:
                    for n_ret in r_u_cold.keys():
                        for key in r_u_cold[n_ret].keys():
                            standard_results[f"cold_start/Gen{n_ret}_Recall@{key}"] = float(np.mean(r_u_cold[n_ret][key]))
                            standard_results[f"cold_start/Gen{n_ret}_NDCG@{key}"] = float(np.mean(n_u_cold[n_ret][key]))
                # NoCold (genret candidates only)
                print("\n--- Unified Cold-Start NoCold (genret candidates only) ---")
                r_u_cold_nc, n_u_cold_nc = _run_unified_eval(
                    model, unseen_test_dataloader, unseen_semantic_ids, device,
                    method_config, "uni_cold_test", returned_cand_cold, returned_embd_cold,
                    item2sid, retrieval_embedding, include_cold=False,
                )
                if r_u_cold_nc is not None:
                    for n_ret in r_u_cold_nc.keys():
                        for key in r_u_cold_nc[n_ret].keys():
                            standard_results[f"cold_start_NoCold/Gen{n_ret}_Recall@{key}"] = float(np.mean(r_u_cold_nc[n_ret][key]))
                            standard_results[f"cold_start_NoCold/Gen{n_ret}_NDCG@{key}"] = float(np.mean(n_u_cold_nc[n_ret][key]))

        n_in = len(test_dataset)
        n_cold = len(unseen_test_dataset)
        all_test_standard = _compute_all_test_standard(standard_results, n_in, n_cold)
        standard_results.update(all_test_standard)

        # Build the split dict for printing (auto-detect all variant groups)
        split_names = []
        split_dict = {}
        # Standard
        for sn, prefix in [("in_set", "in_set/"), ("cold_start", "cold_start/"),
                           ("all_test", "all_test/")]:
            d = {k: v for k, v in standard_results.items() if k.startswith(prefix)}
            if d:
                split_dict[sn] = d
                split_names.append(sn)
        # AddCold (Hybrid)
        for sn, prefix in [("in_set_AddCold", "in_set_AddCold/"),
                           ("cold_start_AddCold", "cold_start_AddCold/"),
                           ("all_test_AddCold", "all_test_AddCold/")]:
            d = {k: v for k, v in standard_results.items() if k.startswith(prefix)}
            if d:
                split_dict[sn] = d
                split_names.append(sn)
        # NoCold (Liger)
        for sn, prefix in [("in_set_NoCold", "in_set_NoCold/"),
                           ("cold_start_NoCold", "cold_start_NoCold/"),
                           ("all_test_NoCold", "all_test_NoCold/")]:
            d = {k: v for k, v in standard_results.items() if k.startswith(prefix)}
            if d:
                split_dict[sn] = d
                split_names.append(sn)

        _print_standard_results(split_dict, split_names)

        # Save standard results
        standard_path = os.path.join(output_path, "standard_eval_results.json")
        with open(standard_path, "w") as f:
            json.dump(standard_results, f, indent=2)
        print(f"\nStandard results saved to: {standard_path}")
        print(f"  in_set samples: {n_in}, cold_start samples: {n_cold}, "
              f"all_test samples: {n_in + n_cold}")

        # ==================================================================
        # Part 2: Inference FLOPs Analysis (per query, stage by stage)
        # ==================================================================
        try:
            from src.flops import measure_inference_flops

            flops_num_batches = int(config.get("flops_num_batches", 3))
            print("\n" + "=" * 70)
            print("Inference FLOPs Analysis (per query)")
            print("=" * 70)

            if is_hybrid:
                num_candidates_list = method_config.get(
                    "hybrid_num_candidates", [20, 40, 60, 80, 100]
                )
                if isinstance(num_candidates_list, int):
                    num_candidates_list = [num_candidates_list]
                flops_results = measure_inference_flops(
                    model, test_dataloader, device, n_codebook, method_config,
                    item_embedding=retrieval_embedding,
                    prefix2items=prefix2items,
                    num_candidates_list=num_candidates_list,
                    num_batches=flops_num_batches,
                    # pass the full-dataset loader so C_N stats cover all queries
                    cand_stat_dataloader=test_dataloader,
                )
            else:
                flops_results = measure_inference_flops(
                    model, test_dataloader, device, n_codebook, method_config,
                    item_embedding=retrieval_embedding,
                    unseen_semantic_ids=unseen_semantic_ids,
                    num_candidates_list=RETRIEVE_KEY,
                    num_batches=flops_num_batches,
                )

            _print_flops_results(flops_results)
            flops_path = os.path.join(output_path, "flops_results.json")
            with open(flops_path, "w") as f:
                json.dump(flops_results, f, indent=2)
            print(f"\nFLOPs results saved to: {flops_path}")
        except Exception:
            print("FLOPs measurement failed (evaluation continues):")
            traceback.print_exc(file=sys.stderr)

        # ==================================================================
        # Part 3: Per-Layer SID Accuracy Analysis
        # ==================================================================
        if method_config["use_id"] != "sid":
            print(
                "\nWARNING: use_id is not 'sid' (got "
                f"'{method_config['use_id']}'). Per-layer SID accuracy "
                "analysis only applies to SID-based generation (tiger / liger)."
            )
            print("Skipping per-layer SID evaluation.")
            return

        # For the hybrid method, the model is only trained to predict the
        # first `prefix_depth` SIDs; layers beyond that are not meaningful.
        prefix_depth = method_config.get("prefix_depth", 0)
        if prefix_depth > 0 and prefix_depth < n_codebook:
            print(
                f"\nNOTE: prefix_depth={prefix_depth} < n_codebook={n_codebook}. "
                f"Only the first {prefix_depth} layers have meaningful results."
            )

        print("\n" + "=" * 70)
        print("Per-Layer SID Accuracy Analysis")
        print("=" * 70)
        print(f"  n_codebook (SID depth): {n_codebook}")
        print(f"  beam width / num_return: 10")
        print(f"  checkpoint: {checkpoint_path}")
        print()

        all_results = {}
        raw_in, raw_cold = {}, {}

        # --- In-set (seen items) ---
        print("\n--- In-Set (seen items) ---")
        if len(test_dataloader) > 0:
            results_in, raw_in = evaluate_layer_accuracy(
                model,
                test_dataloader,
                device,
                method_config,
                KEYS=[10],
                num_beams=10,
            )
            all_results["in_set"] = results_in
            _print_layer_results(results_in, n_codebook)
        else:
            print("  (empty test set, skipped)")

        # --- Cold-start (unseen items) ---
        print("\n--- Cold-Start (unseen items) ---")
        if len(unseen_test_dataloader) > 0:
            results_cold, raw_cold = evaluate_layer_accuracy(
                model,
                unseen_test_dataloader,
                device,
                method_config,
                KEYS=[10],
                num_beams=10,
            )
            all_results["cold_start"] = results_cold
            _print_layer_results(results_cold, n_codebook)
        else:
            print("  (empty cold-start set, skipped)")

        # --- All-test (whole test set) ---
        print("\n--- All-Test (in_set + cold_start merged) ---")
        all_test_layer = _compute_all_test_layer(raw_in, raw_cold, n_codebook, key=KEYS[0])
        if all_test_layer:
            all_results["all_test"] = all_test_layer
            _print_layer_results(all_test_layer, n_codebook)
        else:
            print("  (no data to merge, skipped)")

        # ------------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        _print_summary_table(all_results, n_codebook)

        # Save layer accuracy results
        results_path = os.path.join(output_path, "layer_accuracy_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nLayer accuracy results saved to: {results_path}")

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
