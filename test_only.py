"""
test_only.py — Evaluate an existing model checkpoint on the test set WITHOUT training.

Usage:
    python test_only.py \
        dataset=amazon \
        dataset.name=Beauty \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        checkpoint_path="results/tiger/Amazon_Beauty/experiment_seed_42/results/ckpt_best.pt" \
        experiment_id="test_Beauty"

For residual decoder models:
    python test_only.py \
        dataset=amazon \
        dataset.name=Beauty \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        method.use_residual_decoder=True \
        method.codebook_loss_weight=1.0 \
        checkpoint_path="results/tiger/Amazon_Beauty/residual_Beauty_seed_42/results/ckpt_best.pt" \
        experiment_id="test_residual_Beauty"

Key arguments:
    checkpoint_path  — Path to the saved model state_dict (.pt file).
                       If not provided, defaults to:
                       {output_path}/results/ckpt_best.pt
                       (same path used during training).
    test_set         — Which test set to evaluate on: "test", "unseen_test", or "both".
                       Default: "both".
"""

import os
import sys
import traceback

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from src.training import train_tiger, train_tiger_residual
from src.evaluation import (
    evaluate,
    evaluate_residual,
    evaluate_dense_ids,
    evaluate_dense_sids,
    generate_then_dense,
    get_target_embed,
    model_forward,
    model_forward_residual,
)
from src.load_data import load_data
from src.tiger import TIGER
from src.tiger_residual import TIGER_Residual
from torch.utils.data import DataLoader
from transformers import T5Config

from utils import set_seed, CustomDataset, setup_logging


class set_dir:
    """Same directory setup as run.py — ensures all paths exist."""
    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        if config["test_method"] in ["tiger", "liger"]:
            self.rqvae_save_dir = "./ID_generation/ID/"
            os.makedirs(self.rqvae_save_dir, exist_ok=True)

            id_filename = (
                f"{config['dataset']['name']}_{config['dataset']['content_model']}"
            )
            self.id_save_location = os.path.join(
                self.rqvae_save_dir, id_filename + f"_{config['seed']}.pkl"
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
        return config


def _build_model_tiger(method_config, model_config, n_semantic_codebook,
                        max_items_per_seq, device):
    """Instantiate TIGER model (standard, no residual decoder)."""
    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)
    return model


def _build_model_residual(method_config, model_config, n_semantic_codebook,
                           max_items_per_seq, codebook_size, latent_size,
                           rqvae_codebook_weights, device):
    """Instantiate TIGER_Residual model."""
    codebook_loss_weight = method_config.get("codebook_loss_weight", 1.0)
    num_residual_levels = method_config.get(
        "num_residual_levels", n_semantic_codebook - 1
    )
    soft_label_K = method_config.get("soft_label_K", 0)
    soft_label_temperature = method_config.get("soft_label_temperature", 1.0)
    cumulative_residual_loss_weight = method_config.get("cumulative_residual_loss_weight", 0.0)
    cumulative_residual_loss_type = method_config.get("cumulative_residual_loss_type", "mse")

    model = TIGER_Residual(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
        rqvae_codebook_weights=rqvae_codebook_weights,
        codebook_size=codebook_size,
        latent_size=latent_size,
        codebook_loss_weight=codebook_loss_weight,
        num_residual_levels=num_residual_levels,
        soft_label_K=soft_label_K,
        soft_label_temperature=soft_label_temperature,
        cumulative_residual_loss_weight=cumulative_residual_loss_weight,
        cumulative_residual_loss_type=cumulative_residual_loss_type,
    ).to(device)
    return model


def _evaluate_helper(
    model,
    device,
    test_dataloader_dict,
    unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    use_residual_decoder,
    KEYS=[5, 10, 20],
    RETRIEVE_KEY=[10],
):
    """
    Unified evaluation helper that supports both TIGER and TIGER_Residual models.
    Returns logs dict with all metrics.
    """
    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
            if use_residual_decoder:
                recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate_residual(
                    model,
                    dataloader,
                    all_semantic_ids,
                    device,
                    method_config=method_config,
                    KEYS=KEYS,
                    RETRIEVE_KEY=RETRIEVE_KEY,
                )
            else:
                recall_dict, ndcg_dict, returned_cand, returned_embd = evaluate(
                    model,
                    dataloader,
                    all_semantic_ids,
                    device,
                    method_config=method_config,
                    KEYS=KEYS,
                    RETRIEVE_KEY=RETRIEVE_KEY,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        else:
            returned_cand = None
            returned_embd = None
        return logs, returned_cand, returned_embd

    def _dense_evaluate(logs, dataloader, name):
        if method_config["embedding_loss_weight"] > 0:
            if method_config["use_id"] == "item_id":
                recall_dict, ndcg_dict = evaluate_dense_ids(
                    model, dataloader, device, item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config, KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model, dataloader, device, item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config, KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _unified_evaluate(logs, dataloader, returned_cand, returned_embd, name):
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            recall_dict, ndcg_dict = generate_then_dense(
                model, dataloader, unseen_semantic_ids, device,
                method_config=method_config,
                returned_cand=returned_cand,
                returned_embd=returned_embd,
                item2sid=item2sid,
                item_embedding=item_embedding,
                KEYS=KEYS,
                RETRIEVE_KEY=RETRIEVE_KEY,
            )
            for _retrieve_key in recall_dict.keys():
                for key in recall_dict[_retrieve_key].keys():
                    logs = add_log(
                        logs, recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}", name,
                    )
                    logs = add_log(
                        logs, ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}", name,
                    )
        return logs

    # ── In-set (seen items) evaluation ──
    logs, returned_cand_in, returned_embd_in = _evaluate(
        logs, test_dataloader_dict["in_set"], f"genret_in_test"
    )

    # ── Cold-start (unseen items) evaluation ──
    logs, returned_cand_cold, returned_embd_cold = _evaluate(
        logs, test_dataloader_dict["cold_start"], f"genret_cold_test"
    )

    # ── Dense retrieval evaluation ──
    if method_config["flag_use_output_embedding"]:
        logs = _dense_evaluate(
            logs, test_dataloader_dict["in_set_embd"], f"dense_in_test"
        )
        logs = _dense_evaluate(
            logs, test_dataloader_dict["cold_start_embd"], f"dense_cold_test"
        )

    # ── Unified (generate-then-dense) evaluation ──
    if method_config["flag_use_output_embedding"] and method_config["sid_loss_weight"] > 0:
        logs = _unified_evaluate(
            logs, test_dataloader_dict["in_set"],
            returned_cand_in, returned_embd_in, f"uni_in_test",
        )
        logs = _unified_evaluate(
            logs, test_dataloader_dict["cold_start"],
            returned_cand_cold, returned_embd_cold, f"uni_cold_test",
        )

    return logs


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:
    """
    Evaluate a saved model checkpoint on the test set.
    No training is performed — only data loading, model construction,
    weight loading, and evaluation.
    """
    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    config["logging"]["project"] = "liger"
    is_steam = config["dataset"]["type"] == "steam"

    # ── Override wandb to offline/disabled mode for test-only runs ──
    # We don't want test runs to pollute the training wandb dashboard.
    config["logging"]["mode"] = "offline"

    try:
        # ── Step 1: Data preprocessing (same as run.py) ──
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

        # Load id split
        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )

        # Load item embedding
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        # Train SID (RQ-VAE) — this is needed to get semantic IDs even for test-only.
        # In practice, the SID file should already exist from a previous training run.
        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
        )

        use_residual_decoder = method_config.get("use_residual_decoder", False)

        # ── Step 2: Load RQ-VAE codebook weights (for residual decoder) ──
        rqvae_codebook_weights = None
        codebook_size = config["dataset"]["RQ-VAE"]["code_book_size"]

        if use_residual_decoder:
            codebook_save_path = os.path.join(
                PATH_CONFIG.rqvae_save_dir,
                f"codebook_weights_{config['seed']}.pt",
            )
            alt_paths = [
                codebook_save_path,
                os.path.join(
                    os.path.dirname(PATH_CONFIG.id_save_location),
                    f"codebook_weights_{config['seed']}.pt",
                ),
            ]

            loaded = False
            for try_path in alt_paths:
                if os.path.exists(try_path):
                    try:
                        rqvae_codebook_weights = torch.load(try_path, map_location=device)
                        print(f"✓ Loaded codebook weights from {try_path}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Warning: Could not load codebook weights from {try_path}: {e}")
                        rqvae_codebook_weights = None

            if not loaded:
                print(f"⚠️ Codebook weights (.pt) not found at any of: {alt_paths}.")
                print("→ Re-training RQ-VAE to obtain codebook weights...")
                from ID_generation.rqvae.rqvae import RQVAE
                from ID_generation.train_rqvae import train_rqvae as _train_rqvae_inner
                from utils import setup_logging as _setup_logging

                rqvae_cfg = config["dataset"]["RQ-VAE"]
                rqvae_model = RQVAE(
                    input_size=rqvae_cfg["input_dim"],
                    hidden_sizes=rqvae_cfg["hidden_dim"],
                    latent_size=rqvae_cfg["latent_dim"],
                    num_levels=rqvae_cfg["num_layers"],
                    codebook_size=rqvae_cfg["code_book_size"],
                    dropout=rqvae_cfg["dropout"],
                    latent_loss_weight=rqvae_cfg["beta"],
                )
                _writer = _setup_logging(config)
                _train_rqvae_inner(
                    rqvae_model,
                    item_embedding[id_split["seen"] - 1],
                    device,
                    _writer,
                    rqvae_cfg,
                )
                _writer.finish()

                rqvae_codebook_weights = [
                    cb.weight.data.clone()
                    for cb in rqvae_model.quantizer.codebooks
                ]
                os.makedirs(PATH_CONFIG.rqvae_save_dir, exist_ok=True)
                torch.save(rqvae_codebook_weights, codebook_save_path)
                print(f"✓ Saved codebook weights to {codebook_save_path}")

        # ── Step 3: Load data ──
        unseen_val, unseen_test, seen = (
            id_split["unseen_val"],
            id_split["unseen_test"],
            id_split["seen"],
        )

        max_items_per_seq = config["dataset"]["max_items_per_seq"]

        (training_data, val_data, test_data, unseen_val_data, unseen_test_data,
         seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids,
         max_last_semantic_ids, n_semantic_codebook, n_codebook, item2sid,
         ) = load_data(
            PATH_CONFIG.id_save_location,
            user_sequence,
            unseen_val,
            unseen_test,
            seen,
            item_embedding,
            method_config,
            max_length=config["dataset"]["TIGER"]["n_positions"],
            codebook_size=codebook_size,
            max_items_per_seq=max_items_per_seq,
        )

        all_semantic_ids = np.unique(
            np.concatenate(
                [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
                axis=0,
            ),
            axis=0,
        )
        unseen_semantic_ids = np.unique(
            np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
            axis=0,
        )

        if method_config["flag_use_output_embedding"]:
            item_embedding = item_embedding.to(device)

        # ── Step 4: Build dataloaders ──
        eval_batch_size = config["dataset"]["TIGER"]["trainer"]["eval_batch_size"]

        test_dataset = CustomDataset(test_data)
        unseen_test_dataset = CustomDataset(unseen_test_data)

        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        test_dataloader_embedding = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        unseen_test_dataloader = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)
        unseen_test_dataloader_embedding = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)

        test_dataloader_dict = {
            "in_set": test_dataloader,
            "in_set_embd": test_dataloader_embedding,
            "cold_start": unseen_test_dataloader,
            "cold_start_embd": unseen_test_dataloader_embedding,
        }

        # ── Step 5: Build model ──
        seen_semantic_ids_t = torch.from_numpy(seen_semantic_ids)
        val_unseen_semantic_ids_t = torch.from_numpy(val_unseen_semantic_ids)
        test_unseen_semantic_ids_t = torch.from_numpy(test_unseen_semantic_ids)
        all_semantic_ids_t = torch.from_numpy(all_semantic_ids)
        unseen_semantic_ids_t = torch.from_numpy(unseen_semantic_ids)

        last_codebook_size = max(max_last_semantic_ids, codebook_size)
        if method_config["include_user_id"]:
            this_vocab_size = 2000 + codebook_size * n_semantic_codebook + last_codebook_size + 2
        else:
            this_vocab_size = codebook_size * n_semantic_codebook + last_codebook_size + 2
        if method_config["use_id"] == "item_id":
            this_vocab_size = item_embedding.shape[0] + 2

        t5_config = config["dataset"]["TIGER"]["T5"]
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
            n_positions=config["dataset"]["TIGER"]["n_positions"],
            layer_norm_epsilon=1e-8,
            initializer_factor=t5_config["initializer_factor"],
        )

        # Determine latent_size for residual decoder
        latent_size = t5_config["d_model"]
        if use_residual_decoder and rqvae_codebook_weights is not None:
            latent_size = rqvae_codebook_weights[0].shape[-1]

        if use_residual_decoder:
            model = _build_model_residual(
                method_config, model_config, n_semantic_codebook,
                max_items_per_seq, codebook_size, latent_size,
                rqvae_codebook_weights, device,
            )
        else:
            model = _build_model_tiger(
                method_config, model_config, n_semantic_codebook,
                max_items_per_seq, device,
            )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        # ── Step 6: Load checkpoint ──
        checkpoint_path = OmegaConf.to_container(config).get("checkpoint_path", None)
        if checkpoint_path is None or checkpoint_path == "None":
            # Default: use the same path as training's best checkpoint
            checkpoint_path = os.path.join(config["output_path"], "results", "ckpt_best.pt")

        if not os.path.exists(checkpoint_path):
            print(f"\n❌ Checkpoint not found: {checkpoint_path}")
            print("   Possible solutions:")
            print("   1. Provide --checkpoint_path=<path> explicitly")
            print("   2. Make sure you've run training first and the best checkpoint exists")
            print(f"   3. Expected default path: {config['output_path']}/results/ckpt_best.pt")
            sys.exit(1)

        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle both raw state_dict and training_state dict
        if "model_state_dict" in state_dict:
            # This is a full training state (from ckpt.pt)
            state_dict = state_dict["model_state_dict"]
            print("   (Loaded model_state_dict from training checkpoint)")

        # strict=False for residual decoder: codebook_emb_* buffers are registered
        # but may not be in the checkpoint if they were registered_buffers at init.
        # They should be present since they're part of the model, but allow flexibility.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   ⚠️ Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"   ⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        if not missing and not unexpected:
            print("   ✓ All keys matched perfectly")

        # ── Step 7: Run evaluation ──
        # Determine RETRIEVE_KEY based on model type
        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            RETRIEVE_KEY = [20, 40, 60, 80, 100]
        else:
            RETRIEVE_KEY = [10]

        KEYS = [5, 10, 20]  # Recall@5, Recall@10, Recall@20

        print(f"\n🧪 Evaluating on TEST set...")
        print(f"   Model type: {'TIGER_Residual' if use_residual_decoder else 'TIGER'}")
        print(f"   KEYS (Recall/NDCG@K): {KEYS}")
        print(f"   RETRIEVE_KEY: {RETRIEVE_KEY}")
        print(f"   use_residual_decoder: {use_residual_decoder}")
        print(f"   resid_beams: {method_config.get('resid_beams', None)}")
        print(f"   resid_score_weight: {method_config.get('resid_score_weight', 1.0)}")

        logs = _evaluate_helper(
            model,
            device,
            test_dataloader_dict,
            unseen_semantic_ids_t,
            all_semantic_ids_t,
            item2sid,
            item_embedding,
            method_config,
            use_residual_decoder=use_residual_decoder,
            KEYS=KEYS,
            RETRIEVE_KEY=RETRIEVE_KEY,
        )

        # ── Step 8: Print results ──
        print("\n" + "=" * 60)
        print("📊 TEST SET RESULTS")
        print("=" * 60)

        # Group and print results nicely
        for key in sorted(logs.keys()):
            val = logs[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            print(f"  {key}: {val:.4f}")

        print("=" * 60)

        # ── Step 9: Save results to file ──
        results_dir = os.path.join(config["output_path"], "test_results")
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, "metrics.txt")
        with open(results_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TEST SET RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Model type: {'TIGER_Residual' if use_residual_decoder else 'TIGER'}\n")
            f.write(f"KEYS: {KEYS}\n")
            f.write(f"RETRIEVE_KEY: {RETRIEVE_KEY}\n")
            f.write("\n")
            for key in sorted(logs.keys()):
                val = logs[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                f.write(f"{key}: {val:.4f}\n")
            f.write("=" * 60 + "\n")

        print(f"\n✅ Results saved to: {results_file}")

        # Also save as JSON for easy programmatic access
        import json
        results_json = os.path.join(results_dir, "metrics.json")
        json_results = {}
        for key in sorted(logs.keys()):
            val = logs[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            json_results[key] = val
        with open(results_json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ Results (JSON) saved to: {results_json}")

        # Optional: log to wandb (offline mode)
        try:
            writer = setup_logging(config)
            writer.log(logs)
            writer.finish()
            print(f"✅ Results logged to wandb (offline)")
        except Exception as e:
            print(f"⚠️ Could not log to wandb: {e}")

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()