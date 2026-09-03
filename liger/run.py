# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import traceback

import hydra

import torch
from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.sasrec import train_sasrec
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from omegaconf import DictConfig
from src.training import train_tiger

from utils import set_seed


class set_dir:
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

            # When SASRec bold fusion is enabled, the RQ-VAE quantizes the
            # fused (semantic + CF) embedding instead of the raw semantic one.
            # Use a distinct SID filename to avoid reusing the old SID cache.
            sasrec_cfg = config["dataset"].get("SASRec", {})
            fusion_tag = ""
            if sasrec_cfg.get("enabled", False):
                fm = sasrec_cfg.get("fusion_mode", "fused")
                fusion_tag = "_fused" if fm == "fused" else "_cf"
            self.id_save_location = os.path.join(
                self.rqvae_save_dir,
                id_filename + f"{fusion_tag}_{config['seed']}.pkl",
            )

            # Path for SASRec fused embeddings (semantic + CF)
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

        # If force_rerun, redirect SASRec weights & SID to this run's
        # output_path so old caches are never hit — always retrain from scratch.
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
            print(f"[force_rerun] SASRec weights → {self.sasrec_fused_path}")
            print(f"[force_rerun] SID file      → {self.id_save_location}")

        return config


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:

    # print(config)
    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    # Use the wandb project name from config (configs/logging/wandb.yaml),
    # falling back to "hybird_gr" if unset. Override by setting
    # `logging.project` in your run command or config.
    config["logging"]["project"] = config["logging"].get("project", "hybird_gr")
    is_steam = config["dataset"]["type"] == "steam"

    try:
        data_file, id2meta_file, item2attribute_file = preprocessing(config["dataset"])
        # id2meta_file: the file that save item_id to meta info, we will later use it for sentence T5 embedding generation
        # data_file: the file that save the user-item interactions.

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

        # Allow explicitly specifying a pre-quantized SID file — skips
        # RQ-VAE training entirely (train_sid returns early if file exists).
        sid_path_override = config["dataset"].get("sid_path", None)
        if sid_path_override:
            PATH_CONFIG.id_save_location = sid_path_override
            print(f"Using explicit SID file: {PATH_CONFIG.id_save_location}")

        # load item embedding
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        # The fused embedding (semantic + CF) is used for both RQ-VAE
        # quantization (train_sid) and model training / dense retrieval
        # (train_tiger). This keeps SID generation and dense retrieval
        # consistent — both leverage the collaborative signal.
        sasrec_cfg = config["dataset"].get("SASRec", {})
        if sasrec_cfg.get("enabled", False):
            num_items = item_embedding.shape[0]

            # Allow explicitly specifying a fused embedding file — skips
            # SASRec training entirely and loads the given path directly.
            explicit_fused = sasrec_cfg.get("fused_embedding_path", None)
            fused_path = explicit_fused if explicit_fused else PATH_CONFIG.sasrec_fused_path

            if os.path.exists(fused_path):
                print(f"Loading cached fused embeddings from {fused_path}")
                fused_embedding = torch.load(fused_path, weights_only=False).to(device)
            else:
                # Build training-only sequences: exclude val (seq[-2]) and
                # test (seq[-1]) items to prevent data leakage.
                sasrec_train_seqs = [seq[:-2] for seq in user_sequence if len(seq) > 2]

                # Fusion mode: how SASRec combines semantic and CF signals.
                #   "fused" (default) — semantic + CF (bold fusion)
                #   "cf_only"         — pure CF, no semantic input; the model
                #                        learns collaborative embeddings from
                #                        scratch, like original SASRec.
                fusion_mode = sasrec_cfg.get("fusion_mode", "fused")

                # Fusion method: how the semantic embedding is transformed
                # before being added to CF (only relevant when fusion_mode="fused").
                #   "normalize" (default) — L2-normalize semantic, then add CF.
                #                          Requires normalize_semantic for scale control.
                #   "mlp"               — Pass semantic through a learnable Linear
                #                          layer, then add CF. No pre-normalization
                #                          needed; the projection learns the optimal
                #                          scale/rotation. hidden_units can differ
                #                          from the semantic embedding dimension.
                fusion_method = sasrec_cfg.get("fusion_method", "normalize")

                # L2-normalize semantic embeddings before fusion (only used
                # when fusion_method="normalize"). When using dot-product
                # similarity, raw semantic norm ≈ 27.7 makes dot products
                # explode → loss diverges. Normalizing to unit norm keeps
                # dot-product scale ~O(1). With fusion_method="mlp", the
                # learnable projection handles scaling, so normalization is
                # skipped.
                normalize_semantic = sasrec_cfg.get("normalize_semantic", True)

                if fusion_mode == "fused":
                    if fusion_method == "mlp":
                        # MLP mode: pass raw semantic embeddings (no normalization).
                        # The learnable Linear layer handles scaling/rotation.
                        sem_normed = item_embedding
                        print(f"  fusion_method=mlp: raw semantic → Linear → + CF")
                    elif normalize_semantic:
                        sem_normed = item_embedding / item_embedding.norm(
                            dim=-1, keepdim=True).clamp(min=1e-8)
                    else:
                        sem_normed = item_embedding
                    # Build semantic embeddings with padding row at index 0
                    sem_with_pad = torch.cat(
                        [torch.zeros(1, item_embedding.shape[1], device=device),
                         sem_normed],
                        dim=0,
                    )  # [num_items+1, dim]
                else:
                    # cf_only: no semantic embeddings
                    sem_with_pad = None

                print(f"\nTraining SASRec [{fusion_mode}] ...")
                print(f"  num_items={num_items}, hidden_units={item_embedding.shape[1]}")
                if sem_with_pad is not None:
                    print(f"  fusion_method={fusion_method}, normalize_semantic={normalize_semantic}")

                # Set up a wandb run for SASRec training
                from utils import setup_logging
                sasrec_writer = setup_logging(config)

                train_sasrec(
                    user_sequences=sasrec_train_seqs,
                    num_items=num_items,
                    device=device,
                    save_path=fused_path,
                    num_heads=sasrec_cfg.get("num_heads", 2),
                    num_blocks=sasrec_cfg.get("num_blocks", 2),
                    max_len=sasrec_cfg.get("max_len", 50),
                    dropout=sasrec_cfg.get("dropout", 0.2),
                    # In fused mode, hidden_units is auto-overridden inside
                    # train_sasrec to match semantic_embeddings.shape[1].
                    # In cf_only mode, default to the semantic dim so the
                    # output CF embeddings are compatible with RQ-VAE's
                    # input_dim. Override via SASRec.hidden_units in config.
                    hidden_units=sasrec_cfg.get("hidden_units", item_embedding.shape[1]),
                    epochs=sasrec_cfg.get("epochs", 200),
                    lr=sasrec_cfg.get("lr", 0.001),
                    batch_size=sasrec_cfg.get("batch_size", 128),
                    eval_steps=sasrec_cfg.get("eval_steps", 20),
                    patience=sasrec_cfg.get("patience", 5),
                    eval_metric=sasrec_cfg.get("eval_metric", "metric"),
                    seed=config["seed"],
                    semantic_embeddings=sem_with_pad,
                    similarity_metric=sasrec_cfg.get("similarity_metric", "dot"),
                    temperature=sasrec_cfg.get("temperature", 1.0),
                    writer=sasrec_writer,
                    eval_sequences=user_sequence,
                    fusion_method=fusion_method,
                )
                sasrec_writer.finish()
                # Load fused embeddings for RQ-VAE quantization
                # No StandardScaler — the fused embedding (L2-normalized
                # semantic + CF) has its own structure that StandardScaler
                # would distort. RQ-VAE's encoder has LayerNorm and can
                # adapt to the input scale on its own.
                fused_embedding = torch.load(fused_path, weights_only=False).to(device)

            print(f"Using fused embeddings for quantization: {fused_embedding.shape}")

            # Whether to stop the entire pipeline after SASRec training.
            # Useful when you only want to inspect SASRec training quality
            # (Recall@10/NDCG@10, loss curves) without proceeding to the
            # expensive RQ-VAE quantization and TIGER training stages.
            stop_after_sasrec = sasrec_cfg.get("stop_after_sasrec", False)
            if stop_after_sasrec:
                print("\n" + "=" * 60)
                print("stop_after_sasrec=true — pipeline halted after SASRec.")
                print(f"  Fused embeddings saved to: {fused_path}")
                print(f"  Shape: {fused_embedding.shape}")
                print("=" * 60)
                return

        else:
            fused_embedding = item_embedding

        train_sid(
            config, device, fused_embedding, id_split, PATH_CONFIG.id_save_location
        )

        # Decide which embedding the T5 model uses for input, label, and dense
        # retrieval. This is independent of what RQ-VAE used for SID
        # quantization (always fused_embedding above), allowing the
        # combination "SID from fused, model from pure semantic".
        #   "fused"    (default) — use fused embedding (semantic + CF),
        #                           consistent with SID quantization.
        #   "semantic"          — use pure semantic embedding; model sees
        #                           clean semantic signal while SID still
        #                           benefits from CF fusion. May perform
        #                           better when CF adds noise to dense retrieval.
        model_emb_source = config["dataset"].get("model_embedding", "fused")
        if model_emb_source == "semantic":
            model_embedding = item_embedding
            print(f"Model (train_tiger) uses PURE SEMANTIC embedding: {model_embedding.shape}")
        else:
            model_embedding = fused_embedding
            print(f"Model (train_tiger) uses FUSED embedding: {model_embedding.shape}")

        train_tiger(
            config,
            train_config,
            method_config,
            id_split,
            user_sequence,
            model_embedding,
            PATH_CONFIG.id_save_location,
            device=device,
        )

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
