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
            fusion_tag = "_fused" if sasrec_cfg.get("enabled", False) else ""
            self.id_save_location = os.path.join(
                self.rqvae_save_dir,
                id_filename + f"{fusion_tag}_{config['seed']}.pkl",
            )

            # Path for SASRec fused embeddings (semantic + CF)
            self.sasrec_save_dir = "./ID_generation/sasrec/ckpt/"
            os.makedirs(self.sasrec_save_dir, exist_ok=True)
            self.sasrec_fused_path = os.path.join(
                self.sasrec_save_dir, id_filename + f"_fused_{config['seed']}.pt"
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
            fusion_tag = "_fused" if sasrec_cfg.get("enabled", False) else ""

            self.sasrec_fused_path = os.path.join(
                run_dir, f"sasrec_fused{fusion_tag}_{config['seed']}.pt"
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
    config["logging"]["project"] = "hybird_gr"
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

        # load item embedding
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        # --- Bold fusion: train SASRec to fuse semantic + CF embeddings ---
        # The fused embedding replaces item_embedding *only* for RQ-VAE
        # quantization (train_sid). train_tiger still uses the original
        # semantic embedding — "之后的都先不动".
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

                # L2-normalize semantic embeddings before fusion.
                # After StandardScaler, raw semantic norm ≈ sqrt(768) ≈ 27.7,
                # which makes BPR dot products explode (~768) → sigmoid
                # saturates → CF gradients explode → loss diverges.
                # Normalizing to unit norm keeps dot-product scale ~O(1).
                sem_normed = item_embedding / item_embedding.norm(
                    dim=-1, keepdim=True).clamp(min=1e-8)
                # Build semantic embeddings with padding row at index 0
                sem_with_pad = torch.cat(
                    [torch.zeros(1, item_embedding.shape[1], device=device),
                     sem_normed],
                    dim=0,
                )  # [num_items+1, dim]

                print(f"\nTraining SASRec for bold fusion (semantic + CF)...")
                print(f"  num_items={num_items}, hidden_units={item_embedding.shape[1]}")

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
                    epochs=sasrec_cfg.get("epochs", 200),
                    lr=sasrec_cfg.get("lr", 0.001),
                    batch_size=sasrec_cfg.get("batch_size", 128),
                    eval_steps=sasrec_cfg.get("eval_steps", 20),
                    patience=sasrec_cfg.get("patience", 5),
                    eval_metric=sasrec_cfg.get("eval_metric", "metric"),
                    seed=config["seed"],
                    semantic_embeddings=sem_with_pad,
                    writer=sasrec_writer,
                    eval_sequences=user_sequence,
                )
                sasrec_writer.finish()
                # Load fused embeddings for RQ-VAE quantization
                # No StandardScaler — the fused embedding (L2-normalized
                # semantic + CF) has its own structure that StandardScaler
                # would distort. RQ-VAE's encoder has LayerNorm and can
                # adapt to the input scale on its own.
                fused_embedding = torch.load(fused_path, weights_only=False).to(device)

            print(f"Using fused embeddings for quantization: {fused_embedding.shape}")
        else:
            fused_embedding = item_embedding

        train_sid(
            config, device, fused_embedding, id_split, PATH_CONFIG.id_save_location
        )

        train_tiger(
            config,
            train_config,
            method_config,
            id_split,
            user_sequence,
            item_embedding,
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
