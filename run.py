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
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from omegaconf import DictConfig
from src.training import train_tiger, train_tiger_residual

from utils import set_seed


class set_dir:
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
    config["logging"]["project"] = "liger"
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

        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
        )

        # --- Residual Decoder Experiment ---
        if method_config.get("use_residual_decoder", False):
            # Load RQ-VAE codebook weights for TIGER_Residual
            # We only need the codebook embedding matrices, NOT the entire RQ-VAE model.
            # Each codebook is [codebook_size, latent_size] — e.g. [256, 128], tiny (~384KB total).
            rqvae_codebook_weights = None
            codebook_save_path = os.path.join(
                PATH_CONFIG.rqvae_save_dir,
                f"codebook_weights_{config['seed']}.pt",
            )
            # Also check alongside the SID .pkl file
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
                # Fallback: re-train RQ-VAE, extract & save codebook weights
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

                # Extract codebook weights only — no need to keep the full model
                rqvae_codebook_weights = [
                    cb.weight.data.clone()
                    for cb in rqvae_model.quantizer.codebooks
                ]
                # Save for future runs (only codebook weights, ~384KB)
                os.makedirs(PATH_CONFIG.rqvae_save_dir, exist_ok=True)
                torch.save(rqvae_codebook_weights, codebook_save_path)
                print(f"✓ Saved codebook weights to {codebook_save_path}")

            train_tiger_residual(
                config,
                train_config,
                method_config,
                id_split,
                user_sequence,
                item_embedding,
                PATH_CONFIG.id_save_location,
                device=device,
                rqvae_codebook_weights=rqvae_codebook_weights,
            )
        else:
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
