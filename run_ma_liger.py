"""
MA-LIGER: Memory-Augmented LIGER — Entry Script
===================================================

This script runs Stage 2 of MA-LIGER training:
  1. Loads data and pre-trained RQ-VAE codebook weights (same as run.py)
  2. Calls train_ma_liger() which:
     - Loads Stage 1 LIGER checkpoint
     - Initializes Prototype Memory via K-Means
     - Jointly fine-tunes all parameters

Usage:
  # Stage 1: Train standard LIGER first (using run.py)
  python run.py dataset.name=Beauty method.use_residual_decoder=true method.flag_use_output_embedding=true method.embedding_loss_weight=1.0

  # Stage 2: Train MA-LIGER (this script)
  python run_ma_liger.py dataset.name=Beauty \
    method.prototype.stage1_checkpoint=./results/liger/Amazon_Bauty/<exp_id>/results/ckpt_best.pt \
    method.prototype.K=256 method.prototype.p=3 method.prototype.M=10

Prerequisites:
  - Stage 1 LIGER checkpoint must exist at the specified path
  - RQ-VAE codebook weights must exist (same as run.py requirements)
  - Data must be preprocessed (same as run.py)
"""

import os
import sys

import hydra
import torch
from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.train_letter import train as train_letter_sid
from ID_generation.utils import process_data_split, process_embeddings
from omegaconf import DictConfig
from src.training_ma import train_ma_liger

from utils import set_seed


class set_dir:
    """Same as run.py's set_dir — manages directory structure for data/IDs/results."""

    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        if config["test_method"] in ["tiger", "liger", "letter", "ma_liger"]:
            self.rqvae_save_dir = "./ID_generation/ID/"
            os.makedirs(self.rqvae_save_dir, exist_ok=True)

            self.id_filename = (
                f"{config['dataset']['name']}_{config['dataset']['content_model']}"
            )
            if config["test_method"] == "letter":
                self.id_save_location = os.path.join(
                    self.rqvae_save_dir, self.id_filename + f"_letter_{config['seed']}.pkl"
                )
            else:
                self.id_save_location = os.path.join(
                    self.rqvae_save_dir, self.id_filename + f"_{config['seed']}.pkl"
                )

        self.embedding_save_name = f"_{config['dataset']['content_model']}"
        self.embedding_save_path = os.path.join(
            self.directory_processed, self.id_filename + "_embeddings.pt"
        )

        self.result_save_dir = f"./results/ma_liger/"
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


def _load_codebook_weights(config, device, PATH_CONFIG, item_embedding, id_split):
    """Load RQ-VAE codebook weights (shared utility for all residual-based methods)."""
    rqvae_codebook_weights = None
    codebook_save_path = os.path.join(
        PATH_CONFIG.rqvae_save_dir,
        f"{PATH_CONFIG.id_filename}_codebook_weights_{config['seed']}.pt",
    )
    letter_codebook_save_path = os.path.join(
        PATH_CONFIG.rqvae_save_dir,
        f"{PATH_CONFIG.id_filename}_letter_codebook_weights_{config['seed']}.pt",
    )
    alt_paths = [
        codebook_save_path,
        letter_codebook_save_path,
        os.path.join(
            os.path.dirname(PATH_CONFIG.id_save_location),
            f"{PATH_CONFIG.id_filename}_codebook_weights_{config['seed']}.pt",
        ),
        os.path.join(
            os.path.dirname(PATH_CONFIG.id_save_location),
            f"{PATH_CONFIG.id_filename}_letter_codebook_weights_{config['seed']}.pt",
        ),
    ]

    loaded = False
    for try_path in alt_paths:
        if os.path.exists(try_path):
            try:
                rqvae_codebook_weights = torch.load(try_path, map_location=device)
                print(f"Loaded codebook weights from {try_path}")
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Could not load codebook weights from {try_path}: {e}")
                rqvae_codebook_weights = None

    if not loaded:
        print(f"Codebook weights not found. Re-training RQ-VAE...")
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
        print(f"Saved codebook weights to {codebook_save_path}")

    return rqvae_codebook_weights


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:

    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    config["logging"]["project"] = "ma_liger"
    is_steam = config["dataset"]["type"] == "steam"

    # Validate prototype config
    method_config = {**config["method"], **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]}}
    if not method_config.get("use_prototype", False):
        print("ERROR: use_prototype must be True for run_ma_liger.py")
        print("Set method.use_prototype=true in config or command line.")
        sys.exit(1)

    prototype_config = method_config.get("prototype", {})
    stage1_checkpoint = prototype_config.get("stage1_checkpoint", None)
    if not stage1_checkpoint or not os.path.exists(stage1_checkpoint):
        print(f"ERROR: Stage 1 checkpoint not found: {stage1_checkpoint}")
        print("Please train LIGER first (run.py) and provide the checkpoint path.")
        sys.exit(1)

    print(f"=" * 70)
    print(f"MA-LIGER Training")
    print(f"  Dataset: {config['dataset']['name']}")
    print(f"  Stage 1 checkpoint: {stage1_checkpoint}")
    print(f"  Prototype: K={prototype_config.get('K', 256)}, "
          f"p={prototype_config.get('p', 3)}, M={prototype_config.get('M', 10)}")
    print(f"=" * 70)

    try:
        data_file, id2meta_file, item2attribute_file = preprocessing(config["dataset"])

        train_config = {
            **config["dataset"],
            **{k: v for k, v in config.items() if k not in ["logging", "dataset", "method"]},
        }

        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )

        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        # Train SID tokenizer (same as run.py)
        if config["test_method"] == "letter":
            train_letter_sid(
                config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
            )
            letter_config = config["dataset"]["LETTER"]
            train_config["RQ-VAE"]["code_book_size"] = max(letter_config["num_emb_list"])
            train_config["RQ-VAE"]["num_layers"] = len(letter_config["num_emb_list"])
            letter_codebook_sizes = list(letter_config["num_emb_list"])
        else:
            train_sid(
                config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
            )
            letter_codebook_sizes = None

        # Load codebook weights
        rqvae_codebook_weights = _load_codebook_weights(
            config, device, PATH_CONFIG, item_embedding, id_split
        )

        # MA-LIGER requires residual decoder (same as LIGER)
        # Force these settings
        method_config["use_residual_decoder"] = True
        method_config["flag_use_output_embedding"] = True
        if "embedding_loss_weight" not in method_config or method_config["embedding_loss_weight"] == 0:
            method_config["embedding_loss_weight"] = 1.0
            print("[MA-LIGER] Set embedding_loss_weight=1.0 (LIGER mode)")

        # Run MA-LIGER training
        train_ma_liger(
            config,
            train_config,
            method_config,
            id_split,
            user_sequence,
            item_embedding,
            PATH_CONFIG.id_save_location,
            device=device,
            rqvae_codebook_weights=rqvae_codebook_weights,
            codebook_sizes=letter_codebook_sizes,
        )

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
