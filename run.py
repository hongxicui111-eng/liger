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
from ID_generation.train_letter import train as train_letter_sid
from ID_generation.utils import process_data_split, process_embeddings
from omegaconf import DictConfig
from src.training import train_tiger, train_tiger_residual, train_tiger_softlabel, train_simple_residual

from utils import set_seed


class set_dir:
    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        if config["test_method"] in ["tiger", "liger", "letter"]:
            self.rqvae_save_dir = "./ID_generation/ID/"
            os.makedirs(self.rqvae_save_dir, exist_ok=True)

            self.id_filename = (
                f"{config['dataset']['name']}_{config['dataset']['content_model']}"
            )
            # LETTER and TIGER/LIGER use different SID structures, so they must
            # be saved to separate files to avoid loading the wrong SID format.
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

    # Validate dataset name — must not be empty and must be a known Amazon dataset
    dataset_name = config["dataset"].get("name", "")
    dataset_type = config["dataset"].get("type", "")
    if not dataset_name:
        raise ValueError(
            f"dataset.name is empty or not set. "
            f"For Amazon datasets, valid names are: Beauty, Toys_and_Games, Sports_and_Outdoors, Amazon_Instant_Video, Home_and_Kitchen. "
            f"Please set it in configs/dataset/amazon.yaml or via the command line, e.g. 'dataset.name=Beauty'."
        )
    if dataset_type == "Amazon" and dataset_name not in [
        "Beauty", "Toys_and_Games", "Sports_and_Outdoors",
        "Amazon_Instant_Video", "Home_and_Kitchen",
    ]:
        print(
            f"WARNING: dataset.name='{dataset_name}' is not in the known Amazon dataset list "
            f"[Beauty, Toys_and_Games, Sports_and_Outdoors, Amazon_Instant_Video, Home_and_Kitchen]. "
            f"Data download may fail with KeyError."
        )

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

        # --- Train Semantic ID Tokenizer ---
        if config["test_method"] == "letter":
            # LETTER tokenizer: uses Sinkhorn-balanced assignment + diversity loss + CF loss
            cf_embedding = None

            # --- Step 1: Obtain CF embeddings ---
            # LETTER uses CF (collaborative filtering) embeddings from SASRec
            # to align quantized representations with user behavior signals.
            # If a pre-trained CF embedding file is specified and exists, load it.
            # Otherwise, automatically train SASRec on the user-item sequences.
            cf_emb_path = method_config.get("letter_cf_embedding_path", None)
            sasrec_config = config["dataset"].get("SASRec", {})

            # Determine default CF embedding save path
            if cf_emb_path is None:
                cf_emb_dir = os.path.join(
                    "./ID_generation/sasrec/ckpt/",
                    f"{config['dataset']['name']}_{config['dataset']['content_model']}"
                )
                cf_emb_path = os.path.join(
                    cf_emb_dir,
                    f"{config['dataset']['name']}-{sasrec_config.get('hidden_units', 32)}d-sasrec.pt"
                )

            if os.path.exists(cf_emb_path):
                # Load existing CF embeddings
                cf_embedding = torch.load(cf_emb_path, map_location=device, weights_only=False)
                if isinstance(cf_embedding, torch.Tensor):
                    cf_embedding = cf_embedding.squeeze().detach().cpu().numpy()
                print(f"✓ Loaded CF embeddings from {cf_emb_path}, shape: {cf_embedding.shape}")
            else:
                # Auto-train SASRec to generate CF embeddings
                print(f"⚠️ CF embedding file not found: {cf_emb_path}")
                print("→ Auto-training SASRec to generate CF embeddings...")

                from ID_generation.sasrec import train_sasrec as _train_sasrec
                from ID_generation.sasrec.train_sasrec import get_num_items_from_sequences

                num_items = get_num_items_from_sequences(user_sequence)
                print(f"  Total items: {num_items}, User sequences: {len(user_sequence)}")

                _train_sasrec(
                    user_sequences=user_sequence,
                    num_items=num_items,
                    device=device,
                    save_path=cf_emb_path,
                    hidden_units=sasrec_config.get("hidden_units", 32),
                    num_heads=sasrec_config.get("num_heads", 2),
                    num_blocks=sasrec_config.get("num_blocks", 2),
                    max_len=sasrec_config.get("max_len", 50),
                    dropout=sasrec_config.get("dropout", 0.2),
                    epochs=sasrec_config.get("epochs", 200),
                    lr=sasrec_config.get("lr", 0.001),
                    batch_size=sasrec_config.get("batch_size", 128),
                    weight_decay=sasrec_config.get("weight_decay", 0.0),
                    eval_steps=sasrec_config.get("eval_steps", 10),
                    patience=sasrec_config.get("patience", 20),
                    seed=config.get("seed", 42),
                )

                # Load the newly saved CF embeddings
                cf_embedding = torch.load(cf_emb_path, map_location=device, weights_only=False)
                if isinstance(cf_embedding, torch.Tensor):
                    cf_embedding = cf_embedding.squeeze().detach().cpu().numpy()
                print(f"✓ CF embeddings generated and loaded, shape: {cf_embedding.shape}")

            # --- Step 2: Train LETTER tokenizer with CF embeddings ---
            train_letter_sid(
                config, device, item_embedding, id_split, PATH_CONFIG.id_save_location,
                cf_embedding=cf_embedding,
            )

            # Override RQ-VAE config in train_config for LETTER compatibility.
            # LETTER's num_emb_list may differ from RQ-VAE's code_book_size/num_layers.
            # Set code_book_size to the max across all LETTER VQ layers,
            # and num_layers to the number of VQ levels.
            letter_config = config["dataset"]["LETTER"]
            train_config["RQ-VAE"]["code_book_size"] = max(letter_config["num_emb_list"])
            train_config["RQ-VAE"]["num_layers"] = len(letter_config["num_emb_list"])
            # Store LETTER's per-level codebook sizes for passing to training functions.
            # This is needed so that load_data/expand_id_arr correctly maps SID indices
            # to T5 vocabulary tokens using cumulative offsets (not uniform codebook_size*level).
            letter_codebook_sizes = list(letter_config["num_emb_list"])
        else:
            # Original TIGER/LIGER tokenizer (EMA-based RQ-VAE)
            train_sid(
                config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
            )
            letter_codebook_sizes = None  # not using LETTER tokenizer

        # --- Model Selection ---
        # Four modes:
        #   1. use_residual_decoder=True  → TIGER_Residual (soft label + residual interleaving)
        #   2. use_simple_residual=True   → Simple_Residual (aux codebook/residual losses, no residual in input)
        #   3. use_softlabel_sid=True     → TIGER_SoftLabel (soft label, no residual)
        #   4. default                    → TIGER (hard label, no residual)
        # Note: all modes work with both EMA RQ-VAE and LETTER tokenizer IDs.
        #       The backbone model (TIGER) is the same; only the tokenizer differs.
        if method_config.get("use_residual_decoder", False):
            # Load RQ-VAE codebook weights for TIGER_Residual
            # We only need the codebook embedding matrices, NOT the entire RQ-VAE model.
            # Each codebook is [codebook_size, latent_size] — e.g. [256, 128], tiny (~384KB total).
            rqvae_codebook_weights = None
            codebook_save_path = os.path.join(
                PATH_CONFIG.rqvae_save_dir,
                f"{PATH_CONFIG.id_filename}_codebook_weights_{config['seed']}.pt",
            )
            # For LETTER tokenizer, also check letter-specific codebook weights
            letter_codebook_save_path = os.path.join(
                PATH_CONFIG.rqvae_save_dir,
                f"{PATH_CONFIG.id_filename}_letter_codebook_weights_{config['seed']}.pt",
            )
            # Also check alongside the SID .pkl file
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
                codebook_sizes=letter_codebook_sizes,
            )
        elif method_config.get("use_simple_residual", False):
            # --- Simple Residual Experiment ---
            # Simple_Residual: auxiliary codebook/residual losses only.
            # No residual interleaving or injection into decoder input.
            # Decoder runs standard autoregressive <BOS> sid_0 sid_1 ...
            rqvae_codebook_weights = None
            codebook_save_path = os.path.join(
                PATH_CONFIG.rqvae_save_dir,
                f"{PATH_CONFIG.id_filename}_codebook_weights_{config['seed']}.pt",
            )
            # For LETTER tokenizer, also check letter-specific codebook weights
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
                        print(f"✓ Loaded codebook weights (simple_residual) from {try_path}")
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

            train_simple_residual(
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
        elif method_config.get("use_softlabel_sid", False):
            # --- Soft Label SID Experiment ---
            # TIGER_SoftLabel: uses distance-based soft labels from codebook,
            # but no residual interleaving. This isolates the soft label effect.
            rqvae_codebook_weights = None
            codebook_save_path = os.path.join(
                PATH_CONFIG.rqvae_save_dir,
                f"{PATH_CONFIG.id_filename}_codebook_weights_{config['seed']}.pt",
            )
            alt_paths = [
                codebook_save_path,
                os.path.join(
                    os.path.dirname(PATH_CONFIG.id_save_location),
                    f"{PATH_CONFIG.id_filename}_codebook_weights_{config['seed']}.pt",
                ),
            ]

            loaded = False
            for try_path in alt_paths:
                if os.path.exists(try_path):
                    try:
                        rqvae_codebook_weights = torch.load(try_path, map_location=device)
                        print(f"✓ Loaded codebook weights (softlabel) from {try_path}")
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

            train_tiger_softlabel(
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
                codebook_sizes=letter_codebook_sizes,
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
