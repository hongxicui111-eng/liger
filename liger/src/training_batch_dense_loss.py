"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from transformers import T5Config
from transformers.optimization import get_scheduler

from utils import CustomDataset, get_lr, setup_logging

from .evaluation import (
    evaluate,
    evaluate_dense_ids,
    evaluate_dense_sids,
    evaluate_prefix_then_dense,
    generate_then_dense,
    get_target_embed,
    model_forward,
)
from .load_data import build_prefix2items, load_data
from .tiger import TIGER


def evaluate_helper(
    model,
    device,
    val_dataloader_dict,
    val_unseen_semantic_ids,
    all_semantic_ids,
    item2sid,
    item_embedding,
    method_config,
    keyword="eval",
    KEYS=[10],  # recall@k, k list
    RETRIEVE_KEY=[20, 40, 60, 80, 100],  # retrieve then rank
    prefix2items=None,
):
    """
    :param KEYS: the keys for the recall@k
    :param RETRIEVE_KEY: the keys for the retrieve then rank, only used for liger
    :param prefix2items: prefix -> item-set mapping, only used for the hybrid method
    """

    model.eval()
    logs = {}

    def add_log(logs, result_dict, result_name, prefix):
        logs[f"{prefix}/{result_name}"] = torch.tensor(result_dict).mean()
        return logs

    def _evaluate(logs, dataloader, name):
        if method_config["sid_loss_weight"] > 0:
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
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            else:
                recall_dict, ndcg_dict = evaluate_dense_sids(
                    model,
                    dataloader,
                    device,
                    item2sid,
                    item_embedding=item_embedding,
                    method_config=method_config,
                    KEYS=KEYS,
                )
            for key in recall_dict.keys():
                logs = add_log(logs, recall_dict[key], f"Recall@{key}", name)
                logs = add_log(logs, ndcg_dict[key], f"NDCG@{key}", name)
        return logs

    def _hybrid_evaluate(logs, dataloader, name):
        # hybrid: generate a prefix, then dense-retrieve within the prefix bucket
        if (
            method_config["evaluation_method"] == "hybrid"
            and method_config["embedding_loss_weight"] > 0
            and prefix2items is not None
        ):
            recall_dict, ndcg_dict = evaluate_prefix_then_dense(
                model,
                dataloader,
                device,
                item2sid,
                item_embedding=item_embedding,
                method_config=method_config,
                prefix2items=prefix2items,
                KEYS=KEYS,
                num_candidates=method_config.get("hybrid_num_candidates", 100),
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
                model,
                dataloader,
                val_unseen_semantic_ids,
                device,
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
                        logs,
                        recall_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_Recall@{key}",
                        name,
                    )
                    logs = add_log(
                        logs,
                        ndcg_dict[_retrieve_key][key],
                        f"Gen{_retrieve_key}_NDCG@{key}",
                        name,
                    )
        return logs

    # Hybrid method: only run the prefix-then-dense evaluation (skip the
    # generative / dense / unified evals that are designed for liger/tiger).
    is_hybrid = method_config["evaluation_method"] == "hybrid"

    if is_hybrid:
        if "test" in keyword:
            logs = _hybrid_evaluate(
                logs, val_dataloader_dict["in_set"], f"hybrid_in_{keyword}"
            )
            logs = _hybrid_evaluate(
                logs, val_dataloader_dict["cold_start"], f"hybrid_cold_{keyword}"
            )
        else:
            logs = _hybrid_evaluate(
                logs, val_dataloader_dict["in_set"], f"hybrid_in_{keyword}"
            )

        ndcg_at_10 = logs[f"hybrid_in_{keyword}/NDCG@10"]
        return logs, ndcg_at_10

    if "test" in keyword:
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        logs, returned_cand_cold, returned_embd_cold = _evaluate(
            logs, val_dataloader_dict["cold_start"], f"genret_cold_{keyword}"
        )

        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )
    else:  # during training, do selected eval
        logs, returned_cand_in, returned_embd_in = _evaluate(
            logs, val_dataloader_dict["in_set"], f"genret_in_{keyword}"
        )
        if method_config["flag_use_output_embedding"]:
            logs = _dense_evaluate(
                logs, val_dataloader_dict["in_set_embd"], f"dense_in_{keyword}"
            )

    if method_config["flag_use_output_embedding"] and "test" in keyword:
        logs = _dense_evaluate(
            logs, val_dataloader_dict["cold_start_embd"], f"dense_cold_{keyword}"
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["in_set"],
            returned_cand_in,
            returned_embd_in,
            f"uni_in_{keyword}",
        )
        logs = _unified_evaluate(
            logs,
            val_dataloader_dict["cold_start"],
            returned_cand_cold,
            returned_embd_cold,
            f"uni_cold_{keyword}",
        )

    if (
        method_config["evaluation_method"] == "dense"
        and method_config["embedding_loss_weight"] > 0
    ):
        ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
    else:
        if method_config["sid_loss_weight"] == 0:
            ndcg_at_10 = logs[f"dense_in_{keyword}/NDCG@10"]
        else:
            ndcg_at_10 = logs[f"genret_in_{keyword}/NDCG@10"]

    return logs, ndcg_at_10


def train_epoch(
    epoch,
    train_dataloader,
    model,
    optimizer,
    device,
    scaler,
    scheduler,
    writer,
    seen_ids,
    n_semantic_codebook,
    n_codebook,
    method_config,
    item2sid,
    item_embedding,
):
    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)

    # Hybrid method: the generative model is only trained to predict the first
    # `prefix_depth` sids (the prefix). sids beyond `prefix_depth` are masked
    # out from the autoregressive loss with -100.
    #
    # Dense retrieval loss (in-batch contrastive, "not 摸高" version): only the
    # B target items in the current batch are projected. The positive for
    # sample i is its own target item; the other B-1 targets in the batch act
    # as negatives. This avoids projecting the full item library every step
    # (the original liger cost) and removes the prefix-bucket lookup. Each
    # train batch only contains seen items (unseen items only appear in
    # val/test splits), so every target is a valid candidate.
    prefix_depth = method_config.get("prefix_depth", 0)

    for batch in train_dataloader:
        optimizer.zero_grad()

        if (
            method_config["use_id"] == "sid"
            and prefix_depth
            and prefix_depth < batch["labels_sids"].shape[1]
        ):
            # only supervise the first `prefix_depth` sids; mask the rest
            labels_sids = batch["labels_sids"].clone()
            labels_sids[:, prefix_depth:] = -100
            batch = {**batch, "labels_sids": labels_sids}

        outputs, _ = model_forward(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Calculating the loss
        loss = 0
        hard_loss = outputs.loss
        loss += hard_loss * method_config["sid_loss_weight"]

        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = (
                model.predicted_embedding
            )  # [B, d_model]
            logits_label = batch["labels_ids"][:, 0].to(device) - 1  # [B], 0-based
            supposed_sid_label = item2sid[logits_label.cpu()]  # [B, n_code]
            # Only the supervised (unmasked) prefix positions are guaranteed to
            # match the label; positions beyond `prefix_depth` are -100.
            n_cmp = (
                prefix_depth
                if (prefix_depth and prefix_depth < n_codebook)
                else n_codebook
            )
            assert (
                supposed_sid_label[:, :n_cmp]
                == batch["labels_sids"][:, :n_cmp].numpy()
            ).all()

            # In-batch contrastive: project only the B target items in this
            # batch (not the full library). get_target_embed returns logits
            # [B, B] where logits[i, j] = cos(query_i, target_j) / temp; the
            # diagonal is the positive pair, off-diagonals are negatives.
            target_emb = item_embedding[logits_label]  # [B, 768]
            _, logits = get_target_embed(
                predicted_embedding, model, method_config, target_emb
            )  # [B, B]
            targets = torch.arange(
                logits.shape[0], device=logits.device
            )  # diagonal = positive
            embedding_loss = F.cross_entropy(logits, targets)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    progress_bar.close()

    logs = {
        "train/loss": loss.item(),
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
    }
    logs["train/sid_loss"] = hard_loss
    logs["train/embedding_loss"] = embedding_loss

    writer.log(logs)

    return model


def train_tiger(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
):

    output_path = config["output_path"]
    codebook_size = config["RQ-VAE"]["code_book_size"]
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    ######### NOW RE-WRITE CONFIGS ###########
    config = config["TIGER"]
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
        id_save_location,
        user_sequence,
        unseen_val,
        unseen_test,
        seen,
        item_embedding,
        method_config,
        max_length=config["n_positions"],
        codebook_size=codebook_size,
        max_items_per_seq=max_items_per_seq,
    )
    all_semantic_ids = np.unique(
        np.concatenate(
            [seen_semantic_ids, val_unseen_semantic_ids, test_unseen_semantic_ids],
            axis=0,
        ),
        axis=0,
    )  # [n_items, n_code]
    unseen_semantic_ids = np.unique(
        np.concatenate([val_unseen_semantic_ids, test_unseen_semantic_ids], axis=0),
        axis=0,
    )  # [n_items, n_code]

    # Build the prefix -> item-set mapping used by the hybrid method (eval only;
    # the training dense loss now uses in-batch contrastive and does not need
    # the prefix buckets).
    prefix2items = None
    if method_config["evaluation_method"] == "hybrid":
        prefix2items = build_prefix2items(item2sid, method_config["prefix_depth"])
        print(
            f"[hybrid] Built prefix2items mapping: {len(prefix2items)} unique "
            f"prefixes (prefix_depth={method_config['prefix_depth']}, "
            f"n_codebook={n_codebook})"
        )

    # deal with the output embedding
    if method_config["flag_use_output_embedding"]:
        item_embedding = item_embedding.to(device)

    unseen_val_dataset = CustomDataset(unseen_val_data)
    unseen_test_dataset = CustomDataset(unseen_test_data)
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    seen_semantic_ids = torch.from_numpy(seen_semantic_ids)
    val_unseen_semantic_ids = torch.from_numpy(val_unseen_semantic_ids)
    test_unseen_semantic_ids = torch.from_numpy(test_unseen_semantic_ids)
    all_semantic_ids = torch.from_numpy(all_semantic_ids)
    unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

    # Check the fourth semantic ID size
    if method_config.get("use_extra_sid", True):
        # original liger: 3 RQ-VAE codebooks + 1 extra conflict-resolution codebook
        last_codebook_size = max(max_last_semantic_ids, codebook_size)
    else:
        # hybrid method: only the 3 RQ-VAE codebooks, no extra sid
        last_codebook_size = 0
    if method_config["include_user_id"]:
        this_vocab_size = (
            2000 + codebook_size * n_semantic_codebook + last_codebook_size + 2
        )  # by default this is 3026
        # 2000 for user
    else:
        this_vocab_size = codebook_size * n_semantic_codebook + last_codebook_size + 2

    if method_config["use_id"] == "item_id":
        this_vocab_size = item_embedding.shape[0] + 2

    t5_config = config["T5"]
    trainer_config = config["trainer"]
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
        n_positions=config["n_positions"],
        layer_norm_epsilon=1e-8,
        initializer_factor=t5_config["initializer_factor"],
    )

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    # Initialize the model with the custom configuration
    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader_embedding = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    unseen_val_dataloader = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_val_dataloader_embedding = DataLoader(
        unseen_val_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    unseen_test_dataloader_embedding = DataLoader(
        unseen_test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    val_dataloader_dict = {
        "in_set": val_dataloader,
        "in_set_embd": val_dataloader_embedding,
        "cold_start": unseen_val_dataloader,
        "cold_start_embd": unseen_val_dataloader_embedding,
    }
    test_dataloader_dict = {
        "in_set": test_dataloader,
        "in_set_embd": test_dataloader_embedding,
        "cold_start": unseen_test_dataloader,
        "cold_start_embd": unseen_test_dataloader_embedding,
    }

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})

    print(f"Total number of parameters: {total_params}")
    print(
        f"Total number of epochs: {int(np.ceil(total_steps / len(train_dataloader)))}"
    )

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        # then this is the liger method
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        # then this could be TIGER or dense method only
        RETRIEVE_KEY = [10]

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"

    optimizer = AdamW(
        model.parameters(),
        lr=trainer_config["lr"],
        weight_decay=trainer_config["weight_decay"],
    )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    scheduler = None
    if trainer_config["scheduler"] != "none":
        scheduler = get_scheduler(
            name=trainer_config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=trainer_config["warmup_steps"],
            num_training_steps=total_steps,
        )

    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )  # NOTE: change to cpu if OOM
        state_dict = training_state["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_ndcg_10 = training_state["best_ndcg_10"]
        global_step = training_state["global_step"]
        best_epoch = training_state["best_epoch"]
        start_epoch = training_state["train_step"]
        if scheduler is not None:
            for _ in range(global_step):
                scheduler.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Load the model from: ", state_path)

    for epoch in range(
        start_epoch + 1, int(np.ceil(total_steps / len(train_dataloader)))
    ):
        model = train_epoch(
            epoch,
            train_dataloader,
            model,
            optimizer,
            device,
            scaler,
            scheduler,
            writer,
            seen,
            n_semantic_codebook,
            n_codebook,
            method_config,
            item2sid,
            item_embedding,
        )
        global_step += len(train_dataloader)

        if (epoch + 1) % trainer_config["eval_frequence"] == 0:

            logs, ndcg_at_10 = evaluate_helper(
                model,
                device,
                val_dataloader_dict,
                unseen_semantic_ids,
                all_semantic_ids,
                item2sid,
                item_embedding,
                method_config,
                keyword="val",
                RETRIEVE_KEY=RETRIEVE_KEY,
                prefix2items=prefix2items,
            )
            logs["train/step"] = global_step

            if ndcg_at_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

            # save from time to time
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": epoch,
                "best_ndcg_10": best_ndcg_10,
                "global_step": global_step,
                "best_epoch": best_epoch,
            }
            torch.save(training_state, state_path)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            print("Finish because the patience run out.")
            break

    print("Testing...")

    model = TIGER(
        config=model_config,
        n_semantic_codebook=n_semantic_codebook,
        max_items_per_seq=max_items_per_seq,
        flag_use_output_embedding=method_config["flag_use_output_embedding"],
        flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
        embedding_head_dict=method_config["embedding_head_dict"],
    ).to(device)

    model.load_state_dict(torch.load(best_state_path), strict=True)

    logs, _ = evaluate_helper(
        model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
        prefix2items=prefix2items,
    )

    writer.log(logs)
    writer.finish()

    return
