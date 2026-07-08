"""
MA-LIGER: Memory-Augmented LIGER Training Module
==================================================

Two-stage training:
  Stage 1: Standard LIGER training (train_tiger_residual from parent)
  Stage 2: Joint fine-tuning with Prototype Memory

Stage 2 flow:
  1. Load Stage 1 checkpoint (LIGER encoder + decoder weights)
  2. Initialize Prototype Memory via K-Means on training embeddings
  3. Build optimizer with differential learning rates:
     - Encoder: base_lr * encoder_lr_factor (slow, avoid representation drift)
     - Decoder: base_lr (normal, must adapt to Prototype-augmented encoder_hidden)
     - Prototype + Cross-Attention + Gate: base_lr (fast, main learning target)
  4. Train with combined loss:
     L_total = L_rec + beta * L_commitment + delta * L_diversity
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import T5Config
from transformers.optimization import get_scheduler

from utils import CustomDataset, get_lr, setup_logging
from .load_data import load_data
from .tiger_residual_ma import TIGER_Residual_MA
from .prototype_memory import build_prototype_init


# =============================================================================
# MA-LIGER Training Epoch
# =============================================================================


def train_epoch_ma_liger(
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
    """Training epoch for MA-LIGER (TIGER_Residual_MA) model.

    Same as train_epoch_residual but also handles prototype losses
    (commitment_loss, diversity_loss) which are already included in
    outputs['loss'] by the model's forward_residual.
    """
    from .evaluation import model_forward_residual, get_target_embed

    model.train()
    all_ids = np.arange(item2sid.shape[0]) + 1
    unseen_ids = np.setdiff1d(all_ids, seen_ids)

    # Accumulators
    loss = torch.tensor(0.0, device=device)
    hard_loss = 0.0
    codebook_loss_val = torch.tensor(0.0, device=device)
    grad_norm = 0.0
    embedding_loss = 0.0
    num_valid_batches = 0

    # Prototype loss accumulators
    commitment_loss_sum = 0.0
    diversity_loss_sum = 0.0
    prototype_active_sum = 0
    prototype_max_usage_max = 0
    gate_values_sum = 0.0  # Track average gate value per epoch

    # Non-TF statistics
    nontf_resid_pct_sum = 0.0
    nontf_ntp_pct_sum = 0.0

    # Soft-label temperature tracking
    current_soft_label_temp = None

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        outputs, _ = model_forward_residual(
            model,
            batch,
            device,
            n_codebook,
            method_config,
        )

        # Extract losses
        hard_loss = outputs["sid_loss"]
        codebook_loss_val = outputs["codebook_loss"]
        cumulative_residual_loss_val = outputs.get("cumulative_residual_loss", 0.0)

        gate_values_batch = outputs.get("gate_values", None)
        if gate_values_batch is not None:
            gate_values_sum += gate_values_batch.mean().item()

        # Prototype losses
        commitment_loss_batch = outputs.get("commitment_loss", torch.tensor(0.0))
        diversity_loss_batch = outputs.get("diversity_loss", torch.tensor(0.0))
        prototype_active = outputs.get("prototype_active_count", 0)
        prototype_max_usage = outputs.get("prototype_max_usage", 0)

        commitment_loss_sum += commitment_loss_batch.item() if isinstance(commitment_loss_batch, torch.Tensor) else commitment_loss_batch
        diversity_loss_sum += diversity_loss_batch.item() if isinstance(diversity_loss_batch, torch.Tensor) else diversity_loss_batch
        prototype_active_sum += prototype_active
        prototype_max_usage_max = max(prototype_max_usage_max, prototype_max_usage)

        # Accumulate non-TF statistics
        nontf_resid_pct_sum += outputs.get("nontf_resid_pct", 0.0)
        nontf_ntp_pct_sum += outputs.get("nontf_ntp_pct", 0.0)

        if "current_soft_label_temp" in outputs:
            current_soft_label_temp = outputs["current_soft_label_temp"]

        logits = outputs["logits"]

        # NaN guard
        logits_has_nan = torch.isnan(logits).any().item()
        logits_has_inf = torch.isinf(logits).any().item()
        loss_is_nan = torch.isnan(outputs["loss"]).item() if isinstance(outputs["loss"], torch.Tensor) else False

        if logits_has_nan or logits_has_inf or loss_is_nan:
            print(f"\n[NaN Warning] epoch={epoch}, skipping batch")
            continue

        # Build total loss (prototype losses already included in outputs['loss'])
        loss = hard_loss * method_config["sid_loss_weight"]
        loss += codebook_loss_val * method_config.get("codebook_loss_weight", 1.0)
        loss += cumulative_residual_loss_val * method_config.get("cumulative_residual_loss_weight", 0.0)

        # Add prototype losses explicitly
        if model.use_prototype and model.prototype_memory is not None:
            loss = loss + model.prototype_memory.beta * commitment_loss_batch
            loss = loss + model.prototype_memory.delta * diversity_loss_batch

        num_valid_batches += 1

        # Embedding loss (LIGER dense retrieval)
        embedding_loss = 0
        if method_config["flag_use_output_embedding"]:
            predicted_embedding = model.predicted_embedding
            _, logits_dense = get_target_embed(
                predicted_embedding, model, method_config, item_embedding
            )
            logits_label = batch["labels_ids"][:, 0].to(device) - 1
            supposed_sid_label = item2sid[logits_label.cpu()]
            assert (
                supposed_sid_label == batch["labels_sids"][:, :n_codebook].numpy()
            ).all()
            logits_dense[:, unseen_ids - 1] = -100
            embedding_loss = F.cross_entropy(logits_dense, logits_label)
        loss += embedding_loss * method_config["embedding_loss_weight"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    if num_valid_batches == 0:
        print(f"Warning: No valid batches in epoch {epoch + 1}")

    nontf_resid_pct_avg = nontf_resid_pct_sum / num_valid_batches if num_valid_batches > 0 else 0.0
    nontf_ntp_pct_avg = nontf_ntp_pct_sum / num_valid_batches if num_valid_batches > 0 else 0.0

    # Compute averages
    avg_commitment = commitment_loss_sum / num_valid_batches if num_valid_batches > 0 else 0.0
    avg_diversity = diversity_loss_sum / num_valid_batches if num_valid_batches > 0 else 0.0
    avg_prototype_active = prototype_active_sum / num_valid_batches if num_valid_batches > 0 else 0.0
    avg_gate = gate_values_sum / num_valid_batches if num_valid_batches > 0 else 0.0

    sid_loss_val = hard_loss if isinstance(hard_loss, float) else hard_loss.item()
    codebook_loss_val_item = codebook_loss_val.item() if isinstance(codebook_loss_val, torch.Tensor) else codebook_loss_val
    loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

    # ---- Per-epoch terminal print ----
    proto_str = ""
    if model.use_prototype and model.prototype_memory is not None:
        proto_str = (
            f" | commit={avg_commitment:.4f} divers={avg_diversity:.4f}"
            f" | Proto: active={avg_prototype_active:.0f}/{model.prototype_memory.K}"
            f" max_usage={prototype_max_usage_max} gate={avg_gate:.3f}"
        )
    embed_str = f"{embedding_loss:.4f}" if isinstance(embedding_loss, (float, int)) else f"{embedding_loss}"
    print(
        f"Epoch {epoch+1}: loss={loss_val:.4f}"
        f" | sid={sid_loss_val:.4f} codebook={codebook_loss_val_item:.4f}"
        f" embed={embed_str}{proto_str}"
    )

    logs = {
        "train/loss": loss_val,
        "train/epoch": epoch + 1,
        "train/lr": get_lr(optimizer),
        "train/grad_norm": grad_norm,
        "train/sid_loss": sid_loss_val,
        "train/codebook_loss": codebook_loss_val_item,
        "train/embedding_loss": embedding_loss,
        "train/nontf_resid_pct": nontf_resid_pct_avg,
        "train/nontf_ntp_pct": nontf_ntp_pct_avg,
        # Prototype-specific logs
        "train/commitment_loss": avg_commitment,
        "train/diversity_loss": avg_diversity,
        "train/prototype_active_avg": avg_prototype_active,
        "train/prototype_max_usage": prototype_max_usage_max,
        "train/gate_values_avg": avg_gate,
    }

    if current_soft_label_temp is not None:
        logs["train/soft_label_temp"] = current_soft_label_temp

    per_position_alphas = outputs.get("per_position_alphas", None)
    if per_position_alphas is not None:
        for k, alpha_k in enumerate(per_position_alphas):
            logs[f"train/cb_weight_alpha_k{k}"] = alpha_k

    writer.log(logs)
    return model


# =============================================================================
# MA-LIGER Main Training Function
# =============================================================================


def train_ma_liger(
    orig_config,
    config,
    method_config,
    id_split,
    user_sequence,
    item_embedding,
    id_save_location,
    device,
    rqvae_codebook_weights=None,
    codebook_sizes=None,
):
    """Main training function for MA-LIGER (Stage 2: joint fine-tuning with Prototype).

    This function assumes Stage 1 (standard LIGER training) has already been completed.
    The Stage 1 checkpoint path is specified in method_config['prototype']['stage1_checkpoint'].

    Args:
        (same as train_tiger_residual, plus prototype config in method_config)
    """
    from .training import evaluate_helper_residual

    output_path = config["output_path"]

    if codebook_sizes is None:
        codebook_sizes = config["RQ-VAE"]["code_book_size"]
    if isinstance(codebook_sizes, int):
        codebook_size = codebook_sizes
    else:
        codebook_size = codebook_sizes[0]
    max_items_per_seq = config["max_items_per_seq"]

    writer = setup_logging(orig_config)

    config_tiger = config["TIGER"]
    unseen_val, unseen_test, seen = (
        id_split["unseen_val"],
        id_split["unseen_test"],
        id_split["seen"],
    )

    # Load data (same as train_tiger_residual)
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
        max_length=config_tiger["n_positions"],
        codebook_sizes=codebook_sizes,
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

    # Vocab size computation
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
        sid_vocab_size = codebook_sizes * n_semantic_codebook + last_codebook_size
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))
        sid_vocab_size = sum(codebook_sizes) + last_codebook_size

    flag_separate_bos_representation = method_config.get("flag_separate_bos_representation", False)
    if method_config["include_user_id"]:
        this_vocab_size = 2000 + sid_vocab_size + 2
    else:
        this_vocab_size = sid_vocab_size + 2

    if flag_separate_bos_representation:
        this_vocab_size += 1

    sid_start_token_id = this_vocab_size - 2 if flag_separate_bos_representation else None

    t5_config = config_tiger["T5"]
    trainer_config = config_tiger["trainer"]
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

    os.makedirs(f"{output_path}/logs", exist_ok=True)
    os.makedirs(f"{output_path}/results", exist_ok=True)

    latent_size = t5_config["d_model"]
    if rqvae_codebook_weights is not None:
        latent_size = rqvae_codebook_weights[0].shape[-1]

    # Residual config (same as train_tiger_residual)
    codebook_loss_weight = method_config.get("codebook_loss_weight", 1.0)
    num_residual_levels = method_config.get("num_residual_levels", n_semantic_codebook - 1)
    soft_label_K = method_config.get("soft_label_K", 0)
    soft_label_temperature = method_config.get("soft_label_temperature", 1.0)
    soft_label_temp_min = method_config.get("soft_label_temp_min", None)
    soft_label_temp_decay_steps = method_config.get("soft_label_temp_decay_steps", 10000)
    codebook_loss_weight_decay_steps = method_config.get("codebook_loss_weight_decay_steps", None)
    cumulative_residual_loss_weight = method_config.get("cumulative_residual_loss_weight", 0.0)
    cumulative_residual_loss_type = method_config.get("cumulative_residual_loss_type", "mse")
    cumulative_residual_loss_temperature = method_config.get("cumulative_residual_loss_temperature", 1.0)
    resid_nontf_ratio = method_config.get("resid_nontf_ratio", 0.0)
    ntp_nontf_ratio = method_config.get("ntp_nontf_ratio", 0.0)

    # ===== Prototype config =====
    prototype_config = method_config.get("prototype", {})
    K = prototype_config.get("K", 256)
    p = prototype_config.get("p", 3)
    L = prototype_config.get("L", 8)
    beta = prototype_config.get("beta", 0.25)
    delta = prototype_config.get("delta", 0.1)
    ema_decay = prototype_config.get("ema_decay", 0.99)
    n_heads_proto = prototype_config.get("n_heads", 8)
    gate_hidden = prototype_config.get("gate_hidden", 64)
    dropout_proto = prototype_config.get("dropout", 0.1)
    stage1_checkpoint = prototype_config.get("stage1_checkpoint", None)
    encoder_lr_factor = prototype_config.get("encoder_lr_factor", 0.1)

    # ===== Build MA-LIGER model =====
    model = TIGER_Residual_MA(
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
        soft_label_temp_min=soft_label_temp_min,
        soft_label_temp_decay_steps=soft_label_temp_decay_steps,
        codebook_loss_weight_decay_steps=codebook_loss_weight_decay_steps,
        cumulative_residual_loss_weight=cumulative_residual_loss_weight,
        cumulative_residual_loss_type=cumulative_residual_loss_type,
        cumulative_residual_loss_temperature=cumulative_residual_loss_temperature,
        resid_nontf_ratio=resid_nontf_ratio,
        ntp_nontf_ratio=ntp_nontf_ratio,
        flag_separate_bos_representation=flag_separate_bos_representation,
        sid_start_token_id=sid_start_token_id,
        use_prototype=True,
        prototype_config={
            "K": K, "p": p, "L": L,
            "beta": beta, "delta": delta,
            "ema_decay": ema_decay,
            "n_heads": n_heads_proto,
            "gate_hidden": gate_hidden,
            "dropout": dropout_proto,
        },
    ).to(device)

    # Set codebook offsets for LETTER
    if not isinstance(codebook_sizes, int):
        model.set_codebook_offsets(codebook_sizes)

    # ===== Load Stage 1 checkpoint =====
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        print(f"[MA-LIGER] Loading Stage 1 checkpoint from: {stage1_checkpoint}")
        checkpoint = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load with strict=False: Prototype and Cross-Attention params are new
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[MA-LIGER] Loaded Stage 1 weights. Missing keys (new params): {len(missing)}")
        print(f"[MA-LIGER]   Missing: {missing}")
        print(f"[MA-LIGER]   Unexpected: {len(unexpected)}")
    else:
        print(f"[MA-LIGER] WARNING: No Stage 1 checkpoint provided. Training from scratch.")
        print(f"[MA-LIGER] For best results, provide stage1_checkpoint in prototype config.")

    # ===== Initialize Prototype Memory via K-Means =====
    batch_size = trainer_config["batch_size"]
    eval_batch_size = trainer_config["eval_batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"[MA-LIGER] Initializing Prototype Memory via K-Means (K={K}, L={L})...")
    init_dict = build_prototype_init(
        model=model,
        train_dataloader=train_dataloader,
        device=device,
        K=K,
        L=L,
    )

    # Set prototype_tokens from K-Means centers
    model.prototype_memory.prototype_tokens.data.copy_(
        torch.from_numpy(init_dict["prototype_tokens"]).to(device)
    )
    print(f"[MA-LIGER] Prototype Memory initialized from K-Means.")

    # ===== Build DataLoaders =====
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    val_dataloader_embedding = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    test_dataloader_embedding = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    unseen_val_dataloader = DataLoader(unseen_val_dataset, batch_size=eval_batch_size, shuffle=False)
    unseen_val_dataloader_embedding = DataLoader(unseen_val_dataset, batch_size=eval_batch_size, shuffle=False)
    unseen_test_dataloader = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)
    unseen_test_dataloader_embedding = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)

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

    # ===== Optimizer with Differential Learning Rates =====
    # Group 1: Encoder (low lr — avoid representation drift)
    # Group 2: Decoder (normal lr — must adapt to Prototype-augmented encoder_hidden)
    # Group 3: Prototype + Cross-Attention + Gate (normal lr — new params need to learn)
    encoder_params = []
    decoder_params = []
    prototype_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "prototype_memory" in name or "proto_cross_attn" in name:
            prototype_params.append(param)
        elif "encoder" in name:
            encoder_params.append(param)
        else:
            # Decoder + shared embeddings + other components
            decoder_params.append(param)

    base_lr = trainer_config["lr"]
    optimizer = AdamW([
        {
            "params": encoder_params,
            "lr": base_lr * encoder_lr_factor,
            "weight_decay": trainer_config["weight_decay"],
        },
        {
            "params": decoder_params,
            "lr": base_lr,
            "weight_decay": trainer_config["weight_decay"],
        },
        {
            "params": prototype_params,
            "lr": base_lr,
            "weight_decay": trainer_config["weight_decay"],
        },
    ])

    print(f"[MA-LIGER] Optimizer: encoder_lr={base_lr * encoder_lr_factor:.6f}, "
          f"decoder_lr={base_lr:.6f}, prototype_lr={base_lr:.6f}")
    print(f"[MA-LIGER] Encoder params: {sum(p.numel() for p in encoder_params)}")
    print(f"[MA-LIGER] Decoder params: {sum(p.numel() for p in decoder_params)}")
    print(f"[MA-LIGER] Prototype params: {sum(p.numel() for p in prototype_params)}")

    total_steps = trainer_config["steps"]
    scheduler = None
    if trainer_config["scheduler"] != "none":
        scheduler = get_scheduler(
            name=trainer_config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=trainer_config["warmup_steps"],
            num_training_steps=total_steps,
        )

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda")
    elif hasattr(torch.cuda, "amp"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # ===== Training Loop =====
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    writer.log({"total_param": total_params})
    print(f"Total number of parameters: {total_params}")

    if (
        method_config["embedding_loss_weight"] > 0
        and method_config["sid_loss_weight"] > 0
    ):
        RETRIEVE_KEY = [20, 40, 60, 80, 100]
    else:
        RETRIEVE_KEY = [10]

    best_ndcg_10 = -0.01
    global_step = 0
    best_epoch = 0
    start_epoch = -1
    state_path = output_path + "/ckpt.pt"
    best_state_path = output_path + "/results/ckpt_best.pt"

    # Resume from interrupted training
    if os.path.exists(state_path):
        training_state = torch.load(
            state_path, map_location=device, weights_only=False
        )
        model.load_state_dict(training_state["model_state_dict"], strict=False)
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
        print(f"[MA-LIGER] Resumed from: {state_path}, global_step={global_step}")

    total_epochs = int(np.ceil(total_steps / len(train_dataloader)))
    print(f"Total number of epochs: {total_epochs}")

    for epoch in range(start_epoch + 1, total_epochs):
        model = train_epoch_ma_liger(
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
            logs, ndcg_at_10 = evaluate_helper_residual(
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
            )
            logs["train/step"] = global_step

            if ndcg_at_10 > best_ndcg_10:
                best_ndcg_10 = ndcg_at_10
                best_epoch = epoch
                model.cpu()
                torch.save(model.state_dict(), best_state_path)
                model.to(device)

            writer.log(logs)

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
            print("Early stopping: patience exhausted.")
            break

    # ===== Testing =====
    print("[MA-LIGER] Testing...")

    test_model = TIGER_Residual_MA(
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
        soft_label_temp_min=soft_label_temp_min,
        soft_label_temp_decay_steps=soft_label_temp_decay_steps,
        codebook_loss_weight_decay_steps=codebook_loss_weight_decay_steps,
        cumulative_residual_loss_weight=cumulative_residual_loss_weight,
        cumulative_residual_loss_type=cumulative_residual_loss_type,
        cumulative_residual_loss_temperature=cumulative_residual_loss_temperature,
        resid_nontf_ratio=resid_nontf_ratio,
        ntp_nontf_ratio=ntp_nontf_ratio,
        flag_separate_bos_representation=flag_separate_bos_representation,
        sid_start_token_id=sid_start_token_id,
        use_prototype=True,
        prototype_config={
            "K": K, "p": p, "L": L,
            "beta": beta, "delta": delta,
            "ema_decay": ema_decay,
            "n_heads": n_heads_proto,
            "gate_hidden": gate_hidden,
            "dropout": dropout_proto,
        },
    ).to(device)

    if not isinstance(codebook_sizes, int):
        test_model.set_codebook_offsets(codebook_sizes)

    test_model.load_state_dict(
        torch.load(best_state_path, map_location=device), strict=False
    )
    test_model.resid_nontf_ratio = 0.0
    test_model.ntp_nontf_ratio = 0.0

    logs, _ = evaluate_helper_residual(
        test_model,
        device,
        test_dataloader_dict,
        unseen_semantic_ids,
        all_semantic_ids,
        item2sid,
        item_embedding,
        method_config,
        keyword="test",
        RETRIEVE_KEY=RETRIEVE_KEY,
    )

    writer.log(logs)
    writer.finish()

    return
