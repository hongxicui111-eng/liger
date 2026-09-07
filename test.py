"""
独立测试脚本：直接指定已训练好的模型路径进行评测。
用法示例：
    python test.py \
        dataset=amazon \
        dataset.name=Beauty \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        experiment_id=tiger_Beauty_xxx \
        test.ckpt_path=./results/tiger/Amazon_Beauty/tiger_Beauty_xxx_seed_42/results/ckpt_best.pt
"""

import os
import sys
import traceback
from collections import Counter

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import T5Config

from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from run import set_dir
from src.evaluation import (
    ITEM_GROUP_LABELS,
    USER_GROUP_LABELS,
    evaluate,
    evaluate_dense_ids,
    evaluate_dense_sids,
    generate_then_dense,
)
from src.load_data import load_data
from src.tiger import TIGER
from src.training import _log_group_results, evaluate_helper
from utils import CustomDataset, set_seed, setup_logging


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
    config["logging"]["project"] = "liger"
    is_steam = config["dataset"]["type"] == "steam"

    # ── 解析 ckpt 路径 ────────────────────────────────────────────────────────
    # 优先从命令行参数 test.ckpt_path 读取，否则用默认的 ckpt_best.pt 路径
    ckpt_path = config.get("test", {}).get(
        "ckpt_path",
        os.path.join(config["output_path"], "results", "ckpt_best.pt"),
    )
    assert os.path.exists(ckpt_path), (
        f"Checkpoint not found: {ckpt_path}\n"
        f"请通过 test.ckpt_path=<path> 指定模型路径"
    )
    print(f"\n使用模型: {ckpt_path}")

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

        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        # RQ-VAE SID（如已存在直接复用，否则重新训练）
        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
        )

        # ── 加载数据 ──────────────────────────────────────────────────────────
        codebook_size   = train_config["RQ-VAE"]["code_book_size"]
        max_items_per_seq = train_config["max_items_per_seq"]
        tiger_config    = train_config["TIGER"]
        trainer_config  = tiger_config["trainer"]
        eval_batch_size = trainer_config["eval_batch_size"]

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
            item_embedding,
            method_config,
            max_length=tiger_config["n_positions"],
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

        # ── 构建物品频次数组（用于分组评测）────────────────────────────────────
        _item_counter = Counter()
        for seq in user_sequence:
            for item in seq[:-2]:
                _item_counter[item] += 1
        n_items = item2sid.shape[0]
        item_freq_arr = np.zeros(n_items, dtype=np.int32)
        for item_id, freq in _item_counter.items():
            if 1 <= item_id <= n_items:
                item_freq_arr[item_id - 1] = freq

        if method_config["flag_use_output_embedding"]:
            item_embedding = item_embedding.to(device)

        from torch.utils.data import DataLoader
        from utils import CustomDataset

        test_dataset          = CustomDataset(test_data)
        unseen_test_dataset   = CustomDataset(unseen_test_data)
        test_dataloader       = DataLoader(test_dataset,        batch_size=eval_batch_size, shuffle=False)
        test_dataloader_embd  = DataLoader(test_dataset,        batch_size=eval_batch_size, shuffle=False)
        unseen_test_dataloader = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)
        unseen_test_dataloader_embd = DataLoader(unseen_test_dataset, batch_size=eval_batch_size, shuffle=False)

        test_dataloader_dict = {
            "in_set":         test_dataloader,
            "in_set_embd":    test_dataloader_embd,
            "cold_start":     unseen_test_dataloader,
            "cold_start_embd": unseen_test_dataloader_embd,
        }

        all_semantic_ids  = torch.from_numpy(all_semantic_ids)
        unseen_semantic_ids = torch.from_numpy(unseen_semantic_ids)

        # ── 构建模型 ──────────────────────────────────────────────────────────
        last_codebook_size = max(max_last_semantic_ids, codebook_size)
        if method_config["include_user_id"]:
            this_vocab_size = (
                2000 + codebook_size * n_semantic_codebook + last_codebook_size + 2
            )
        else:
            this_vocab_size = codebook_size * n_semantic_codebook + last_codebook_size + 2

        if method_config["use_id"] == "item_id":
            this_vocab_size = item_embedding.shape[0] + 2

        t5_cfg = tiger_config["T5"]
        model_config = T5Config(
            num_layers=t5_cfg["encoder_layers"],
            num_decoder_layers=t5_cfg["decoder_layers"],
            d_model=t5_cfg["d_model"],
            d_ff=t5_cfg["d_ff"],
            num_heads=t5_cfg["num_heads"],
            d_kv=t5_cfg["d_kv"],
            dropout_rate=t5_cfg["dropout_rate"],
            vocab_size=this_vocab_size,
            pad_token_id=0,
            eos_token_id=int(this_vocab_size - 1),
            decoder_start_token_id=0,
            feed_forward_proj=t5_cfg["feed_forward_proj"],
            n_positions=tiger_config["n_positions"],
            layer_norm_epsilon=1e-8,
            initializer_factor=t5_cfg["initializer_factor"],
        )

        model = TIGER(
            config=model_config,
            n_semantic_codebook=n_semantic_codebook,
            max_items_per_seq=max_items_per_seq,
            flag_use_output_embedding=method_config["flag_use_output_embedding"],
            flag_use_learnable_text_embed=method_config["flag_add_input_embedding"],
            embedding_head_dict=method_config["embedding_head_dict"],
        ).to(device)

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        # 兼容两种保存格式：纯 state_dict 或含 model_state_dict 的 training_state
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print("模型加载成功")

        if (
            method_config["embedding_loss_weight"] > 0
            and method_config["sid_loss_weight"] > 0
        ):
            RETRIEVE_KEY = [20, 40, 60, 80, 100]
        else:
            RETRIEVE_KEY = [10]

        # ── 评测 ──────────────────────────────────────────────────────────────
        writer = setup_logging(config)
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
            item_freq_arr=item_freq_arr,
        )

        # 打印整体结果
        print("\n" + "=" * 55)
        print("  整体评测结果")
        print("=" * 55)
        for k, v in sorted(logs.items()):
            if "group" not in k:
                print(f"  {k}: {float(v):.4f}")

        writer.log(logs)
        writer.finish()

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
