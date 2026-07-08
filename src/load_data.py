"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pickle

import numpy as np
import torch
from tqdm import trange


def expand_id(id, codebook_sizes):
    """Expand raw codebook indices to unique token IDs.

    Args:
        id: list of raw codebook indices (0-based per level)
        codebook_sizes: list of codebook sizes per level (int or list[int]).
            If int, all levels use the same size (backward compatible).
            If list, each level uses its own size (needed for LETTER with
            variable num_emb_list like [256, 256, 256, 256]).

    Mapping: token_id[level_i] = raw_index[level_i] + sum(sizes[:i]) + 1
    Token 0 is reserved for padding, token 1..(sum+1) are for SID levels,
    and the last token is EOS.
    """
    if isinstance(codebook_sizes, int):
        # Backward compatible: uniform codebook size for all levels
        codebook_sizes = [codebook_sizes] * len(id)

    expanded_id = list(id)
    offset = 1  # token 0 is padding, so start from 1
    for i_codebook_depth in range(len(expanded_id)):
        expanded_id[i_codebook_depth] += offset
        offset += codebook_sizes[i_codebook_depth]
    return expanded_id


def expand_id_arr(id_arr, codebook_sizes):
    """Expand raw codebook indices array to unique token IDs (in-place).

    Args:
        id_arr: numpy array of shape [n_items, n_levels] with raw indices
        codebook_sizes: list of codebook sizes per level (int or list[int]).
            If int, all levels use the same size (backward compatible).
    """
    n_levels = id_arr.shape[1]
    if isinstance(codebook_sizes, int):
        # Backward compatible: uniform codebook size for all levels
        for i_codebook_depth in range(n_levels):
            id_arr[:, i_codebook_depth] += codebook_sizes * i_codebook_depth + 1
    else:
        # Variable codebook sizes: cumulative offset
        offset = 1  # token 0 is padding, start from 1
        for i_codebook_depth in range(n_levels):
            id_arr[:, i_codebook_depth] += offset
            offset += codebook_sizes[i_codebook_depth]


def pad_sequence(sequence, length, pad_token):
    assert (
        len(sequence) + 1 <= length
    ), "we want the max_sequence_length to include all SIDs and one extra eos token"
    return sequence + [pad_token] + [0] * (length - len(sequence) - 1)


def get_unique_semantic_ids_by_extra_position(semantic_ids, codebook_sizes): #增添额外的id，避免冲突
    """
    In this function, we load the semantic IDs, and then assign the extra one semantic id to avoid duplication.
    :param semantic_ids: numpy array of shape [n_items, n_levels]
    :param codebook_sizes: list of codebook sizes per level (int or list[int])
    """
    semantic_id_2_item_count = dict({})
    item_2_semantic_id, semantic_id_2_item = {}, {}

    # NOTE: the first three semantic ID range from 0 to codebook_size - 1
    # so the last one should also start from 0 to last_codebook_size - 1
    # otherwise, the max(4th semantic ID) may = user_id (in very rare case)
    for i in range(len(semantic_ids)):
        id = semantic_ids[i]  # list of length codebook_depth
        count_dict = semantic_id_2_item_count
        for i_depth in range(len(id)):
            if id[i_depth] not in count_dict.keys():
                count_dict[id[i_depth]] = dict({})
            count_dict = count_dict[id[i_depth]]
        id_dict = count_dict
        id_dict[len(id_dict)] = i + 1
        last_semantic_id = len(id_dict) - 1
        item_2_semantic_id[i + 1] = (*id, last_semantic_id)
        semantic_id_2_item[item_2_semantic_id[i + 1]] = i + 1
    # item_2_semantic_id: {item_id: [semantic_ids]}, item_id start from 1

    assert (
        len(item_2_semantic_id) == semantic_ids.shape[0]
    ), "Not all semanticid -> item collisions have been avoided!"

    # check if last dimension of semantic ids exceeds the number of available codebook entries
    all_semantic_ids_list = [
        item_2_semantic_id[idx] for idx in np.arange(1, len(semantic_ids) + 1)
    ]
    all_semantic_ids = np.array(all_semantic_ids_list)  # [n_items, n_codebook]
    max_last_semantic_ids = (
        all_semantic_ids[:, -1].max() + 1
    )  # this is actually the number of unique semantic id in the 4th position
    expand_id_arr(all_semantic_ids, codebook_sizes)
    return item_2_semantic_id, max_last_semantic_ids


def generate_input_sequence(
    user_id,
    user_sequence,
    item_2_semantic_id,
    max_items_per_seq,
    max_sequence_length,
    codebook_sizes,
    item_embedding,
    id_only,
):
    if user_id is not None:  # indicating that we are using user_id
        input_sids = [user_id]
        attention_mask_sids = [1]
        input_ids = [user_id]
        attention_mask_ids = [1]
    else:
        input_sids, input_ids, attention_mask_sids, attention_mask_ids = [], [], [], []

    input_embeddings, labels_sids, labels_ids, label_embeddings = [], [], [], []

    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            if not id_only:
                labels_sids.extend(
                    expand_id(item_2_semantic_id[user_sequence[i]], codebook_sizes)
                )
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            # item id starts from 1, 1 ~ n_item
            # item_embedding shape [n_item, 768]
            labels_ids.append(user_sequence[i])
            label_embeddings = this_item_embedding  # [1, 768]
        else:
            if not id_only:
                input_semantic_ids = expand_id(
                    item_2_semantic_id[user_sequence[i]], codebook_sizes
                )
                input_sids.extend(input_semantic_ids)
            input_ids.append(user_sequence[i])
            attention_mask_sids.extend([1] * len(input_semantic_ids))
            attention_mask_ids.append(1)
            this_item_embedding = item_embedding[[user_sequence[i] - 1]]
            input_embeddings.append(this_item_embedding)

    if not id_only:
        labels_sids = np.array(labels_sids)
        input_sids = np.array(
            pad_sequence(input_sids, max_sequence_length, pad_token=0)
        )
        attention_mask_sids = np.array(
            pad_sequence(attention_mask_sids, max_sequence_length, pad_token=0)
        )
        assert not np.any(labels_sids == 0)
        labels_sids[labels_sids == 0] = -100

    labels_ids = np.array(labels_ids)
    input_ids = np.array(pad_sequence(input_ids, max_sequence_length, pad_token=0))
    attention_mask_ids = np.array(
        pad_sequence(attention_mask_ids, max_sequence_length, pad_token=0)
    )
    input_embeddings = torch.cat(
        input_embeddings
        + [torch.zeros_like(input_embeddings[0])]
        * (max_items_per_seq - len(input_embeddings)),
        dim=0,
    )

    return (
        input_sids,
        input_ids,
        input_embeddings,
        attention_mask_sids,
        attention_mask_ids,
        labels_sids,
        labels_ids,
        label_embeddings,
    )


def load_data_helper(
    user_sequence,
    unseen_val,
    unseen_test,
    item_2_semantic_id,
    item_embedding,
    method_config,
    max_items_per_seq,
    max_sequence_length,
    user_id_offset,
    codebook_sizes,
    id_only=False,
):
    total_user_dict = {}
    for key in ["train", "val", "unseen_val", "test", "unseen_test"]:
        total_user_dict[key] = {
            "input_ids": [],
            "attention_mask_ids": [],
            "labels_ids": [],
            "input_embeddings": [],
            "label_embeddings": [],
        }
        if not id_only:
            total_user_dict[key]["input_sids"] = []
            total_user_dict[key]["labels_sids"] = []
            total_user_dict[key]["attention_mask_sids"] = []

    len_user_sequence = len(user_sequence)

    for i in trange(len_user_sequence):
        if not id_only and method_config["include_user_id"]:
            # use Hashing Trick to map the user to 2000 user tokens
            user_id = user_id_offset + i % 2000
        else:
            user_id = None

        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        for j in range(2, len(user_sequence[i]) + 1):
            this_sequence = user_sequence[i][:j]

            #最后一个是测试集，倒数第二个是测试集
            if j == len(user_sequence[i]) - 1:
                this_key = "unseen_val" if this_sequence[-1] in unseen_val else "val"
            elif j == len(user_sequence[i]):
                this_key = "unseen_test" if this_sequence[-1] in unseen_test else "test"
            else:
                this_key = "train"

            (
                input_sids,
                input_ids,
                input_embeddings,
                attention_mask_sids,
                attention_mask_ids,
                labels_sids,
                labels_ids,
                labels_embeddings,
            ) = generate_input_sequence(
                user_id,
                this_sequence,
                item_2_semantic_id,
                max_items_per_seq,
                max_sequence_length,
                codebook_sizes,
                item_embedding,
                id_only,
            )
            if not id_only:
                total_user_dict[this_key]["input_sids"].append(input_sids)
                total_user_dict[this_key]["attention_mask_sids"].append(
                    attention_mask_sids
                )
                total_user_dict[this_key]["labels_sids"].append(labels_sids)

            total_user_dict[this_key]["input_ids"].append(input_ids)
            total_user_dict[this_key]["input_embeddings"].append(input_embeddings.cpu())
            total_user_dict[this_key]["attention_mask_ids"].append(attention_mask_ids)
            total_user_dict[this_key]["labels_ids"].append(labels_ids)
            total_user_dict[this_key]["label_embeddings"].append(
                labels_embeddings.cpu()
            )

    for key in total_user_dict.keys():
        for sub_key in total_user_dict[key].keys():
            if sub_key in ["label_embeddings"]:
                if len(total_user_dict[key][sub_key]) > 0:
                    total_user_dict[key][sub_key] = torch.cat(
                        total_user_dict[key][sub_key]
                    )
                else:
                    total_user_dict[key][sub_key] = torch.tensor([])
            elif sub_key in ["input_embeddings"]:
                if len(total_user_dict[key][sub_key]) > 0:
                    total_user_dict[key][sub_key] = torch.stack(
                        total_user_dict[key][sub_key]
                    )
                else:
                    total_user_dict[key][sub_key] = torch.tensor([])
            else:
                total_user_dict[key][sub_key] = torch.tensor(
                    total_user_dict[key][sub_key], dtype=torch.long
                )

    return (
        total_user_dict["train"],
        total_user_dict["val"],
        total_user_dict["unseen_val"],
        total_user_dict["test"],
        total_user_dict["unseen_test"],
    )


def load_data(
    path,
    user_sequence,
    unseen_val,
    unseen_test,
    seen,
    item_embedding,
    method_config,
    max_length=258,
    codebook_sizes=None,
    max_items_per_seq=np.inf,
):
    """
    :param path: path to load the semantic ID
    :param user_sequence: user sequence
    :param unseen_val, unseen_test, seen: semantic ID np.array
    :param item_embedding: [n_item, n_embd], where n_embd is sentence-T5 embedding dimension
    :param max_length: for the generated sequence
    :param codebook_sizes: list of codebook sizes per level (int or list[int]).
        If None, uses uniform codebook_size=256 (backward compatible).
    :param max_items_per_seq: number of items in each sequence
    """
    n_semantic_codebook = 3

    semantic_ids = pickle.load(open(path, "rb"))

    # Dynamically determine n_semantic_codebook from the actual data shape
    # LETTER tokenizer can produce different numbers of codebook levels (e.g., 4)
    if isinstance(semantic_ids, np.ndarray) and semantic_ids.ndim == 2:
        n_semantic_codebook = semantic_ids.shape[1]

    # Default: uniform codebook size of 256 for all levels (backward compatible)
    if codebook_sizes is None:
        codebook_sizes = 256

    item_2_semantic_id, max_last_semantic_ids = (
        get_unique_semantic_ids_by_extra_position(semantic_ids, codebook_sizes)
    )

    # Compute last_codebook_size based on whether codebook_sizes is a list or int
    if isinstance(codebook_sizes, int):
        last_codebook_size = max(max_last_semantic_ids, codebook_sizes)
    else:
        last_codebook_size = max(max_last_semantic_ids, max(codebook_sizes))

    n_codebook = n_semantic_codebook + 1  # actual number of codebook

    # split into train/val/test items
    val_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in unseen_val])
    expand_id_arr(val_unseen_semantic_ids, codebook_sizes)
    test_unseen_semantic_ids = np.array(
        [item_2_semantic_id[idx] for idx in unseen_test]
    )
    expand_id_arr(test_unseen_semantic_ids, codebook_sizes)
    seen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in seen])
    expand_id_arr(seen_semantic_ids, codebook_sizes)
    all_semantic_ids = np.array(
        [item_2_semantic_id[idx] for idx in item_2_semantic_id.keys()]
    )
    expand_id_arr(all_semantic_ids, codebook_sizes)

    # get semantic id - embedding dict
    semantic_id_2_embd = dict({})
    for key in item_2_semantic_id.keys():  # again, the key here start from 1
        semantic_id_2_embd[tuple(expand_id(item_2_semantic_id[key], codebook_sizes))] = (
            item_embedding[key - 1]
        )

    # Compute user_id_offset: 1 (padding) + sum of all codebook level sizes + last_codebook_size
    if isinstance(codebook_sizes, int):
        user_id_offset = 1 + n_semantic_codebook * codebook_sizes + last_codebook_size
    else:
        user_id_offset = 1 + sum(codebook_sizes) + last_codebook_size
    training_data, val_data, unseen_val_data, test_data, unseen_test_data = (
        load_data_helper(
            user_sequence,
            unseen_val,
            unseen_test,
            item_2_semantic_id,
            item_embedding,
            method_config,
            max_items_per_seq,
            max_length,
            user_id_offset,
            codebook_sizes,
        )
    )

    return (
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
        all_semantic_ids,
    )


def load_data_id(
    user_sequence,
    unseen_val,
    unseen_test,
    item_embedding,
    method_config,
    max_length=258,
    max_items_per_seq=np.inf,
):

    training_data, val_data, unseen_val_data, test_data, unseen_test_data = (
        load_data_helper(
            user_sequence,
            unseen_val,
            unseen_test,
            None,
            item_embedding,
            method_config,
            max_items_per_seq,
            max_length,
            None,
            None,
            id_only=True,
        )
    )

    return training_data, val_data, test_data, unseen_val_data, unseen_test_data
