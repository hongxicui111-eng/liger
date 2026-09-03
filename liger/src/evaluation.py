"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList


def model_forward(model, batch, device, n_codebook, method_config, skip_forward=False):
    # Effective SID depth for encoder input items. When input_sid_depth > 0,
    # each input item is represented by only the first N SIDs (shorter encoder
    # sequence). 0 = use all n_codebook SIDs (backward compatible).
    input_sid_depth = method_config.get("input_sid_depth", 0)
    enc_sid_depth = input_sid_depth if input_sid_depth > 0 else n_codebook

    if method_config["use_id"] == "sid":
        input_sids = batch["input_sids"].to(device)
        attention_mask_sids = batch["attention_mask_sids"].to(device)
        labels_sids = batch["labels_sids"].to(device)
        if method_config["flag_add_input_embedding"]:
            input_text_embeddings = batch["input_embeddings"].to(device).detach()
            max_length = input_text_embeddings.shape[1]

        item_idx_start = 1 if method_config["include_user_id"] else 0
        # Model forwarding
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            if method_config["flag_add_input_embedding"]:
                inputs_embeds = model.shared(input_sids)
                # process the input text embeddings
                input_text_embeddings_shape = (
                    input_text_embeddings.shape
                )  # this is the original embedding
                input_text_embeddings_repeat = (
                    input_text_embeddings[:, :, None, :]
                    .repeat(1, 1, enc_sid_depth, 1)
                    .reshape(-1, input_text_embeddings_shape[-1])
                )
                proj_embd = model.emb_proj(input_text_embeddings_repeat)
                proj_embd = proj_embd.reshape(
                    input_text_embeddings_shape[0], -1, proj_embd.shape[-1]
                )
                # add positional embedding
                pos_id = torch.arange(max_length, dtype=torch.long, device=device)
                pos_id = pos_id[:, None].repeat(1, enc_sid_depth).reshape(-1)
                pos_embd = model.pos_embedding(pos_id)  # [n_seq, n_embd]
                proj_embd += pos_embd[None, :]
                append_embedding = torch.zeros_like(inputs_embeds)
                append_embedding[
                    :, item_idx_start : item_idx_start + enc_sid_depth * max_length
                ] = proj_embd
                # add the text embedding to the inputs embeds
                inputs_embeds += append_embedding
                # add the semantic positional embedding
                seq_len = inputs_embeds.shape[1]
                pattern = torch.arange(enc_sid_depth)
                semantic_pos = pattern.repeat(seq_len // enc_sid_depth + 1)[:seq_len]
                pos_embedding = model.semantic_pos(
                    semantic_pos.to(device)
                )  # [n_seq, n_embd]
                inputs_embeds += pos_embedding[None, :, :]
                # process the inputs embeds
                inputs_embeds = model.input_embed_layernorm(inputs_embeds)
                inputs_embeds = model.input_embed_dropout(inputs_embeds)

                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask_sids,
                        labels=labels_sids,
                    )

                input_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask_sids,
                }
            else:
                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        input_ids=input_sids,
                        attention_mask=attention_mask_sids,
                        labels=labels_sids,
                    )

                input_kwargs = {
                    "input_ids": input_sids,
                    "attention_mask": attention_mask_sids,
                }
    else:
        input_ids = batch["input_ids"].to(device)
        attention_mask_ids = batch["attention_mask_ids"].to(device)
        labels_ids = batch["labels_ids"].to(device)
        if method_config["flag_add_input_embedding"]:
            input_embeddings = batch["input_embeddings"].to(device).detach()
            max_length = input_embeddings.shape[1]

        item_idx_start = 0
        # Model forwarding
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            if method_config["flag_add_input_embedding"]:
                inputs_embeds = model.shared(input_ids)
                # process the input text embeddings
                input_embeddings_shape = input_embeddings.shape
                proj_embd = model.emb_proj(
                    input_embeddings.reshape(
                        input_embeddings_shape[0] * input_embeddings_shape[1],
                        input_embeddings_shape[2],
                    )
                )
                proj_embd = proj_embd.reshape(
                    input_embeddings_shape[0], input_embeddings_shape[1], -1
                )
                # add positional embedding
                pos_id = torch.arange(max_length, dtype=torch.long, device=device)
                pos_embd = model.pos_embedding(pos_id)  # [n_seq, n_embd]
                proj_embd += pos_embd[None, :]
                append_embedding = torch.zeros_like(inputs_embeds)
                append_embedding[:, item_idx_start : item_idx_start + max_length] = (
                    proj_embd
                )
                inputs_embeds += append_embedding

                inputs_embeds = model.input_embed_layernorm(inputs_embeds)
                inputs_embeds = model.input_embed_dropout(inputs_embeds)

                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask_ids,
                        labels=labels_ids,
                    )

                input_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask_ids,
                }
            else:
                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_ids,
                        labels=labels_ids,
                    )

                input_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask_ids,
                }
    return outputs, input_kwargs


def dcg_torch(scores: torch.Tensor):
    """Compute DCG using gain = 2^rel - 1."""
    scores = scores.float()
    device = scores.device
    gains = torch.pow(2.0, scores) - 1
    discounts = torch.log2(torch.arange(2, scores.size(0) + 2, device=device).float())
    return torch.sum(gains / discounts)


def ndcg_at_k_torch(r: torch.Tensor, k: int):
    """Compute NDCG at rank k from relevance vector (1D torch.Tensor)"""
    r = r[:k].float()
    dcg_val = dcg_torch(r)
    ideal_r, _ = torch.sort(r, descending=True)
    dcg_max = dcg_torch(ideal_r)

    if dcg_max == 0:
        return torch.tensor(0.0, device=r.device)
    return dcg_val / dcg_max


def calculate_metrics(outputs, labels, KEYS, codebook_level=4, lift_constraint=False):
    """
    n_codebook: the number of semantic id for each item
    :param outputs: shape [batch_size, num_return_seq, n_codebook]
    :param labels: shape [batch_size, n_codebook]
    """
    batch_size = len(outputs)

    ndcg_at_i, recall_at_i = dict({}), dict({})
    for key in KEYS:
        ndcg_at_i[key] = []
        recall_at_i[key] = []

    if not lift_constraint:
        for i in range(batch_size):
            assert (
                torch.unique(outputs[i], dim=0).shape[0] == outputs.shape[1]
            ), "Unless something is wrong, beam search should not return non-unique outputs."

    matches = (outputs == labels[:, None, :codebook_level]).all(axis=-1)
    for key in KEYS:
        recall_at_i[key] = matches[:, :key].any(-1).float().tolist()  # [batch_size]

    for i in range(batch_size):
        for key in KEYS:
            ndcg_at_i[key].append(ndcg_at_k_torch(matches[i], key).item())

    metrics = (
        recall_at_i,
        ndcg_at_i,
    )

    return metrics


def calculate_metrics_id(outputs, labels, KEYS):
    """
    n_codebook: the number of semantic id for each item
    :param outputs: shape [batch_size, num_return_seq]
    :param labels: shape [batch_size,]
    """
    batch_size = len(outputs)

    ndcg_at_i, recall_at_i = dict({}), dict({})
    for key in KEYS:
        ndcg_at_i[key] = []
        recall_at_i[key] = []

    matches = outputs == labels
    for key in KEYS:
        recall_at_i[key] = matches[:, :key].any(-1).float().tolist()  # [batch_size]

    for i in range(batch_size):
        for key in KEYS:
            ndcg_at_i[key].append(ndcg_at_k_torch(matches[i], key).item())

    # Calculate mean metrics
    metrics = (
        recall_at_i,
        ndcg_at_i,
    )

    return metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    all_semantic_ids,
    device,
    method_config,
    KEYS,
    RETRIEVE_KEY,
):

    model.eval()
    recall_dict, ndcg_dict = dict({}), dict({})
    returned_cand = []
    returned_predicted_embedding = []

    for batch in tqdm(dataloader):
        labels = batch["labels_sids"].to(device)

        with torch.no_grad():
            _, input_kwargs = model_forward(
                model,
                batch,
                device,
                all_semantic_ids.shape[-1],
                method_config,
                skip_forward=True,
            )

        batch_size, n_codebook = (
            labels.shape[0],
            labels.shape[1],
        )  # this batch_size is before ddp

        num_return_sequences = max(RETRIEVE_KEY)
        num_beams = max(RETRIEVE_KEY)
        assert (
            max(KEYS) <= num_return_sequences
        ), "The number of return sequences should be greater than or equal to the number of keys."
        gen_kwargs = {
            "num_beams": num_beams,
            "max_new_tokens": n_codebook,
            "num_return_sequences": num_return_sequences,
        }
        gen_kwargs["use_cache"] = True

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**input_kwargs, **gen_kwargs)
        predicted_embedding = model.predicted_embedding

        outputs = outputs[:, 1 : 1 + n_codebook].reshape(
            batch_size, num_return_sequences, -1
        )  # [B, n_return_seq, n_codebook]
        if predicted_embedding is not None:
            returned_predicted_embedding.extend(
                predicted_embedding.reshape(batch_size, num_return_sequences, -1)[:, 0]
            )  # [B, n_embd]
        # along the num_return_sequences, all the outputs are the same

        if outputs.shape[-1] < n_codebook:
            # if output shape is smaller than label shape, pad with zeros to label shape
            # can happen when the LM predicts eos early
            # if padded with zero, the remaining prediction should measure the recall / ndcg as 0.
            to_pad = n_codebook - outputs.shape[-1]
            pad_tensor = torch.zeros(
                (batch_size, gen_kwargs["num_return_sequences"], to_pad),
                device=outputs.device,
            )
            outputs = torch.cat([outputs, pad_tensor], dim=-1)

        outputs = outputs
        returned_cand.extend(outputs)

        _recall_at_i, _ndcg_at_i = calculate_metrics(
            outputs, labels, codebook_level=n_codebook, KEYS=KEYS
        )

        for key in _recall_at_i.keys():
            if key not in recall_dict.keys():
                recall_dict[key] = []
                ndcg_dict[key] = []

        for key in _recall_at_i.keys():
            recall_dict[key].extend(_recall_at_i[key])
            ndcg_dict[key].extend(_ndcg_at_i[key])

    return recall_dict, ndcg_dict, returned_cand, returned_predicted_embedding


def get_target_embed(predicted_embedding, model, method_config, item_embedding):
    if len(item_embedding.shape) == 3:  # this is used in generate_then_dense
        assert (
            method_config["embedding_head_dict"]["embed_target"]
            != "ground_truth+item_id"
        ), "We don't expect the use of dense retrieval with item id in liger"
        item_embedding_shape = (
            item_embedding.shape
        )  # [batch_size, num_candidate, n_embd]
        item_embedding = item_embedding.reshape(
            item_embedding_shape[0] * item_embedding_shape[1], item_embedding_shape[2]
        )
        _item_embedding = model.emb_proj(item_embedding)
        _item_embedding = _item_embedding.reshape(
            item_embedding_shape[0], item_embedding_shape[1], -1
        )
    else:
        _item_embedding = model.emb_proj(item_embedding)
        if (
            method_config["embedding_head_dict"]["embed_target"]
            == "ground_truth+item_id"
        ):
            learned_embedding = model.shared.weight[1:-1]
            _item_embedding += learned_embedding

    logits = None
    if predicted_embedding is not None:
        temperature = 1.0
        if method_config["embedding_head_dict"]["normalize_logits"]:
            temperature = method_config["embedding_head_dict"]["logits_temperature"]

        # Similarity metric: "cosine" (default) or "dot" (raw dot-product).
        # "dot" is useful when the item embeddings (e.g. SASRec CF) were
        # trained with dot-product and their norm carries signal that
        # cosine normalization would discard.
        sim_metric = method_config.get("similarity_metric", "cosine")

        if sim_metric == "dot":
            logits = (
                predicted_embedding[:, None, :]
                * _item_embedding.type(predicted_embedding.dtype)
            ).sum(-1) / temperature
        else:  # cosine (default)
            predicted_embedding = F.normalize(predicted_embedding, dim=1)
            _item_embedding = F.normalize(_item_embedding, dim=-1)
            logits = (
                predicted_embedding[:, None, :]
                * _item_embedding.type(predicted_embedding.dtype)
            ).sum(-1) / temperature

    return _item_embedding, logits


@torch.no_grad()
def evaluate_dense_sids(
    model,
    dataloader,
    device,
    item2sid,
    item_embedding,
    method_config,
    KEYS=[
        10,
    ],
):

    model.eval()
    recall_dict, ndcg_dict = dict({}), dict({})
    item2sid_tensor = torch.from_numpy(item2sid).to(device)

    if len(dataloader) == 0:
        return recall_dict, ndcg_dict

    num_candidates = max(KEYS)

    for batch in tqdm(dataloader, desc="Dense Retrieval"):
        labels = batch["labels_sids"].to(device)
        n_codebook = labels.shape[1]

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, _ = model_forward(
                    model,
                    batch,
                    device,
                    n_codebook,
                    method_config,
                )
                predicted_embedding = model.predicted_embedding

        _, logits = get_target_embed(
            predicted_embedding, model, method_config, item_embedding
        )
        # [batch_size, n_items]

        candidate_idx = logits.topk(num_candidates, dim=1, largest=True)[
            1
        ]  # [batch_size, num_candidates]
        candidate_sid = item2sid_tensor[
            candidate_idx
        ]  # [batch_size, num_candidates, n_codebook]
        constrained_outputs = candidate_sid

        _recall_at_i, _ndcg_at_i = calculate_metrics(
            constrained_outputs,
            labels,
            codebook_level=n_codebook,
            KEYS=KEYS,
        )
        for key in _recall_at_i.keys():
            if key not in recall_dict.keys():
                recall_dict[key] = []
                ndcg_dict[key] = []

        for key in _recall_at_i.keys():
            recall_dict[key].extend(_recall_at_i[key])
            ndcg_dict[key].extend(_ndcg_at_i[key])

    return recall_dict, ndcg_dict


@torch.no_grad()
def evaluate_dense_ids(
    model,
    dataloader,
    device,
    item2sid,
    item_embedding,
    method_config,
    KEYS=[
        10,
    ],
):

    model.eval()
    recall_dict, ndcg_dict = dict({}), dict({})

    if len(dataloader) == 0:
        return recall_dict, ndcg_dict

    num_candidates = max(KEYS)

    for batch in dataloader:
        labels = batch["labels_ids"].to(device)
        n_codebook = labels.shape[1]

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = model_forward(
                    model,
                    batch,
                    device,
                    n_codebook,
                    method_config,
                )
                predicted_embedding = model.predicted_embedding

        _, logits = get_target_embed(
            predicted_embedding, model, method_config, item_embedding
        )
        # [batch_size, n_items]

        candidate_idx = logits.topk(num_candidates, dim=1, largest=True)[
            1
        ]  # [batch_size, num_candidates]

        labels = labels.cpu()
        constrained_outputs = candidate_idx + 1

        _recall_at_i, _ndcg_at_i = calculate_metrics_id(
            constrained_outputs.cpu(), labels, KEYS=KEYS
        )
        for key in _recall_at_i.keys():
            if key not in recall_dict.keys():
                recall_dict[key] = []
                ndcg_dict[key] = []

        for key in _recall_at_i.keys():
            recall_dict[key].extend(_recall_at_i[key])
            ndcg_dict[key].extend(_ndcg_at_i[key])

    return recall_dict, ndcg_dict


@torch.no_grad()
def generate_then_dense(
    model,
    dataloader,
    unseen_semantic_ids,
    device,
    method_config,
    returned_cand,
    returned_embd,
    item2sid,
    item_embedding,
    KEYS=[
        10,
    ],
    RETRIEVE_KEY=[20, 40, 60, 80, 100],
    include_cold=True,
):
    """Generate-then-dense unified evaluation.

    When ``include_cold=True`` (default, standard behaviour), the cold-start
    SIDs are appended to the generative candidates before dense re-ranking —
    this is the original Liger behaviour.  When ``include_cold=False``, the
    candidate pool consists of *only* the generative (beam-search) SIDs,
    giving a **NoCold** variant that isolates the generative retrieval quality
    without the cold-start safety net.
    """

    model.eval()
    recall_dict_total, ndcg_dict_total = dict({}), dict({})
    if len(returned_cand) == 0:
        return recall_dict_total, ndcg_dict_total

    returned_cand = torch.stack(returned_cand)  # [num_items, num_candidate, n_code]
    item2sid_tensor = torch.from_numpy(item2sid).to(device)
    unseen_semantic_ids = unseen_semantic_ids.to(device)
    returned_embd = torch.stack(returned_embd, dim=0)  # [num_items, n_embd]
    num_candidates = max(KEYS)

    idx_start = 0
    for batch in tqdm(
        dataloader,
        desc="Generating and Dense Retrieval"
        + ("" if include_cold else " (NoCold)"),
    ):
        labels = batch["labels_sids"].to(device)
        batch_size, n_codebook = (
            labels.shape[0],
            labels.shape[1],
        )  # this batch_size is before ddp

        predicted_embedding = returned_embd[idx_start : idx_start + batch_size]
        this_batch_cand = returned_cand[
            idx_start : idx_start + batch_size,
        ]  # [batch_size, num_candidate, n_code]
        idx_start += batch_size
        if include_cold:
            cold_cand = torch.stack(
                [unseen_semantic_ids] * this_batch_cand.shape[0], dim=0
            )

        for _retrieve_key in RETRIEVE_KEY:
            if _retrieve_key not in recall_dict_total.keys():
                recall_dict_total[_retrieve_key] = dict({})
                ndcg_dict_total[_retrieve_key] = dict({})

            _this_batch_cand = this_batch_cand[:, :_retrieve_key]
            if include_cold:
                _this_batch_cand = torch.cat([_this_batch_cand, cold_cand], dim=1)
            matches = torch.all(
                _this_batch_cand[:, :, None] == item2sid_tensor[None, None, :, :],
                dim=-1,
            )
            # [batch_size, num_candidate, num_items]
            indices = torch.argmax(matches.int(), dim=2)  # [batch_size, num_candidate]

            cand_item_embedding = item_embedding[
                indices
            ]  # [batch_size, num_candidate_item_embedding]

            _, logits = get_target_embed(
                predicted_embedding, model, method_config, cand_item_embedding
            )
            # [batch_size, n_items]

            _topk = min(num_candidates, logits.shape[1])
            candidate_idx = logits.topk(_topk, dim=1, largest=True)[
                1
            ]  # [batch_size, num_candidates]
            candidate_sid = _this_batch_cand[
                torch.arange(batch_size)[:, None], candidate_idx
            ]  # [batch_size, num_candidates, n_codebook] -> [batch_size, topk, n_codebook]

            _recall_at_i, _ndcg_at_i = calculate_metrics(
                candidate_sid,
                labels,
                codebook_level=n_codebook,
                KEYS=KEYS,
                lift_constraint=True,
            )
            # lift the constraint to have unique candidate, since we append the output from generative retrieval with the cold start candidates
            for key in _recall_at_i.keys():
                if key not in recall_dict_total[_retrieve_key].keys():
                    recall_dict_total[_retrieve_key][key] = []
                    ndcg_dict_total[_retrieve_key][key] = []

            for key in _recall_at_i.keys():
                recall_dict_total[_retrieve_key][key].extend(_recall_at_i[key])
                ndcg_dict_total[_retrieve_key][key].extend(_ndcg_at_i[key])

    return recall_dict_total, ndcg_dict_total


@torch.no_grad()
def evaluate_prefix_then_dense(
    model,
    dataloader,
    device,
    item2sid,
    item_embedding,
    method_config,
    prefix2items,
    KEYS=[10],
    num_candidates_list=None,
    add_cold_items=None,
):
    """
    Hybrid evaluation: generate a prefix of length ``prefix_depth`` with the
    autoregressive model, then dense-retrieve the target item among the items
    that share that prefix (looked up from ``prefix2items``).

    The candidate set for a sample is the *union* of the item-sets of all the
    generated (beam) prefixes. The dense retriever ranks this candidate set
    using the encoder's predicted embedding, and the top-K items are compared
    to the ground-truth target item (item-level Recall@K / NDCG@K).

    ``num_candidates_list`` controls how many beam prefixes are used per
    sample.  When a list is given (e.g. ``[20, 40, 60, 80, 100]``), the
    function evaluates with each candidate count and returns nested dicts
    ``recall_dict[n][key]`` — mirroring the ``Gen{n}_Recall@{key}`` logging
    of Liger's ``generate_then_dense``.

    When ``add_cold_items`` is provided (a 1-D LongTensor of 0-based item
    indices for cold-start / unseen items), the function *also* computes an
    **AddCold** variant for each candidate count: the candidate set becomes
    ``prefix_items ∪ add_cold_items`` (deduplicated), and metrics are
    collected into ``recall_dict_cold`` / ``ndcg_dict_cold``.

    :param item2sid: [n_items, n_codebook] expanded semantic ids, ``item2sid[i]``
                     are the sids of item ``i+1``.
    :param prefix2items: dict ``tuple(prefix)`` -> list of 0-based item indices.
    :param item_embedding: [n_items, n_embd] raw sentence embeddings (on device).
    :param num_candidates_list: list of beam-prefix counts to evaluate with.
           Defaults to ``[100]`` for backward compatibility.
    :param add_cold_items: optional 1-D LongTensor of 0-based cold-start item
           indices. When provided, also computes AddCold metrics.
    :return: ``(recall_dict, ndcg_dict, recall_dict_cold, ndcg_dict_cold)``.
             The cold dicts are empty when ``add_cold_items`` is None.
    """
    # Normalise to a list for uniform handling
    if num_candidates_list is None:
        num_candidates_list = [100]
    if isinstance(num_candidates_list, int):
        num_candidates_list = [num_candidates_list]

    model.eval()
    # Nested dicts: recall_dict[n_candidates][key] -> list[float]
    recall_dict, ndcg_dict = dict({}), dict({})
    for n in num_candidates_list:
        recall_dict[n] = {key: [] for key in KEYS}
        ndcg_dict[n] = {key: [] for key in KEYS}

    # AddCold variant dicts (empty when add_cold_items is not provided)
    recall_dict_cold, ndcg_dict_cold = dict({}), dict({})
    if add_cold_items is not None:
        add_cold_items = add_cold_items.to(device)
        for n in num_candidates_list:
            recall_dict_cold[n] = {key: [] for key in KEYS}
            ndcg_dict_cold[n] = {key: [] for key in KEYS}

    if len(dataloader) == 0:
        return recall_dict, ndcg_dict, recall_dict_cold, ndcg_dict_cold

    prefix_depth = method_config["prefix_depth"]
    n_codebook = item2sid.shape[1]
    assert prefix_depth <= n_codebook, (
        f"prefix_depth ({prefix_depth}) must be <= n_codebook ({n_codebook})"
    )

    # generate enough beams for the largest candidate count
    max_candidates = max(num_candidates_list)
    num_return_sequences = max(max_candidates, max(KEYS))
    num_beams = num_return_sequences

    # pre-convert the prefix -> item-index lists to device tensors for fast lookup
    prefix2items_tensor = {
        prefix: torch.tensor(items, dtype=torch.long, device=device)
        for prefix, items in prefix2items.items()
    }

    for batch in tqdm(dataloader, desc="Prefix-then-Dense"):
        labels_sids = batch["labels_sids"].to(device)  # [B, n_codebook]
        labels_ids = batch["labels_ids"].to(device)  # [B, 1]
        batch_size = labels_sids.shape[0]

        with torch.no_grad():
            _, input_kwargs = model_forward(
                model,
                batch,
                device,
                n_codebook,
                method_config,
                skip_forward=True,
            )

        gen_kwargs = {
            "num_beams": num_beams,
            "max_new_tokens": prefix_depth,
            "num_return_sequences": num_return_sequences,
            "use_cache": True,
        }
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**input_kwargs, **gen_kwargs)
        predicted_embedding = model.predicted_embedding

        # generated prefixes: drop the decoder-start token, take prefix_depth tokens.
        # pad with 0 (pad token) if the model emitted eos early so the prefix has a
        # fixed length; padded prefixes will not match any real prefix -> skipped.
        gen_prefixes = outputs[:, 1 : 1 + prefix_depth].reshape(
            batch_size, num_return_sequences, -1
        )  # [B, num_return_sequences, <=prefix_depth]
        if gen_prefixes.shape[-1] < prefix_depth:
            to_pad = prefix_depth - gen_prefixes.shape[-1]
            pad_tensor = torch.zeros(
                (batch_size, num_return_sequences, to_pad),
                dtype=gen_prefixes.dtype,
                device=gen_prefixes.device,
            )
            gen_prefixes = torch.cat([gen_prefixes, pad_tensor], dim=-1)
        # now [B, num_return_sequences, prefix_depth]

        # predicted_embedding is the encoder representation of the input sequence;
        # it does not depend on the generated tokens, so take the first return seq.
        if predicted_embedding is not None:
            predicted_embedding = predicted_embedding.reshape(
                batch_size, num_return_sequences, -1
            )[:, 0]  # [B, n_embd]

        for b in range(batch_size):
            target_item = labels_ids[b, 0].item() - 1  # 0-based target item index
            pred_emb = predicted_embedding[b]  # [n_embd]

            # Pre-resolve prefix -> item tensor for every generated prefix
            # so each candidate-count slice can reuse the same lookups.
            prefix_items = []  # list of tensor or None
            for r in range(num_return_sequences):
                prefix = tuple(int(x) for x in gen_prefixes[b, r].tolist())
                prefix_items.append(prefix2items_tensor.get(prefix))

            for n in num_candidates_list:
                # collect candidate items from the first n prefixes
                cand_item_tensors = [
                    prefix_items[r] for r in range(n) if prefix_items[r] is not None
                ]
                has_prefix = len(cand_item_tensors) > 0

                # ---- Standard (prefix items only) ----
                if not has_prefix:
                    for key in KEYS:
                        recall_dict[n][key].append(0.0)
                        ndcg_dict[n][key].append(0.0)
                else:
                    cand_items = torch.unique(torch.cat(cand_item_tensors))  # [n_cand]

                    # dense retrieval: rank the candidate items by similarity to pred_emb
                    cand_emb = item_embedding[cand_items]  # [n_cand, n_embd]
                    _, logits = get_target_embed(
                        pred_emb[None], model, method_config, cand_emb
                    )  # [1, n_cand]
                    logits = logits[0]  # [n_cand]

                    topk = min(max(KEYS), cand_items.shape[0])
                    topk_idx = logits.topk(topk, largest=True)[1]  # [topk]
                    topk_items = cand_items[topk_idx]  # [topk] 0-based item indices

                    matches = (topk_items == target_item)  # [topk] bool
                    for key in KEYS:
                        recall_dict[n][key].append(matches[:key].any().float().item())
                        ndcg_dict[n][key].append(ndcg_at_k_torch(matches, key).item())

                # ---- AddCold (prefix items ∪ cold-start items, deduplicated) ----
                if add_cold_items is not None:
                    if has_prefix:
                        # cand_items already computed above
                        cand_items_cold = torch.unique(
                            torch.cat([cand_items, add_cold_items])
                        )
                    else:
                        cand_items_cold = add_cold_items

                    cand_emb_cold = item_embedding[cand_items_cold]
                    _, logits_cold = get_target_embed(
                        pred_emb[None], model, method_config, cand_emb_cold
                    )
                    logits_cold = logits_cold[0]

                    topk_cold = min(max(KEYS), cand_items_cold.shape[0])
                    topk_idx_cold = logits_cold.topk(topk_cold, largest=True)[1]
                    topk_items_cold = cand_items_cold[topk_idx_cold]

                    matches_cold = (topk_items_cold == target_item)
                    for key in KEYS:
                        recall_dict_cold[n][key].append(
                            matches_cold[:key].any().float().item()
                        )
                        ndcg_dict_cold[n][key].append(
                            ndcg_at_k_torch(matches_cold, key).item()
                        )

    return recall_dict, ndcg_dict, recall_dict_cold, ndcg_dict_cold


@torch.no_grad()
def evaluate_layer_accuracy(
    model,
    dataloader,
    device,
    method_config,
    KEYS=[10],
    num_beams=None,
):
    """
    Per-layer SID accuracy analysis via beam search.

    Generates all ``n_codebook`` SID tokens in a single beam search, then
    analyzes the beam candidates at each depth *k* = 1, 2, …, n_codebook.

    For each depth *k*:
      - **Layer_k_Recall@K** (cumulative): fraction of samples where at least
        one of the top-K beam candidates has its first *k* SIDs matching the
        ground-truth's first *k* SIDs.
      - **Layer_k_NDCG@K**  (cumulative): NDCG@K computed on the binary
        relevance vector (match / no-match of the first *k* SIDs) over the
        top-K beam candidates.
      - **Layer_k_Acc** (conditional): probability that the *k*-th SID is
        correct **given** the first *(k-1)* SIDs are correct.  Formally::

            Layer_k_Acc = N(samples correct up to k) / N(samples correct up to k-1)

        For k = 1 there is no prefix condition, so Layer_1_Acc = Layer_1_Recall@K.

    These metrics together reveal where the hierarchical SID decoding
    bottleneck lies: a large drop in Layer_k_Acc at a particular layer
    indicates that the model struggles to predict that layer even when all
    preceding layers are correct.

    :param KEYS: recall / NDCG cutoff(s).  Default ``[10]``.
    :param num_beams: beam width.  Defaults to ``max(KEYS)``.
    :return: dict mapping metric names to scalar values, plus a ``raw`` dict
             with per-sample arrays for further analysis.
    """
    model.eval()

    key = KEYS[0]  # primary cutoff (e.g. 10)
    if num_beams is None:
        num_beams = max(KEYS)

    # Accumulate per-sample binary matches at each depth
    layer_matches = {}   # depth k -> list[float] (0 or 1)
    layer_ndcg = {}      # depth k -> list[float]
    n_codebook_global = None

    for batch in tqdm(dataloader, desc="Layer-wise SID Evaluation"):
        labels = batch["labels_sids"].to(device)
        batch_size, n_codebook = labels.shape[0], labels.shape[1]
        n_codebook_global = n_codebook

        with torch.no_grad():
            _, input_kwargs = model_forward(
                model, batch, device, n_codebook, method_config, skip_forward=True
            )

        gen_kwargs = {
            "num_beams": num_beams,
            "max_new_tokens": n_codebook,
            "num_return_sequences": num_beams,
            "use_cache": True,
        }
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**input_kwargs, **gen_kwargs)

        # Drop decoder-start token -> [B, num_beams, n_codebook]
        outputs = outputs[:, 1 : 1 + n_codebook].reshape(
            batch_size, num_beams, -1
        )
        # Pad with zeros if the model emitted EOS early
        if outputs.shape[-1] < n_codebook:
            to_pad = n_codebook - outputs.shape[-1]
            pad_tensor = torch.zeros(
                (batch_size, num_beams, to_pad), device=outputs.device,
                dtype=outputs.dtype,
            )
            outputs = torch.cat([outputs, pad_tensor], dim=-1)

        # Analyze at each depth k = 1 .. n_codebook
        for k in range(1, n_codebook + 1):
            # [B, num_beams] — does candidate's first k SIDs match?
            match_k = (
                outputs[:, :, :k] == labels[:, None, :k]
            ).all(dim=-1)  # [B, num_beams]

            # Recall@key: any match among top-key candidates
            any_match = match_k[:, :key].any(dim=-1).float()  # [B]

            # NDCG@key per sample
            ndcg_vals = [
                ndcg_at_k_torch(match_k[b], key).item() for b in range(batch_size)
            ]

            layer_matches.setdefault(k, []).extend(any_match.tolist())
            layer_ndcg.setdefault(k, []).extend(ndcg_vals)

    # ---- Aggregate ----
    results = {}
    raw = {}
    for k in range(1, n_codebook_global + 1):
        mk = np.array(layer_matches[k])
        nk = np.array(layer_ndcg[k])

        recall_k = float(mk.mean())
        ndcg_k = float(nk.mean())

        results[f"Layer_{k}_Recall@{key}"] = recall_k
        results[f"Layer_{k}_NDCG@{key}"] = ndcg_k

        if k == 1:
            acc_k = recall_k  # no prefix condition
        else:
            n_correct_prev = int(np.array(layer_matches[k - 1]).sum())
            n_correct_k = int(mk.sum())
            acc_k = (
                n_correct_k / n_correct_prev
                if n_correct_prev > 0
                else 0.0
            )

        results[f"Layer_{k}_Acc"] = acc_k
        raw[f"Layer_{k}_matches"] = mk.tolist()
        raw[f"Layer_{k}_ndcg"] = nk.tolist()

    return results, raw
