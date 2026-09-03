"""
Stage-wise inference FLOPs measurement for the test program.

Breaks one retrieval query into the two stages that actually cost compute
and reports FLOPs for each:

  1. Beam-search generation — one encoder forward plus ``steps`` autoregressive
     beam-decoding steps. Measured with ``torch.profiler(with_flops=True)``
     and cross-checked against an analytical formula derived from the T5
     config (matmul-only ops are counted by the profiler, so the two numbers
     should agree within ~10-15%).
  2. Dense retrieval / re-ranking — the ``emb_proj`` projection plus
     cosine/dot scoring over the candidate set. The exact scoring path
     (``get_target_embed``) is replayed with the real candidate sets for
     every ``Gen{N}`` candidate count, so the per-N numbers reflect the
     actual evaluation code path.

The per-``Gen{N}`` totals are what should be reported next to
``Gen{N}_Recall@K`` in complexity tables: query cost depends on the
candidate count N (and, for hybrid, on the prefix-bucket union sizes), not
on the cutoff K (top-k over already-scored candidates is negligible).

Note on beam width: decoding FLOPs scale ~linearly with ``num_beams``
beyond the first step; generation is measured once at the beam width the
evaluation actually uses.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from src.evaluation import get_target_embed, model_forward


def _profiler_activities():
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def _profile_call(fn):
    """Run ``fn()`` under the profiler; return (flops, result, top_ops).

    ``with_flops=True`` attributes FLOPs to matmul-type aten ops (mm, addmm,
    bmm, einsum, conv). ``top_ops`` is the top-10 (op, count, flops) list —
    when measured and analytical totals disagree, this shows where the
    profiler attributed the FLOPs (e.g. CPU aten ops vs CUDA kernels both
    being counted, or an unexpected head running inside generate()).
    """
    with profile(activities=_profiler_activities(), with_flops=True) as prof:
        result = fn()
    flops = 0.0
    ops = []
    for e in prof.key_averages():
        if e.flops and e.flops > 0:
            flops += e.flops
            ops.append((e.key, int(e.count), float(e.flops)))
    ops.sort(key=lambda x: -x[2])
    return flops, result, ops[:10]


def _emb_proj_flops_per_item(model):
    """Analytical FLOPs of one ``emb_proj`` forward for a single item."""
    proj = getattr(model, "emb_proj", None)
    if proj is None:
        return 0.0
    flops = 0.0
    for m in proj.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
    return flops


def analytical_generation_flops(model, seq_len, steps, num_beams):
    """Analytical FLOPs (2x MACs) of one ``generate()`` call, per query.

    Encoder: ``seq_len`` tokens x enc_layers x (self-attn projections 8d^2
    + attention 4*L*d + FFN 4*d*d_ff).

    Decoder (incremental, KV-cached): HF beam search expands the batch to
    ``batch * num_beams`` from step 0, so all ``steps`` decoding steps each
    run ``num_beams`` sequences — ``steps * num_beams`` decoder tokens per
    query. Per token per layer: self-attn 8d^2 + cross-attn Q 2d^2
    + cross-attn scores 4*L*d + FFN 4*d*d_ff. Cross-attn K/V projections
    run once per layer. LM head adds 2*d*V per generated token.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    d, dff, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
    enc_l = getattr(cfg, "num_layers", 6)
    dec_l = getattr(cfg, "num_decoder_layers", 6)

    f_enc_layer = 8 * d * d + 4 * seq_len * d + 4 * d * dff
    encoder = seq_len * enc_l * f_enc_layer

    f_dec_step_layer = 8 * d * d + 2 * d * d + 4 * seq_len * d + 4 * d * dff
    dec_tokens = steps * num_beams
    decoder = dec_tokens * dec_l * f_dec_step_layer
    cross_kv_once = dec_l * 4 * d * d
    lm_head = dec_tokens * 2 * d * v

    return float(encoder + decoder + cross_kv_once + lm_head)


@torch.no_grad()
def measure_inference_flops(
    model,
    dataloader,
    device,
    n_codebook,
    method_config,
    item_embedding,
    prefix2items=None,
    unseen_semantic_ids=None,
    num_beams=None,
    max_new_tokens=None,
    num_candidates_list=None,
    num_batches=3,
    cand_stat_dataloader=None,
):
    """Measure per-query inference FLOPs, stage by stage.

    :param prefix2items: hybrid mode — dict ``tuple(prefix) -> item indices``.
    :param unseen_semantic_ids: liger mode — cold-start SIDs appended to the
        generative candidates; candidate count per Gen{N} is N + n_unseen.
    :param num_beams: beam width of the generation stage. Defaults to
        ``max(num_candidates_list)`` — the width the evaluation uses.
    :param max_new_tokens: autoregressive steps. Defaults to ``prefix_depth``
        for hybrid, ``n_codebook`` otherwise.
    :param num_candidates_list: the Gen{N} candidate counts to measure the
        dense-retrieval stage with.
    :param num_batches: number of batches used for FLOPs profiling (generation
        + rerank stages).  FLOPs are deterministic for fixed seq_len / C, so
        a small value (default 3) is sufficient and reflects per-query cost
        faithfully without running the whole dataset.
    :param cand_stat_dataloader: optional separate dataloader used to collect
        (a) candidate-count statistics (C_N mean/p95/max) and (b) wall-clock
        generation latency over the full dataset for hybrid mode.  When
        provided, ALL batches are consumed; FLOPs profiling still uses at most
        ``num_batches`` from the main ``dataloader``.  Pass the full-dataset
        dataloader here for accurate per-query latency averages.
        When None, both stats and latency come from the same ``num_batches``.
    :return: nested dict with ``generation`` and per-N ``rerank`` entries,
        each carrying measured FLOPs and wall-clock latency per query;
        ``rerank`` entries also carry candidate-count stats and a ``total``.
    """
    model.eval()
    mode = "hybrid" if prefix2items is not None else "liger"
    if num_candidates_list is None:
        num_candidates_list = [20, 40, 60, 80, 100]
    num_candidates_list = sorted(num_candidates_list)
    if num_beams is None:
        num_beams = max(num_candidates_list)
    if max_new_tokens is None:
        prefix_depth = method_config.get("prefix_depth", 0)
        max_new_tokens = (
            prefix_depth
            if mode == "hybrid" and 0 < prefix_depth < n_codebook
            else n_codebook
        )

    gen_kwargs = {
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_beams,
        "use_cache": True,
    }

    # ---- Stage 1: profile generation, keep outputs for stage 2 ----
    gen_flops_total = 0.0
    n_samples = 0
    batches_measured = 0
    seq_len = None
    samples_pred_emb = []    # one [n_embd] tensor per query
    samples_gen_tokens = []  # one [num_beams, steps] tensor per query

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        _, input_kwargs = model_forward(
            model, batch, device, n_codebook, method_config, skip_forward=True
        )
        if seq_len is None:
            src = input_kwargs.get("inputs_embeds", input_kwargs.get("input_ids"))
            seq_len = int(src.shape[1])

        def _gen():
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    return model.generate(**input_kwargs, **gen_kwargs)
            return model.generate(**input_kwargs, **gen_kwargs)

        flops, outputs, top_ops = _profile_call(_gen)
        gen_flops_total += flops
        batches_measured += 1
        gen_top_ops = [
            {"op": k, "count": c, "flops": f} for k, c, f in top_ops
        ]

        batch_size = batch["labels_sids"].shape[0]
        n_samples += batch_size

        predicted_embedding = model.predicted_embedding
        if predicted_embedding is not None:
            predicted_embedding = predicted_embedding.reshape(
                batch_size, num_beams, -1
            )[:, 0]

        gen_tokens = outputs[:, 1 : 1 + max_new_tokens].reshape(
            batch_size, num_beams, -1
        )
        if gen_tokens.shape[-1] < max_new_tokens:
            to_pad = max_new_tokens - gen_tokens.shape[-1]
            gen_tokens = torch.cat(
                [
                    gen_tokens,
                    torch.zeros(
                        (batch_size, num_beams, to_pad),
                        dtype=gen_tokens.dtype,
                        device=gen_tokens.device,
                    ),
                ],
                dim=-1,
            )

        if predicted_embedding is not None:
            samples_pred_emb.extend(list(predicted_embedding))
        samples_gen_tokens.extend(list(gen_tokens))

    gen_per_query = gen_flops_total / max(n_samples, 1)

    # ---- Generation latency: full-dataset pass (no profiler overhead) ----
    # Use cand_stat_dataloader when provided (full dataset); otherwise reuse
    # the batches already profiled above as a quick approximation.
    _use_cuda_events = device.type == "cuda" and torch.cuda.is_available()
    gen_latency_total_ms = 0.0
    gen_lat_samples = 0

    def _time_generate(kwargs):
        if _use_cuda_events:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0.record()
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.generate(**kwargs, **gen_kwargs)
            else:
                out = model.generate(**kwargs, **gen_kwargs)
            t1.record()
            torch.cuda.synchronize()
            return t0.elapsed_time(t1), out
        else:
            import time
            t0 = time.perf_counter()
            out = model.generate(**kwargs, **gen_kwargs)
            return (time.perf_counter() - t0) * 1000.0, out

    lat_loader = cand_stat_dataloader if cand_stat_dataloader is not None else dataloader
    for lat_batch in lat_loader:
        _, lat_kwargs = model_forward(
            model, lat_batch, device, n_codebook, method_config, skip_forward=True
        )
        ms, _ = _time_generate(lat_kwargs)
        gen_latency_total_ms += ms
        gen_lat_samples += lat_batch["labels_sids"].shape[0]

    gen_latency_per_query_ms = gen_latency_total_ms / max(gen_lat_samples, 1)

    # ---- Stage 2: replay the dense-retrieval scoring per Gen{N} ----
    proj_flops = _emb_proj_flops_per_item(model)
    d_model = getattr(getattr(model, "config", None), "d_model", None)
    n_unseen = (
        0 if unseen_semantic_ids is None else int(unseen_semantic_ids.shape[0])
    )

    rerank = {}
    if samples_pred_emb:
        if mode == "hybrid":
            prefix2items_tensor = {
                p: torch.tensor(v, dtype=torch.long, device=device)
                for p, v in prefix2items.items()
            }

            # ---- Candidate-count statistics ----
            # Use cand_stat_dataloader (full dataset) when provided; otherwise
            # fall back to the gen tokens already collected from num_batches.
            if cand_stat_dataloader is not None:
                stat_counts = {n: [] for n in num_candidates_list}
                for stat_batch in cand_stat_dataloader:
                    _, stat_kwargs = model_forward(
                        model, stat_batch, device, n_codebook, method_config,
                        skip_forward=True,
                    )
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad():
                        stat_outputs = model.generate(
                            **stat_kwargs,
                            num_beams=num_beams,
                            max_new_tokens=max_new_tokens,
                            num_return_sequences=num_beams,
                            use_cache=True,
                        )
                    bsz = stat_batch["labels_sids"].shape[0]
                    stat_tokens = stat_outputs[:, 1 : 1 + max_new_tokens].reshape(
                        bsz, num_beams, -1
                    )
                    if stat_tokens.shape[-1] < max_new_tokens:
                        to_pad = max_new_tokens - stat_tokens.shape[-1]
                        stat_tokens = torch.cat([
                            stat_tokens,
                            torch.zeros((bsz, num_beams, to_pad),
                                        dtype=stat_tokens.dtype,
                                        device=stat_tokens.device),
                        ], dim=-1)
                    for b in range(bsz):
                        for n in num_candidates_list:
                            tensors = [
                                prefix2items_tensor.get(
                                    tuple(int(x) for x in stat_tokens[b, r].tolist())
                                )
                                for r in range(n)
                            ]
                            tensors = [t for t in tensors if t is not None]
                            stat_counts[n].append(
                                int(torch.unique(torch.cat(tensors)).shape[0])
                                if tensors else 0
                            )
                full_counts = {n: np.asarray(v, dtype=float) for n, v in stat_counts.items()}
            else:
                # derive counts from the profiling-batch gen_tokens already collected
                full_counts = None  # filled per-N below from profiler loop counts

            for n in num_candidates_list:
                counts = []

                def _rerank():
                    for pred_emb, gen_tok in zip(
                        samples_pred_emb, samples_gen_tokens
                    ):
                        tensors = [
                            prefix2items_tensor.get(
                                tuple(int(x) for x in gen_tok[r].tolist())
                            )
                            for r in range(n)
                        ]
                        tensors = [t for t in tensors if t is not None]
                        if not tensors:
                            counts.append(0)
                            continue
                        cand_items = torch.unique(torch.cat(tensors))
                        counts.append(int(cand_items.shape[0]))
                        cand_emb = item_embedding[cand_items]
                        get_target_embed(
                            pred_emb[None], model, method_config, cand_emb
                        )

                flops, _, _ = _profile_call(_rerank)

                # ---- rerank latency: measure over profiling samples ----
                # (full-dataset rerank latency not measured separately because
                # it requires re-running generate; use profiling-batch estimate)
                def _rerank_time():
                    for pred_emb, gen_tok in zip(
                        samples_pred_emb, samples_gen_tokens
                    ):
                        tensors = [
                            prefix2items_tensor.get(
                                tuple(int(x) for x in gen_tok[r].tolist())
                            )
                            for r in range(n)
                        ]
                        tensors = [t for t in tensors if t is not None]
                        if not tensors:
                            continue
                        cand_items = torch.unique(torch.cat(tensors))
                        cand_emb = item_embedding[cand_items]
                        get_target_embed(pred_emb[None], model, method_config, cand_emb)

                if _use_cuda_events:
                    _t0 = torch.cuda.Event(enable_timing=True)
                    _t1 = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    _t0.record()
                    _rerank_time()
                    _t1.record()
                    torch.cuda.synchronize()
                    rerank_ms = _t0.elapsed_time(_t1)
                else:
                    import time as _time
                    _ts = _time.perf_counter()
                    _rerank_time()
                    rerank_ms = (_time.perf_counter() - _ts) * 1000.0
                rerank_lat_per_query_ms = rerank_ms / max(n_samples, 1)

                # pick the right counts array for statistics
                if full_counts is not None:
                    counts_arr = full_counts[n]
                else:
                    counts_arr = np.asarray(counts, dtype=float)
                mean_c = float(counts_arr.mean()) if counts_arr.size else 0.0
                proj_c = mean_c * proj_flops if d_model is not None else None
                score_c = mean_c * 2 * d_model if d_model is not None else None
                rerank[n] = {
                    "measured_flops_per_query": flops / max(n_samples, 1),
                    # split so the N-dependent pure-retrieval cost (scoring)
                    # can be quoted separately from the (offline-precomputable)
                    # projection cost in complexity tables
                    "analytical_projection_flops_per_query": proj_c,
                    "analytical_scoring_flops_per_query": score_c,
                    "analytical_flops_per_query": (
                        proj_c + score_c if d_model is not None else None
                    ),
                    "candidate_count_mean": mean_c,
                    "candidate_count_p95": (
                        float(np.percentile(counts_arr, 95))
                        if counts_arr.size
                        else 0.0
                    ),
                    "candidate_count_max": (
                        float(counts_arr.max()) if counts_arr.size else 0.0
                    ),
                    "latency_ms_per_query": rerank_lat_per_query_ms,
                    "total_latency_ms_per_query": gen_latency_per_query_ms + rerank_lat_per_query_ms,
                }
        else:
            n_items = item_embedding.shape[0]
            for n in num_candidates_list:
                c = n + n_unseen
                idx = torch.arange(c, device=device) % n_items
                cand_emb_template = item_embedding[idx]  # [c, n_embd_items]

                def _rerank():
                    # per-query replay of generate_then_dense's batched
                    # scoring: [1, c, emb] candidates vs one query embedding
                    for pred_emb in samples_pred_emb:
                        get_target_embed(
                            pred_emb[None], model, method_config,
                            cand_emb_template[None],
                        )

                flops, _, _ = _profile_call(_rerank)

                # rerank latency over profiling samples
                if _use_cuda_events:
                    _t0 = torch.cuda.Event(enable_timing=True)
                    _t1 = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    _t0.record()
                    for pred_emb in samples_pred_emb:
                        get_target_embed(
                            pred_emb[None], model, method_config,
                            cand_emb_template[None],
                        )
                    _t1.record()
                    torch.cuda.synchronize()
                    rerank_ms = _t0.elapsed_time(_t1)
                else:
                    import time as _time
                    _ts = _time.perf_counter()
                    for pred_emb in samples_pred_emb:
                        get_target_embed(
                            pred_emb[None], model, method_config,
                            cand_emb_template[None],
                        )
                    rerank_ms = (_time.perf_counter() - _ts) * 1000.0
                rerank_lat_per_query_ms = rerank_ms / max(n_samples, 1)

                proj_c = c * proj_flops if d_model is not None else None
                score_c = c * 2 * d_model if d_model is not None else None
                rerank[n] = {
                    "measured_flops_per_query": flops / max(n_samples, 1),
                    "analytical_projection_flops_per_query": proj_c,
                    "analytical_scoring_flops_per_query": score_c,
                    "analytical_flops_per_query": (
                        proj_c + score_c if d_model is not None else None
                    ),
                    "candidate_count_mean": float(c),
                    "candidate_count_p95": float(c),
                    "candidate_count_max": float(c),
                    "latency_ms_per_query": rerank_lat_per_query_ms,
                    "total_latency_ms_per_query": gen_latency_per_query_ms + rerank_lat_per_query_ms,
                }

    for r in rerank.values():
        r["total_flops_per_query"] = gen_per_query + r["measured_flops_per_query"]

    return {
        "mode": mode,
        "num_beams": num_beams,
        "steps": max_new_tokens,
        "num_batches_measured": batches_measured,
        "n_samples_flops": n_samples,
        "n_samples_latency": gen_lat_samples,
        "seq_len": seq_len,
        "generation": {
            "measured_flops_per_query": gen_per_query,
            "analytical_flops_per_query": (
                analytical_generation_flops(model, seq_len, max_new_tokens, num_beams)
                if seq_len is not None
                else None
            ),
            "latency_ms_per_query": gen_latency_per_query_ms,
            "top_ops_by_flops": gen_top_ops,
        },
        "rerank": rerank,
    }
