# Copyright (c) Meta Platforms, Inc. and affiliates.

# Hybrid generative-prefix + dense retrieval method.
#
# The generative (T5) model is only trained to autoregressively predict the
# first `prefix_depth` semantic ids (the "prefix"). After the prefix is
# generated, a dense retriever ranks the candidate items that fall under that
# prefix (prefix -> item-set mapping). No extra (4th) conflict-resolution sid
# is appended: collisions inside a prefix bucket are resolved by the dense
# retriever instead.
#
# Loss = sid autoregressive loss (only on the first `prefix_depth` sids)
#      + dense retrieval loss (positive = target item, negatives = all other
#        items in the library -- "摸高" / upper-bound version).
#
# Run with the default prefix depth (2):
#     bash scripts/hybrid.sh
# Run with prefix depth 3 (full 3-codebook sid, dense resolves collisions):
#     PREFIX_DEPTH=3 bash scripts/hybrid.sh

# PREFIX_DEPTH=${PREFIX_DEPTH:-2}
# export CUDA_VISIBLE_DEVICES=1
PREFIX_DEPTH=2
INPUT_SID_DEPTH=0




# ##对于模型内部也使用fused向量

# for dataset_name in Beauty 
# do
#     python run.py \
#         dataset=amazon \
#         dataset.name=$dataset_name \
#         dataset.SASRec.enabled=True \
#         dataset.SASRec.num_heads=8 \
#         dataset.SASRec.eval_metric="loss" \
#         dataset.SASRec.fusion_method="mlp" \
#         dataset.model_embedding="semantic" \
#         dataset.SASRec.stop_after_sasrec=Ture \
#         seed=42 \
#         device_id=1 \
#         method=hybrid \
#         test_method=hybrid \
#         method.prefix_depth=$PREFIX_DEPTH \
#         method.input_sid_depth=$INPUT_SID_DEPTH \
#         method.embedding_loss_weight=1 \
#         method.bucket_loss_weight=0.0 \
#         method.similarity_metric="cosine" \
#         logging.project="sidcid_Merge_HyGR" \
#         experiment_id="MLP_SASRecLoss_codePre${PREFIX_DEPTH}Input${INPUT_SID_DEPTH}_embloss1_bucket0_ALL$dataset_name" \
#         force_rerun=True
# done






for dataset_name in Beauty 
do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        dataset.SASRec.enabled=True \
        dataset.SASRec.num_heads=8 \
        dataset.SASRec.eval_metric="loss" \
        dataset.SASRec.fusion_mode="cf_only" \
        dataset.model_embedding="semantic" \
        dataset.SASRec.stop_after_sasrec=Ture \
        dataset.SASRec.dropout=0.5 \
        seed=42 \
        device_id=1 \
        method=hybrid \
        test_method=hybrid \
        method.prefix_depth=$PREFIX_DEPTH \
        method.input_sid_depth=$INPUT_SID_DEPTH \
        method.embedding_loss_weight=1 \
        method.bucket_loss_weight=0.0 \
        method.similarity_metric="cosine" \
        logging.project="sidcid_Merge_HyGR" \
        experiment_id="CFOnly_dr05_SASRecLoss_codePre${PREFIX_DEPTH}Input${INPUT_SID_DEPTH}_embloss1_bucket0_ALL$dataset_name" \
        force_rerun=True
done
