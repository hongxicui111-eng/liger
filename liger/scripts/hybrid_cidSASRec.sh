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



#SASRec只用cid去训练，然后量化，但是模型中还是用SID
for dataset_name in Beauty 
do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        dataset.SASRec.enabled=True \
        dataset.SASRec.eval_metric="loss" \
        dataset.SASRec.similarity_metric="dot" \
        dataset.SASRec.fusion_mode="cf_only"
        dataset.SASRec.temperature=1.0 \
        dataset.SASRec.num_heads=8 \
        dataset.SASRec.epochs=200 \
        dataset.RQ-VAE.use_standard_scaler=True \
        dataset.model_embedding="semantic" \
        seed=42 \
        device_id=0 \
        method=hybrid \
        test_method=hybrid \
        method.prefix_depth=$PREFIX_DEPTH \
        method.embedding_loss_weight=1.0 \
        method.bucket_loss_weight=0.0 \
        method.similarity_metric="cosine"\
        logging.project="sidcid_Merge_HyGR" \
        experiment_id="SASRecLoss200Epoch_DotCfOnly_T1_SIDScaler_ModelSem_Pre${PREFIX_DEPTH}_embloss1_bucket0_ALL$dataset_name" \
        force_rerun=True
done