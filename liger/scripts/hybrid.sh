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

PREFIX_DEPTH=${PREFIX_DEPTH:-2}

# Amazon datasets
for dataset_name in Beauty Toys_and_Games Sports_and_Outdoors
do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=0 \
        method=hybrid \
        test_method=hybrid \
        method.prefix_depth=$PREFIX_DEPTH \
        experiment_id="hybrid_prefix${PREFIX_DEPTH}_$dataset_name"
done

# Steam dataset
for dataset_name in steam
do
    python run.py \
        dataset=steam \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=0 \
        method=hybrid \
        test_method=hybrid \
        method.prefix_depth=$PREFIX_DEPTH \
        experiment_id="hybrid_prefix${PREFIX_DEPTH}_$dataset_name"
done
