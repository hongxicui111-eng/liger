#!/bin/bash
# ====================================================================
# Standalone test script for TIGER / Liger models with per-layer SID
# accuracy analysis.
#
# Usage:
#   bash scripts/test.sh
#
# Modify the variables below to match your trained model.
# ====================================================================

export CUDA_VISIBLE_DEVICES=0


#Sports

# python test.py \
#   dataset=amazon \
#   dataset.name=Sports_and_Outdoors \
#   method=hybrid \
#   test_method=hybrid \
#   seed=42 \
#   device_id=0 \
#   experiment_id=SASRecLoss200Epoch_DotNorm_T1_SIDScaler_ModelSem_Pre2_embloss1_bucket0_ALLSports_and_Outdoors \
#   dataset.model_embedding=semantic \
#   method.prefix_depth=2 \
#   method.input_sid_depth=0 \
#   dataset.sid_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/results/hybrid/Amazon_Sports_and_Outdoors/SASRecLoss200Epoch_DotNorm_T1_SIDScaler_ModelSem_Pre2_embloss1_bucket0_ALLSports_and_Outdoors_seed_42/sid_fused_42.pkl \
#   dataset.semantic_embedding_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/preprocessing/processed/Sports_and_Outdoors_sentence-t5-xxl_embeddings.pt

# python test.py \
#   dataset=amazon \
#   dataset.name=Sports_and_Outdoors \
#   method=setting \
#   test_method=liger \
#   seed=42 \
#   device_id=0 \
#   experiment_id=embloss1_ALLSports_and_Outdoors \
#   dataset.model_embedding=semantic \
#   dataset.sid_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/results/liger/Amazon_Sports_and_Outdoors/embloss1_ALLSports_and_Outdoors_seed_42/sid_42.pkl \
#   dataset.semantic_embedding_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/preprocessing/processed/Sports_and_Outdoors_sentence-t5-xxl_embeddings.pt





# Toys

# python test.py \
#   dataset=amazon \
#   dataset.name=Toys_and_Games \
#   method=hybrid \
#   test_method=hybrid \
#   seed=42 \
#   device_id=0 \
#   experiment_id=SASRecLoss200Epoch_DotNorm_T1_SIDScaler_ModelSem_Pre2_embloss1_bucket0_ALLToys_and_Games \
#   dataset.model_embedding=semantic \
#   method.prefix_depth=2 \
#   method.input_sid_depth=0 \
#   dataset.sid_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/results/hybrid/Amazon_Toys_and_Games/SASRecLoss200Epoch_DotNorm_T1_SIDScaler_ModelSem_Pre2_embloss1_bucket0_ALLToys_and_Games_seed_42/sid_fused_42.pkl \
#   dataset.semantic_embedding_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/preprocessing/processed/Toys_and_Games_sentence-t5-xxl_embeddings.pt 


# python test.py \
#   dataset=amazon \
#   dataset.name=Toys_and_Games \
#   method=setting \
#   test_method=liger \
#   seed=42 \
#   device_id=0 \
#   experiment_id=embloss1_ALLToys_and_Games \
#   dataset.model_embedding=semantic \
#   dataset.sid_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/results/liger/Amazon_Toys_and_Games/embloss1_ALLToys_and_Games_seed_42/sid_42.pkl \
#   dataset.semantic_embedding_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/preprocessing/processed/Toys_and_Games_sentence-t5-xxl_embeddings.pt






# Beauty
# python test.py \
#   dataset=amazon \
#   dataset.name=Beauty \
#   method=hybrid \
#   test_method=hybrid \
#   seed=42 \
#   device_id=0 \
#   experiment_id=SASRecSID2_embloss1_ALLBeauty \
#   dataset.model_embedding=semantic \
#   method.prefix_depth=2 \
#   method.input_sid_depth=0 \
#   dataset.sid_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/ID/Beauty_sentence-t5-xxl_fused_42.pkl \
#   dataset.semantic_embedding_path=/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/preprocessing/processed/Beauty_sentence-t5-xxl_embeddings.pt



python test.py \
  dataset=amazon \
  dataset.name=Beauty \
  method=setting \
  test_method=liger \
  seed=42 \
  device_id=1 \
  experiment_id=Beauty_ligerbase_True \
  dataset.model_embedding=semantic \
  dataset.sid_path=/share/cuihongxi/rerank/liger/ID_generation/ID/Beauty_sentence-t5-xxl_42.pkl \
  dataset.semantic_embedding_path=/share/cuihongxi/rerank/liger/ID_generation/preprocessing/processed/Beauty_sentence-t5-xxl_embeddings.pt \
  checkpoint_path=/share/cuihongxi/rerank/liger/results/liger/Amazon_Beauty/Beauty_ligerbase_True_seed_42/results/ckpt_best.pt