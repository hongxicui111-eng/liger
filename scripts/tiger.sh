
# Copyright (c) Meta Platforms, Inc. and affiliates.

# tiger
# for dataset_name in Beauty Toys_and_Games Sports_and_Outdoors
# do   
#     python run.py \
#         dataset=amazon \
#         dataset.name=$dataset_name \
#         seed=42 \
#         device_id=0 \
#         method=base \
#         test_method=tiger \
#         experiment_id="tiger_$dataset_name"
# done


# for dataset_name in steam
# do
#     python run.py \
#         dataset=steam \
#         dataset.name=$dataset_name \
#         seed=42 \
#         device_id=0 \
#         method=base \
#         test_method=tiger \
#         experiment_id="tiger_$dataset_name"
# done
export CUDA_VISIBLE_DEVICES=0
python run.py \
    dataset=amazon \
    dataset.name="Sports_and_Outdoors" \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    experiment_id="tiger_Sports_and_Outdoors_weightloss_base01_bs1024_lr1e-3_rankloss_explore"