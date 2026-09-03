
# PREFIX_DEPTH=2



#跑一下liger base
# for dataset_name in Toys_and_Games
# do
#     python run.py \
#         dataset=amazon \
#         dataset.name=$dataset_name \
#         dataset.SASRec.enabled=False \
#         dataset.SASRec.eval_metric="loss" \
#         dataset.SASRec.similarity_metric="dot" \
#         dataset.SASRec.normalize_semantic=True \
#         dataset.SASRec.temperature=1.0 \
#         dataset.SASRec.num_heads=8 \
#         dataset.SASRec.epochs=200 \
#         dataset.RQ-VAE.use_standard_scaler=False \
#         dataset.model_embedding="semantic" \
#         seed=42 \
#         device_id=0 \
#         method=setting \
#         test_method=liger \
#         method.embedding_loss_weight=1.0 \
#         logging.project="Liger_base" \
#         experiment_id="embloss1_ALL$dataset_name" \
#         force_rerun=True
# done





# for dataset_name in Sports_and_Outdoors
# do
#     python run.py \
#         dataset=amazon \
#         dataset.name=$dataset_name \
#         dataset.SASRec.enabled=False \
#         dataset.SASRec.eval_metric="loss" \
#         dataset.SASRec.similarity_metric="dot" \
#         dataset.SASRec.normalize_semantic=True \
#         dataset.SASRec.temperature=1.0 \
#         dataset.SASRec.num_heads=8 \
#         dataset.SASRec.epochs=200 \
#         dataset.RQ-VAE.use_standard_scaler=False \
#         dataset.model_embedding="semantic" \
#         seed=42 \
#         device_id=1 \
#         method=setting \
#         test_method=liger \
#         method.embedding_loss_weight=1.0 \
#         logging.project="Liger_base" \
#         experiment_id="embloss1_ALL$dataset_name" \
#         force_rerun=True
# done




for dataset_name in Beauty
do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        dataset.SASRec.enabled=False \
        dataset.SASRec.eval_metric="loss" \
        dataset.SASRec.similarity_metric="dot" \
        dataset.SASRec.normalize_semantic=True \
        dataset.SASRec.temperature=1.0 \
        dataset.SASRec.num_heads=8 \
        dataset.SASRec.epochs=200 \
        dataset.RQ-VAE.use_standard_scaler=False \
        dataset.model_embedding="semantic" \
        seed=42 \
        device_id=1 \
        method=setting \
        test_method=liger \
        method.embedding_loss_weight=1.0 \
        method.dense_loss_mode="in_batch" \
        logging.project="Inbatch_Dense" \
        experiment_id="Inbatch_embloss1_ALL$dataset_name"
done