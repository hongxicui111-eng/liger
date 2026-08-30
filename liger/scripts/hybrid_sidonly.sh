for dataset_name in Beauty 
do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        dataset.SASRec.enabled=True \
        dataset.SASRec.eval_metric=metric \
        dataset.sid_path="/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/ID/Beauty_sentence-t5-xxl_fused_42.pkl" \
        dataset.SASRec.fused_embedding_path="/share/cuihongxi/rerank/hybird_gr/liger/liger/ID_generation/sasrec/ckpt/Beauty_sentence-t5-xxl_fused_42.pt" \
        dataset.model_embedding="semantic" \
        seed=42 \
        device_id=0 \
        method=setting \
        test_method=liger \
        method.embedding_loss_weight=1 \
        experiment_id="LigerSemantic_SASRecLoss_sid_embloss1_$dataset_name"
done
