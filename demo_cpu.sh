python run.py \
    --run_name cpu-test \
    --loss mpnrl \
    --per_device_train_batch_size 8 \
    --dataset_name "sentence-transformers/all-nli" \
    --dataset_config "triplet" \
    --dataset_split_val "dev" \
    --dataset_size_train 16
