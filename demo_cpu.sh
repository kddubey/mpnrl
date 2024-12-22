python run.py \
    --run_name cpu-test \
    --loss mnrl \
    --per_device_train_batch_size 8 \
    --dataset_name "sentence-transformers/all-nli" \
    --dataset_config "triplet" \
    --dataset_size_train 16
