#/bin/bash
python scripts/train_bart_seq2seq_augmented_kilt.py \
    --gpus 4 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 64 \
    --max_steps 200000 \
    --divergences kl \
    --train_data_path /datastore/shared/kilt/datasets/structured_zeroshot-train-new_annotated_final_v2.jsonl \
    --use_views \
    2>&1 | tee models/bart_seq2seq_augmented_structured_zeroshot/log.txt
