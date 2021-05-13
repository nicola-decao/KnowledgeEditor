#/bin/bash
python scripts/train_bert_binary_augmented_kilt.py \
    --gpus 1 \
    --accelerator ddp \
    --num_workers 32 \
    --batch_size 256 \
    --max_steps 50000 \
    --divergences kl \
#     --use_views \
    2>&1 | tee models/bert_binary_augmented_fever/log.txt
