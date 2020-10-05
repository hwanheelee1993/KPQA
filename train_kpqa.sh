export CUDA_VISIBLE_DEVICES=2
OUTPUT_DIR=./output/
BATCH_SIZE=64
NUM_EPOCHS=5
SAVE_STEPS=500

SEED=1
BERT_MODEL=bert-base-uncased
MAX_LENGTH=96
SUMMARY=./summary/

python train.py \
    --data_dir data/kpqa_train \
    --model_type bert \
    --labels data/kpqa_train/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_seq_length  $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --do_eval \
    --logging_steps 500 \
    --save_steps 500 \
    --evaluate_during_training \
    --overwrite_output_dir \
    --summary_dir $SUMMARY \
