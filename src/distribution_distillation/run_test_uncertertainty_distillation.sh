#!/bin/bash

export PYTHONPATH="/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/"

PYTHON_SCRIPT="tests/test_uncertainty_distillation.py"

TRANSFER_DATA_DIR="tests/distribution_distillation/transfer_data/aleatoric"
TEACHER_MODEL_SAVE_DIR="tests/bert_grid_search_test/final_hd070_ad070_cd070/model"
M = 5
K = 10
LEARNING_RATE=0.00002
BATCH_SIZE=16
EPOCHS=1
MAX_LENGTH=48
N=20
OUTPUT_DIR="tests/distribution_distillation"
SEED=42

python $PYTHON_SCRIPT --transfer_data_dir $TRANSFER_DATA_DIR \
                      --teacher_model_save_dir $TEACHER_MODEL_SAVE_DIR \
                      --m $M \
                      --k $K \
                      --learning_rate $LEARNING_RATE \
                      --batch_size $BATCH_SIZE \
                      --epochs $EPOCHS \
                      --max_length $MAX_LENGTH \
                      --n $N \
                      --output_dir $OUTPUT_DIR \
                      --seed $SEED