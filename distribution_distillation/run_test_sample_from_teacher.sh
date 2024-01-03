#!/bin/bash

export PYTHONPATH="/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/"

PYTHON_SCRIPT="distribution_distillation/sample_from_teacher.py"

INPUT_DATA_DIR="tests/bert_grid_search_test/data"
TEACHER_MODEL_SAVE_DIR="tests/bert_grid_search_test/final_hd070_ad070_cd070/model"
OUTPUT_DIR="tests/distribution_distillation/transfer_data/aleatoric"
M=5
K=10
SEED=42

python $PYTHON_SCRIPT --input_data_dir $INPUT_DATA_DIR \
                      --teacher_model_save_dir $TEACHER_MODEL_SAVE_DIR \
                      --output_dir $OUTPUT_DIR \
                      --m $M \
                      --k $K \
                      --seed $SEED
