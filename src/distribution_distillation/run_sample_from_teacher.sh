#!/bin/bash

export PYTHONPATH="/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/"

PYTHON_SCRIPT="src/distribution_distillation/sample_from_teacher.py"

INPUT_DATA_DIR="out/bert_teacher/data"
TEACHER_MODEL_SAVE_DIR="out/bert_teacher/final_hd030_ad020_cd035/model"
OUTPUT_DIR="data/distribution_distillation"
M=5
K=5
SEED=42

python $PYTHON_SCRIPT --input_data_dir $INPUT_DATA_DIR \
                      --teacher_model_save_dir $TEACHER_MODEL_SAVE_DIR \
                      --output_dir $OUTPUT_DIR \
                      --m $M \
                      --k $K \
                      --seed $SEED
