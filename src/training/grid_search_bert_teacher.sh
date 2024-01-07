#!/bin/bash
#SBATCH --job-name=rtuq-bert-teacher-grid-search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=bert_finetune_%j.out
#SBATCH --time=60:00:00

conda activate ml

export PYTHONPATH="/vol/fob-vol1/nebenf23/real-time-uncertainty-text-classification/"

python train_bert_teacher.py --input_data_dir data/robustness_study/preprocessed --output_dir training/out/bert_teacher  \
--learning_rate 0.00002 --batch_size 32 --epochs 3 --max_length 48 --mc_dropout_inference --seed 42 \
--save_datasets --cleanup
