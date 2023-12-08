#!/bin/bash
#SBATCH --job-name=rtuq_bert_teacher_grid_search
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=bert_finetune_%j.out  # adapt
#SBATCH --error=bert_finetune_%j.err

module load python/3.8 # needed?
module load cuda
source ~/myenv/bin/activate # change


python bert_finetune.py --input_data_dir ./data/preprocessed/ --output_dir ./out/ --cleanup False
