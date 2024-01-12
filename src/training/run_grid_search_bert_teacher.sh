#!/bin/bash
#SBATCH --job-name=rtuq-bert-teacher-grid-search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=bert_finetune_%j.out
#SBATCH --time=60:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip

pip install -r slurm_requirements.txt

python3.8 src/training/train_bert_teacher.py --input_data_dir data/robustness_study/preprocessed --output_dir src/training/out/bert_teacher  \
--learning_rate 0.00002 --batch_size 32 --epochs 3 --max_length 48 --mc_dropout_inference --seed 42 \
--save_datasets --cleanup