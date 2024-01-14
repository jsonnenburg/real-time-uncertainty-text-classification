#!/bin/bash
#SBATCH --job-name=rtuq-train-student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=train-bert-student_%j.out
#SBATCH --time=60:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip

pip install -r slurm_requirements.txt

python3.8 src/distribution_distillation/uncertainty_distillation.py --transfer_data_dir data/distribution_distillation \
--teacher_model_save_dir out/bert_teacher/final_hd030_ad020_cd035/model --learning_rate 0.00002 --batch_size 32 \
--epochs 2 --max_length 48 --output_dir out/bert_student --m 5 --k 5 --seed 42
