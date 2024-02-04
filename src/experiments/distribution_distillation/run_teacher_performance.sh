#!/bin/bash
#SBATCH --job-name=rtuq-teacher-performance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=teacher-performance_%j.out
#SBATCH --time=04:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip

pip install -r slurm_requirements.txt

python3.8 src/experiments/distribution_distillation/teacher_performance.py --input_data_dir out/bert_teacher_gridsearch/data \
--teacher_model_save_dir out/bert_teacher/final_hd020_ad030_cd020/model --output_dir out/bert_teacher/final_hd020_ad030_cd020 \