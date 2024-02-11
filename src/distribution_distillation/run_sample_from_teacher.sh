#!/bin/bash
#SBATCH --job-name=rtuq-sample-from-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=teacher-sampling_%j.out
#SBATCH --time=04:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip

pip install -r slurm_requirements.txt

python3.8 src/distribution_distillation/sample_from_teacher.py --input_data_dir out/bert_teacher/data \
--teacher_model_save_dir out/bert_teacher/final_e3_lr2_hd020_ad020_cd030/model --output_dir data/distribution_distillation \
--m 5 --k 10 --seed 42
