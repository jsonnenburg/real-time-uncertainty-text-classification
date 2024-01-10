#!/bin/bash
#SBATCH --job-name=rtuq-sample-from-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=teacher-sampling_%j.out
#SBATCH --time=02:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip

pip install -r slurm_requirements.txt

python3.8 src/distribution_distillation/sample_from_teacher.py --input_data_dir out/bert_teacher/data \
--teacher_model_save_dir out/bert_teacher/final_hd030_ad020_cd035/model --output_dir data/distribution_distillation \
--m 5 --k 5 --seed 42
