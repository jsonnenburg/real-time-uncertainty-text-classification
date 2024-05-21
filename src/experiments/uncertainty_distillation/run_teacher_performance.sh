#!/bin/bash
#SBATCH --job-name=rtuq-teacher-performance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=teacher-performance_%j.out
#SBATCH --time=04:00:00

module load python/3.8
module load cuda/11.3

python3.8 -m venv env
source dl_env/bin/activate
pip install --upgrade pip

pip install -r slurm_requirements.txt

echo "PYTHONPATH after activating venv: $PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:"/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"
echo "PYTHONPATH after adding project root: $PYTHONPATH"

python3.8 src/experiments/distribution_distillation/teacher_performance.py --input_data_dir out/bert_teacher/data \
--teacher_model_save_dir out/bert_teacher/final_e3_lr2_hd020_ad020_cd030/model --output_dir out/bert_teacher/final_e3_lr2_hd020_ad020_cd030