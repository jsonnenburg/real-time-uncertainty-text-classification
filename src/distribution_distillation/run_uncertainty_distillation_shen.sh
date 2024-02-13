#!/bin/bash
#SBATCH --job-name=rtuq-train-student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=train-bert-student_%j.out
#SBATCH --time=02:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip

pip install -r slurm_requirements.txt

python3.8 src/distribution_distillation/uncertainty_distillation.py --transfer_data_dir data/distribution_distillation \
--teacher_model_save_dir out/bert_teacher/final_hd020_ad030_cd020/model --version_identifier shen_e10_lr00002 \
--learning_rate 0.0002 --batch_size 256 --epochs 10 --max_length 48 --m 5 --k 10 --output_dir out/bert_student \
--remove_dropout_layers
