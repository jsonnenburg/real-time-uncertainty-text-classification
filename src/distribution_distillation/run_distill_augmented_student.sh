#!/bin/bash
#SBATCH --job-name=rtuq-augment-student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=shen-student-augmentation_%j.out
#SBATCH --time=03:00:00

module load python/3.8
module load cuda/11.3

export PYTHONPATH="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"

python3.8 -m venv env
source env/bin/activate
echo "PYTHONPATH after activating venv: $PYTHONPATH"
pip install --upgrade pip
pip install -r slurm_requirements.txt

shen_loss_weight=2
learning_rate=0.0002
epochs=4

version_identifier="shen_${shen_loss_weight}_lr${learning_rate}_e${epochs}_augmented"

python3.8 src/distribution_distillation/train_augmented_student.py \
--transfer_data_dir data/distribution_distillation \
--teacher_model_save_dir out/bert_teacher/final_e3_lr2_hd020_ad020_cd030/model \
--version_identifier "$version_identifier" \
--shen_loss_weight $shen_loss_weight \
--learning_rate $learning_rate \
--batch_size 32 \
--epochs $epochs \
--max_length 48 \
--m 5 \
--k 10 \
--output_dir out/bert_student \
--remove_dropout_layers
