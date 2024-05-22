#!/bin/bash
#SBATCH --job-name=rtuq-student-gridsearch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=shen-student-gridsearch_%j.out
#SBATCH --time=30:00:00

module load python/3.8
module load cuda/11.3

python3.8 -m venv dl_env
source dl_env/bin/activate
pip install --upgrade pip

pip install -r slurm_requirements.txt

echo "PYTHONPATH after activating venv: $PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:"/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"
echo "PYTHONPATH after adding project root: $PYTHONPATH"

shen_loss_weights=(0.5 1 2)
learning_rates=(0.0002 0.00002 0.000002)
epochs_list=(2 3 4)

for shen_loss_weight in "${shen_loss_weights[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        for epochs in "${epochs_list[@]}"; do
            version_identifier="shen_${shen_loss_weight}_lr${learning_rate}_e${epochs}"
            python3.8 src/experiments/uncertainty_distillation/uncertainty_distillation.py \
            --transfer_data_dir data/uncertainty_distillation \
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
        done
    done
done
