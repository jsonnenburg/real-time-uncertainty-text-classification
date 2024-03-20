#!/bin/bash
#SBATCH --job-name=rtuq-ood-detection
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v10032gb:2
#SBATCH --mem=64G
#SBATCH --output=ood-detection-irony_%j.out
#SBATCH --time=40:00:00

module load python/3.8
module load cuda/11.3

python3.8 -m venv env
source dl_env/bin/activate
pip install --upgrade pip

pip install -r slurm_requirements.txt

echo "PYTHONPATH after activating venv: $PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:"/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"
echo "PYTHONPATH after adding project root: $PYTHONPATH"

python3.8 src/experiments/out_of_distribution_detection/ood_detection.py --teacher_model_path out/bert_teacher/final_e3_lr2_hd020_ad020_cd030/model \
--student_model_path out/bert_student/m5_k10/shen_2_lr0.0002_e4/model \
--augmented_student_model_path out/bert_student_augmented/m5_k10/shen_2_lr0.0002_e2_augmented/model \
--data_dir data/out_of_distribution_detection/preprocessed/irony --output_dir out/out_of_distribution_detection/irony \
--n_trials 20 --run_for_teacher --run_for_student --run_for_augmented_student
