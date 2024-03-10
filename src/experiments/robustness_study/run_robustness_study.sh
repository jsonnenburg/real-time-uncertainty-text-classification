#!/bin/bash
#SBATCH --job-name=rtuq-robustness-study
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=64G
#SBATCH --output=robustness-study_%j.out
#SBATCH --time=20:00:00

module load python/3.8
module load cuda/11.3

python3.8 -m venv env
source dl_env/bin/activate
pip install --upgrade pip

pip install -r slurm_requirements.txt

echo "PYTHONPATH after activating venv: $PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:"/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"
echo "PYTHONPATH after adding project root: $PYTHONPATH"

python3.8 src/experiments/robustness_study/robustness_study.py --teacher_model_path out/bert_teacher/final_e3_lr2_hd020_ad020_cd030/model \
--student_model_path out/bert_student/m5_k10/shen_2_lr0.0002_e4/model \
--augmented_student_model_path out/bert_student_augmented/m5_k10/shen_2_lr0.0002_e2_augmented/model \
--data_dir data/robustness_study/preprocessed_noisy --output_dir out/robustness_study_20trials --n_trials 20 \
--run_for_teacher --run_for_student
