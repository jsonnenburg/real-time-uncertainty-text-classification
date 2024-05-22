#!/bin/bash
#SBATCH --job-name=rtuq-bert-teacher-grid-search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=bert_finetune_%j.out
#SBATCH --time=60:00:00

module load python/3.8
module load cuda/11.3

python3.8 -m venv dl_env
source dl_env/bin/activate
pip install --upgrade pip

pip install -r slurm_requirements.txt

echo "PYTHONPATH after activating venv: $PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:"/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"
echo "PYTHONPATH after adding project root: $PYTHONPATH"

python3.8 src/experiments/uncertainty_distillation/train_bert_teacher.py --input_data_dir data/uncertainty_distillation/preprocessed --output_dir out/bert_teacher \
--batch_size 32 --max_length 48 --seed 42 --save_datasets --cleanup
