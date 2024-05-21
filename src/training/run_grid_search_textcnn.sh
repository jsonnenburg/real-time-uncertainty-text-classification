#!/bin/bash
#SBATCH --job-name=rtuq-textcnn-gridsearch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:3
#SBATCH --mem=32G
#SBATCH --output=textcnn-gridsearch_%j.out
#SBATCH --time=40:00:00

module load python/3.8
module load cuda/11.3

python3.8 -m venv dl_env
source dl_env/bin/activate
pip install --upgrade pip

pip install -r slurm_requirements.txt

echo "PYTHONPATH after activating venv: $PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:"/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/"
echo "PYTHONPATH after adding project root: $PYTHONPATH"

python3.8 src/training/textcnn_gridsearch.py --input_data_dir data/robustness_study/preprocessed_no_stopwords \
--output_dir out/textcnn --epochs 200 --max_length 48 --batch_size 2048 --learning_rate 0.0002
