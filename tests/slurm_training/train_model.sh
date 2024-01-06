#!/bin/bash
#SBATCH --job-name=tensorflow-gpu-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=tensorflow-gpu-test_%j.out

conda activate ml

export PYTHONPATH="/vol/fob-vol1/nebenf23/real-time-uncertainty-text-classification/"

python train_model.py
