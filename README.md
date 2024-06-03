# Enhancing Transformer Architectures with Real-Time Uncertainty for Reliable Text Classification

**Type:** Master's Thesis

**Author:** Johann Sonnenburg

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Benjamin Fabian

![results_summary](/analysis/plots/uncertainty_distillation/results_summary_figure.png)

## Contents

- [Summary](#summary)
- [Working With the Repository](#Working-With-the-Repository)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing Results](#Reproducing-Results)
    - [Uncertainty Distillation](#Uncertainty-Distillation)
    - [Robustness Study](#Robustness-Study)
    - [Out-of-Distribution Detection Analysis](#Out-of-Distribution-Detection-Analysis)
    - [Evaluation Code](#Evaluation-Code)
- [Results](#Results)
- [Project Structure](-Project-Structure)

## Summary

Despite their improved predictive power, modern deep learning architectures frequently suffer from miscalibration, a critical issue in real-world settings. This can be mitigated by enabling a model to accurately assess the uncertainty of its predictions, which is traditionally resource-intensive. This work addresses these closely connected problems in the context of the natural language processing (NLP) domain, specifically for a hate speech detection task. For this purpose, we adapt the uncertainty-aware distribution distillation framework to the transformer-based BERT architecture, which enables distilling a model that generates high-quality uncertainty estimates in real time. We empirically evaluate the predictive performance and the quality of the uncertainty estimates of the resulting teacher and student models and assess their robustness against covariate shifts and out-of-distribution data. Our findings reveal that we are able to successfully distill a student model that retains accuracy but also enhances calibration and uncertainty estimation quality, thereby significantly improving robustness. Thus, our work highlights the effective application of this distillation framework to transformer architectures and underscores its practical applicability to the NLP domain.

**Keywords**: Uncertainty Estimation, Transformer Architectures, Text Classification, Robustness, Calibration

**Full text**: The complete text is available as a PDF.

## Working With the Repository

### Dependencies

We require Python version 3.8.
Follow the instructions below to install the required dependencies.

### Setup
We include two different requirement files which do not differ in their content.
One is for local development and one for running experiments on a remote server with GPU support.
This separation enables us to develop the project locally despite additional requirements for running the experiments not being available on the local machine.

To install the requirements for local development, continue as follows:

1. Clone this repository

2. Create a virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install the requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing Results

Please note that all models were trained on GPU on a remote server. The training process is computationally expensive and time-consuming. We recommend using a GPU for training the models and running the experiments.

A note on the data: We include the unprocessed dataset used throughout the thesis, as well as the unprocessed out-of-distribution datasets.

Below we provide instructions on how to reproduce the results of the experiments conducted in the thesis. If needed, make sure to adapt all parameters throughout, including inside the scripts.

### Uncertainty Distillation

1. Execute ```src/preprocessing/robustness_study/initial_preprocessing.ipynb``` to preprocess the data and create the necessary splits for training and evaluation.
2. Perform a grid search to find the best hyperparameters for the teacher model (on SLURM cluster).
```bash
sbatch src/experiments/uncertainty_distillation/run_teacher_gridsearch.sh
```
Once the grid search is complete, run ```sbatch src/experiments/uncertainty_distillation/run_teacher_performance.sh``` to evaluate the best teacher model across different numbers of MC dropout samples.
3. Create the transfer dataset for the student model (locally).
```bash
python src/distribution_distillation/sample_from_teacher.py --input_data_dir out/bert_teacher_gridsearch/data --teacher_model_save_dir
out/bert_teacher_gridsearch/final_hd020_ad030_cd020/model --output_dir data/distribution_distillation --m 5 --k 10
```
4. Perform a grid search to find the best hyperparameters for the student model (on SLURM cluster).
```bash
sbatch src/experiments/uncertainty_distillation/run_student_gridsearch.sh
```

### Robustness Study 
1. Create perturbed versions of the test dataset (locally).
```bash
python src/experiments/robustness_study/perturb_data.py --input_dir data/robustness_study/preprocessed/test.csv --output_dir data/robustness_study/preprocessed_noisy
```
2. Run the robustness study (on SLURM cluster).
```bash
sbatch src/experiments/robustness_study/run_robustness_study.sh
```

#### Excursion: Augmented Student
1. Create the augmented student dataset (locally).
```bash
python src/experiments/uncertainty_distillation/augment_transfer_dataset.py
```
2. Perform a grid search to find the best hyperparameters for the augmented student model (on SLURM cluster).
```bash
sbatch src/experiments/uncertainty_distillation/run_augmented_student_gridsearch.sh
```

### Out-of-Distribution Detection Analysis
1. Execute ```src/preprocessing/out_of_distribution_detection/preprocessing.ipynb``` to preprocess the out-of-distribution datasets (locally).
2. Run the out-of-distribution detection analysis (on SLURM cluster).
```bash 
sbatch src/experiments/out_of_distribution_detection/run_out_of_distribution_detection.sh
```

### Evaluation Code
We need to evaluate the results of each stage in order to be able to proceed, as we first need to determine the best model hyperparameters.

For evaluating the results of the uncertainty distillation, run the following analysis notebooks:  
- ```analysis/teacher_hyperparameter_analysis.ipynb``` to find the best hyperparameters for the teacher model.  
- ```analysis/student_hyperparameter_analysis.ipynb``` to find the best hyperparameters for the student model.  
- ```analysis/student_vs_teacher.ipynb``` to compare the performance of the teacher and student model.  

For evaluating the results of the robustness study, run the following analysis notebooks:
- ```analysis/robustness_study.ipynb``` to analyze the robustness of the teacher and student model.

For evaluating the results of the out-of-distribution detection analysis, run the following analysis notebooks:
- ```analysis/ood_detection_result_analysis.ipynb``` to analyze the out-of-distribution detection performance.

## Results

All tables and figures presented in the thesis can be found in the analysis notebooks in the ```analysis``` folder.

## Project Structure

```bash
├── analysis                                        -- analysis notebooks
│   └── plots
├── data                                            -- datasets                          
├── out                                             -- experiment results
├── scripts                                         -- helper scripts
├── src
│   ├── experiments                                 -- experiment code, including model training and evaluation
│   │   ├── out_of_distribution_detection           
│   │   ├── robustness_study
│   │   └── uncertainty_distillation                              
│   ├── models                                      -- model implementations
│   ├── preprocessing                               -- preprocessing scripts
│   │   ├── out_of_distribution_detection           
│   │   └── robustness_study                              
│   └── utils                                       -- metrics, logger config, loss functions, etc.
└── tests                                           -- unit tests and test scripts               
```
