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
    - [Training Code](#Training-Code)
    - [Evaluation Code](#Evaluation-Code)
    - [Pre-Trained Models](#Pre-Trained-Models)
- [Results](#Results)
- [Project Structure](-Project-Structure)

## Summary

Despite their improved predictive power, modern deep learning architectures frequently suffer from miscalibration, a critical issue in real-world settings. This can be mitigated by enabling a model to accurately assess the uncertainty of its predictions, which is traditionally resource-intensive. This work addresses these closely connected problems in the context of the natural language processing (NLP) domain, specifically for a hate speech detection task. For this purpose, we adapt the uncertainty-aware distribution distillation framework to the transformer-based BERT architecture, which enables distilling a model that generates high-quality uncertainty estimates in real time. We empirically evaluate the predictive performance and the quality of the uncertainty estimates of the resulting teacher and student models and assess their robustness against covariate shifts and out-of-distribution data. Our findings reveal that we are able to successfully distill a student model that retains accuracy but also enhances calibration and uncertainty estimation quality, thereby significantly improving robustness. Thus, our work highlights the effective application of this distillation framework to transformer architectures and underscores its practical applicability to the NLP domain.

**Keywords**: Uncertainty Estimation, Transformer Architectures, Text Classification, Robustness, Calibration

**Full text**: [include a link that points to the full text of your thesis]
*Remark*: a thesis is about research. We believe in the [open science](https://en.wikipedia.org/wiki/Open_science) paradigm. Research results should be available to the public. Therefore, we expect dissertations to be shared publicly. Preferably, you publish your thesis via the [edoc-server of the Humboldt-Universität zu Berlin](https://edoc-info.hu-berlin.de/de/publizieren/andere). However, other sharing options, which ensure permanent availability, are also possible. <br> Exceptions from the default to share the full text of a thesis require the approval of the thesis supervisor.  

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

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing Results

Please note that all models were trained on GPU on a remote server. The training process is computationally expensive and time-consuming. We recommend using a GPU for training the models and running the experiments.

A note on the data: We include the unprocessed dataset used throughout the thesis, as well as the unprocessed out-of-distribution datasets.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)

### Training Code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation Code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

### Pre-Trained Models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project Structure

```bash
├── analysis                                        -- analysis notebooks
│   └── plots
├── data                                            -- datasets                          
├── out                                             -- experiment results
├── scripts                                         -- helper scripts
├── src
│   ├── experiments                                 -- experiment code
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
