# Enhancing Transformer Architectures with Real-Time Uncertainty for Reliable Text Classification

**Type:** Master's Thesis

**Author:** Johann Sonnenburg

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Benjamin Fabian

![results_summary](/analysis/plots/uncertainty_distillation/results_summary_figure.png)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

(Short summary of motivation, contributions and results)

**Keywords**: xxx (give at least 5 keywords / phrases).

**Full text**: [include a link that points to the full text of your thesis]
*Remark*: a thesis is about research. We believe in the [open science](https://en.wikipedia.org/wiki/Open_science) paradigm. Research results should be available to the public. Therefore, we expect dissertations to be shared publicly. Preferably, you publish your thesis via the [edoc-server of the Humboldt-Universität zu Berlin](https://edoc-info.hu-berlin.de/de/publizieren/andere). However, other sharing options, which ensure permanent availability, are also possible. <br> Exceptions from the default to share the full text of a thesis require the approval of the thesis supervisor.  

## Working with the repo

### Dependencies

Which Python version is required? 

Does a repository have information on dependencies or instructions on how to set up the environment?

### Setup

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

## Reproducing results

Describe steps how to reproduce your results.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)

### Training code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

### Pretrained models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project structure

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
