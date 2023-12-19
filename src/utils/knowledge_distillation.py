# shen teacher sampling
import os

import pandas as pd

from src.utils.data import Dataset
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config

# obtain samples from teacher's posterior predictive distribution on training (+validation) set after having determined
# the optimal hyperparameter settings for the teacher

# *sampling from teacher*
# - if not modelling aleatoric uncertainty:
# 	- generate *m* predictive samples from teacher for each sequence in the transfer training set (as we use dropout during inference time, we end up with *m* different values for each training sequence)
# - if modelling aleatoric uncertainty:
# 	- for each training sequence:
# 		1. compute mean observation noise $\tilde{\sigma}^2$ across **all** teachers (e.g., over all 20 MC dropout samples)
# 		2. then generate the *m* samples for each training sequence ($\{\hat{\mu}_t\}_{t=1}^{m}$) and obtain final predictive samples as $[\hat{\mu}_t, \tilde{\sigma}^2]$, where mean observation noise is the same for all samples t = 1,...,m
# 		3. additionally generate *k* random samples from $N(0,I)$ for each predictive sample t = 1,...,m to learn aleatoric uncertainty
# 	- authors use m=5 and k=10

# TODO: incorporate this directly into training bert_teacher.py with a flag for performing sampling or not, optimize to
#  only perform MC dropout sampling once and save the samples to disk

# transfer training set = training set + validation set (for now)
# transfer test set = test set

# load transfer training set from path
transfer_data_path = './tests/bert_grid_search_test/final_hd070_ad070_cd070/results/eval_results.json'

teacher_config = pd.read_json('./tests/bert_grid_search_test/final_hd070_ad070_cd070/results/eval_results.json')['model_config']

transfer_dataset = Dataset()


def load_data(dataset_dir:str, subset: str) -> pd.DataFrame:
    subset_file = os.path.join(dataset_dir, f'{subset}.csv')
    df = pd.read_csv(subset_file, sep='\t', index_col=0)
    return df

transfer_dataset.train = load_data(transfer_data_path, 'combined_train')
transfer_dataset.test = load_data(transfer_data_path, 'combined_test')


# load BERT teacher model with best configuration
config = create_bert_config(teacher_config.hidden_dropout_prob, teacher_config.attention_probs_dropout_prob, teacher_config.classifier_dropout_prob)

teacher = AleatoricMCDropoutBERT(config=config)

# load weights
teacher.load_weights('./tests/bert_grid_search_test/final_hd070_ad070_cd070/model/model.h5')

# sample from teacher

