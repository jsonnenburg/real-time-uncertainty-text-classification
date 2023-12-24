"""
Sampling from the teacher model's posterior predictive distribution on the transfer training set.
"""
import json
import os

import pandas as pd
from tqdm import tqdm

import tensorflow as tf

from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.utils.data import Dataset
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config
from src.utils.loss_functions import aleatoric_loss, null_loss


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


def mc_dropout_transfer_sampling(model, data: tf.data.Dataset, m: int = 5, k: int = 10, seed_list: list = None) -> pd.DataFrame:
    """
    Perform Monte Carlo Dropout Transfer Sampling on a given model and dataset.

    This method generates an augmented dataset with additional uncertain labels
    created by MC Dropout and aleatoric uncertainty sampling.

    :param  model: The TensorFlow model to use for sampling.
    :param  data: Data to be used for sampling. Each element should be a tuple (features, labels).
    :param  m: Number of MC Dropout iterations.
    :param k: Number of aleatoric uncertainty samples per MC iteration.
    :param seed_list: List of seeds for reproducibility.
    :return: df: Augmented dataset with original features, labels, and uncertain labels.
    """
    augmented_data = {
        'sequences': [],
        'labels': [],
        'predictions': []
    }

    if seed_list is None:
        seed_list = range(m)

    for text, features, labels in tqdm(data, desc="Processing Data"):
        all_logits = []
        for i in range(m):
            tf.random.set_seed(seed_list[i])
            outputs = model(features, training=True)
            logits = outputs['logits']
            all_logits.append(logits)

        all_logits = tf.stack(all_logits, axis=0)
        mu_t = tf.nn.sigmoid(all_logits)  # shape is (m, batch_size, num_classes)

        sigma_hat_sq = tf.math.reduce_variance(all_logits, axis=0)
        sigma_hat = tf.math.sqrt(sigma_hat_sq)
        sigma_tilde = tf.reduce_mean(sigma_hat, axis=1)
        sigma_tilde_reshaped = tf.reshape(sigma_tilde, [1, -1, 1])  # reshape to (1, batch_size, 1)

        eps = tf.random.normal(shape=[k, mu_t.shape[1], mu_t.shape[2]])  # what should the shape be? (k, batch_size, num_classes)
        for i in range(k):
            y_t = tf.clip_by_value(mu_t + (sigma_tilde_reshaped * eps[i, :, :]), clip_value_min=0.0, clip_value_max=1.0)  # probabilistic predictions
            # y_t should be (k, batch_size, num_classes)
            for j in range(m):
                # for each original sequence, we now save m*k augmented sequences
                for seq_idx in range(features['input_ids'].shape[0]):  # iterate over each sequence in the batch
                    # extract individual sequence, label, and prediction
                    sequence = text[seq_idx].numpy().decode('utf-8')
                    label = labels[seq_idx].numpy()
                    prediction = y_t[j, seq_idx, :].numpy()[0] # shape should be (num_classes,)
                    # prediction is shape (batch_size, )
                    # append individual sequence, label, and prediction to augmented_data
                    augmented_data['sequences'].append(sequence)
                    augmented_data['labels'].append(label)
                    augmented_data['predictions'].append(prediction)

            # TODO: switch for loops around so that we end up with augmented sequences grouped by original sequence

    # convert augmented_data to a data frame
    columns = ['sequence', 'ground_truth_label', 'teacher_predicted_label']
    transfer_df = pd.DataFrame(augmented_data, columns=columns)
    return transfer_df


# transfer training set = training set + validation set (for now)
# transfer test set = test set


transfer_data_path = '/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/tests/bert_grid_search_test/data'

# load transfer training set from path
with open('/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/tests/bert_grid_search_test/final_hd070_ad070_cd070/results/eval_results.json', 'r') as f:
    teacher_config = json.load(f)['model_config']

transfer_dataset = Dataset()


def load_data(dataset_dir:str, subset: str) -> pd.DataFrame:
    subset_file = os.path.join(dataset_dir, f'{subset}.csv')
    df = pd.read_csv(subset_file, sep='\t', index_col=0)
    return df

transfer_dataset.train = load_data(transfer_data_path, 'combined_train')
transfer_dataset.test = load_data(transfer_data_path, 'combined_test')

# preprocess transfer training set
input_ids, attention_masks, labels = bert_preprocess(transfer_dataset.train)
training_set_preprocessed = tf.data.Dataset.from_tensor_slices((
        transfer_dataset.train['text'].values,
        {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        },
        labels
    )).batch(16)

# load BERT teacher model with best configuration
config = create_bert_config(teacher_config['hidden_dropout_prob'],
                            teacher_config['attention_probs_dropout_prob'],
                            teacher_config['classifier_dropout'])

# teacher = AleatoricMCDropoutBERT(config=config, custom_loss_fn=aleatoric_loss)

# teacher.built = True  # https://stackoverflow.com/questions/63658086/tensorflow-2-0-valueerror-while-loading-weights-from-h5-file

teacher = AleatoricMCDropoutBERT(config=config, custom_loss_fn=aleatoric_loss)

teacher.load_weights('/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/tests/bert_grid_search_test/final_hd070_ad070_cd070/model/model.tf')

# load weights
# teacher = tf.keras.models.load_model('/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/tests/bert_grid_search_test/final_hd070_ad070_cd070/model/model.tf', custom_objects={'AleatoricMCDropoutBERT': AleatoricMCDropoutBERT, 'aleatoric_loss': aleatoric_loss, 'null_loss': null_loss})

teacher.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss={'classifier': aleatoric_loss, 'log_variance': null_loss},
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )

# sample from teacher
transfer_df = mc_dropout_transfer_sampling(teacher, training_set_preprocessed, m=5, k=10)

save_dir = '/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/tests/distribution_distillation/data'
os.makedirs(save_dir, exist_ok=True)

# save augmented dataset
transfer_df.to_csv(os.path.join(save_dir, 'transfer_data.csv'), sep='\t', index=False)

