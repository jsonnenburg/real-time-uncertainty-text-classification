"""
Sampling from the teacher model's posterior predictive distribution on the transfer training set.
"""
import argparse
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

def epistemic_mc_dropout_transfer_sampling(model, data: tf.data.Dataset, m: int = 5, seed_list: list = None) -> pd.DataFrame:
    """
    Perform Monte Carlo dropout transfer sampling on a given model and dataset.
    Generates an augmented dataset with additional uncertain labels created by MC dropout.

    :param  model: The TensorFlow model to use for sampling.
    :param  data: Data to be used for sampling. Each element should be a tuple (features, labels).
    :param  m: Number of MC dropout iterations.
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

        for j in range(m):
            # for each original sequence, we now save m augmented sequences
            for seq_idx in range(features['input_ids'].shape[0]):  # iterate over each sequence in the batch
                # extract individual sequence, label, and prediction
                sequence = text[seq_idx].numpy().decode('utf-8')
                label = labels[seq_idx].numpy()
                prediction = mu_t[j, seq_idx, :].numpy()[0]  # shape should be (num_classes,)
                # prediction is shape (batch_size, )
                # append individual sequence, label, and prediction to augmented_data
                augmented_data['sequences'].append(sequence)
                augmented_data['labels'].append(label)
                augmented_data['predictions'].append(prediction)

    transfer_df = pd.DataFrame(augmented_data)
    return transfer_df


def aleatoric_mc_dropout_transfer_sampling(model, data: tf.data.Dataset, m: int = 5, k: int = 10, seed_list: list = None) -> pd.DataFrame:
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
        all_log_variances = []
        for i in range(m):
            tf.random.set_seed(seed_list[i])
            outputs = model(features, training=True)
            logits = outputs['logits']
            log_variances = outputs['log_variances']
            all_logits.append(logits)
            all_log_variances.append(log_variances)

        all_logits = tf.stack(all_logits, axis=0)
        all_log_variances = tf.stack(all_log_variances, axis=0)
        mu_t = tf.nn.sigmoid(all_logits)  # shape is (m, batch_size, num_classes)

        # sigma_hat_sq = tf.math.reduce_variance(all_logits, axis=0)
        sigma_hat = all_log_variances  # tf.math.sqrt(sigma_hat_sq)  # TODO: replace with log_variances from output!
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
    # columns = ['sequence', 'ground_truth_label', 'teacher_predicted_label']
    transfer_df = pd.DataFrame(augmented_data)
    return transfer_df


def load_data(dataset_dir:str, subset: str) -> pd.DataFrame:
    subset_file = os.path.join(dataset_dir, f'{subset}.csv')
    df = pd.read_csv(subset_file, sep='\t', index_col=0)
    return df


def preprocess_transfer_data(df: pd.DataFrame) -> tf.data.Dataset:
    input_ids, attention_masks, labels = bert_preprocess(df)
    dataset = tf.data.Dataset.from_tensor_slices((
        df['text'].values,
        {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        },
        labels
    )).batch(16)
    return dataset


def main(args):
    # transfer training set = training set + validation set (for now)
    # transfer test set = test set

    # load config of the best teacher configuration as determined by grid search
    with open(os.path.join(args.teacher_model_save_dir, 'config.json'), 'r') as f:
        teacher_config = json.load(f)

    # load dataset which will be used to create augmented dataset + test set for student training
    input_dataset = Dataset(train=load_data(args.input_data_dir, 'combined_train'),
                            test=load_data(args.input_data_dir, 'combined_test'))

    # preprocess transfer training set
    training_set_preprocessed = preprocess_transfer_data(input_dataset.train)
    test_set_preprocessed = preprocess_transfer_data(input_dataset.test)

    # load BERT teacher model with best configuration
    config = create_bert_config(teacher_config['hidden_dropout_prob'],
                                teacher_config['attention_probs_dropout_prob'],
                                teacher_config['classifier_dropout'])

    # initialize teacher model
    teacher = AleatoricMCDropoutBERT(config=config, custom_loss_fn=aleatoric_loss)
    checkpoint_path = os.path.join(args.teacher_model_save_dir, 'cp-{epoch:02d}.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading weights from", checkpoint_dir)
        teacher.load_weights(latest_checkpoint)

    teacher.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss={'classifier': aleatoric_loss, 'log_variance': null_loss},
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            run_eagerly=True
        )

    # sample from teacher's posterior predictive distribution on transfer training set
    m = args.m
    k = args.k

    if args.epistemic_only:
        output_dir = os.path.join(args.output_dir, 'epistemic_only')
        transfer_data_dir = os.path.join(output_dir, f'm{m}')
        transfer_df_train = epistemic_mc_dropout_transfer_sampling(teacher, training_set_preprocessed, m=m)
        transfer_df_test = epistemic_mc_dropout_transfer_sampling(teacher, test_set_preprocessed, m=m)
    else:
        output_dir = os.path.join(args.output_dir, 'aleatoric_and_epistemic')
        transfer_data_dir = os.path.join(output_dir, f'm{m}_k{k}')
        transfer_df_train = aleatoric_mc_dropout_transfer_sampling(teacher, training_set_preprocessed, m=m, k=k)
        transfer_df_test = aleatoric_mc_dropout_transfer_sampling(teacher, test_set_preprocessed, m=m, k=k)

    os.makedirs(transfer_data_dir, exist_ok=True)

    transfer_df_train.to_csv(os.path.join(transfer_data_dir, 'transfer_train.csv'), sep='\t', index=False)
    transfer_df_test.to_csv(os.path.join(transfer_data_dir, 'transfer_test.csv'), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str)
    parser.add_argument('--teacher_model_save_dir', type=str)
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--epistemic_only', action='store_true')  # if true, only model epistemic uncertainty, else also model aleatoric uncertainty
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
