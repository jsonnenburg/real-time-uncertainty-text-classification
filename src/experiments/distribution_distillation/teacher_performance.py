"""
Used to obtain data for creating graphs similar to 4b and c in Shen et al. (2021).
Evaluates performance via F1 score and uncertainty via ECE score for different number of MC dropout samples.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.distribution_distillation.sample_from_teacher import load_data
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config
from src.utils.data import Dataset
from src.utils.loss_functions import bayesian_binary_crossentropy, null_loss
from src.utils.metrics import serialize_metric, f1_score, ece_score


def compute_mc_dropout_metrics(model, eval_data, n=50) -> dict:
    total_logits = []
    total_probs = []
    total_mean_logits = []
    total_mean_log_variances = []
    total_labels = []

    start_time = time.time()
    for batch in eval_data:
        features, labels = batch

        samples = model.mc_dropout_sample(features, n=n)
        logits = samples['logit_samples']
        probs = samples['prob_samples']
        mean_logits = samples['mean_logits']
        mean_log_variances = samples['mean_log_variances']
        total_logits.append(logits.numpy())
        total_probs.append(probs.numpy())
        total_mean_logits.extend(mean_logits.numpy())
        total_mean_log_variances.extend(mean_log_variances.numpy())
        total_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels) * 1000

    all_labels = np.array(total_labels)

    y_prob_mcd = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
    y_pred_mcd = y_prob_mcd.round(0).astype(int)
    y_true = all_labels

    f1 = f1_score(y_true, y_pred_mcd)
    ece = ece_score(y_true, y_pred_mcd, y_prob_mcd)
    return {
        "average_inference_time": serialize_metric(average_inference_time),
        "f1_score": serialize_metric(f1),
        "ece_score": serialize_metric(ece),
    }


def preprocess_test_data(df: pd.DataFrame) -> tf.data.Dataset:
    input_ids, attention_masks, labels = bert_preprocess(df)
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        },
        labels
    )).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def main(args):
    # load config of the best teacher configuration as determined by grid search
    with open(os.path.join(args.teacher_model_save_dir, 'config.json'), 'r') as f:
        teacher_config = json.load(f)

    # load dataset which will be used to create augmented dataset + test set for student training
    input_dataset = Dataset(test=load_data(args.input_data_dir, 'combined_test'))

    # preprocess transfer training set
    test_set_preprocessed = preprocess_test_data(input_dataset.test)

    # load BERT teacher model with best configuration
    config = create_bert_config(teacher_config['hidden_dropout_prob'],
                                teacher_config['attention_probs_dropout_prob'],
                                teacher_config['classifier_dropout'])

    # initialize teacher model
    teacher = AleatoricMCDropoutBERT(config=config, custom_loss_fn=bayesian_binary_crossentropy)
    checkpoint_path = os.path.join(args.teacher_model_save_dir, 'cp-{epoch:02d}.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading weights from", checkpoint_dir)
        teacher.load_weights(latest_checkpoint).expect_partial()

    teacher.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss={'classifier': bayesian_binary_crossentropy, 'log_variance': null_loss},
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            run_eagerly=True
        )

    # compute metrics for different number of MC dropout samples
    mc_dropout_samples = [1, 10, 20, 30, 40, 50]

    result_path = os.path.join(args.output_dir, 'results')
    os.makedirs(result_path, exist_ok=True)

    for n_mcd in mc_dropout_samples:
        print(f"Computing metrics for {n_mcd} MC dropout samples")
        results = compute_mc_dropout_metrics(teacher, test_set_preprocessed, n=n_mcd)
        with open(os.path.join(result_path, f'results_{n_mcd}.json'), 'w') as f:
            json.dump(results, f)
        print(f"Results for {n_mcd} MC dropout samples saved to {result_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str)
    parser.add_argument('--teacher_model_save_dir', type=str)
    parser.add_argument('--output_dir', type=str, default="out")
    args = parser.parse_args()

    main(args)