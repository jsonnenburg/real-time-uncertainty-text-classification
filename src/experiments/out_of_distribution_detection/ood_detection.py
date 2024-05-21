import dataclasses
import json
import os
from dataclasses import dataclass

import numpy as np
import logging
import argparse

import pandas as pd
import tensorflow as tf

from src.preprocessing.robustness_study.bert_data_preprocessing import bert_preprocess
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config
from src.utils.logger_config import setup_logging
from src.utils.loss_functions import bayesian_binary_crossentropy, null_loss
from src.utils.metrics import bald_score, f1_score, auc_score, json_serialize, accuracy_score, precision_score, \
    recall_score, nll_score, brier_score, ece_score_l1_tfp


logger = logging.getLogger(__name__)


@dataclass
class Results:

    y_true: np.ndarray
    y_prob: np.ndarray
    var: np.ndarray
    acc: float
    prec: float
    rec: float
    f1: float
    auc: float
    nll: float
    bs: float
    ece: float
    bald: np.ndarray

    field_names = None


Results.field_names = [field.name for field in dataclasses.fields(Results)]


def load_bert_model(model_path):
    """
    Loads a custom AleatoricMCDropoutBERT model from a given path and returns it.
    :param model_path:
    :return: model
    """
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        model_config = json.load(f)

    config = create_bert_config(model_config['hidden_dropout_prob'],
                                model_config['attention_probs_dropout_prob'],
                                model_config['classifier_dropout'])

    model = AleatoricMCDropoutBERT(config=config, custom_loss_fn=bayesian_binary_crossentropy(50))

    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss={'classifier': bayesian_binary_crossentropy(50), 'log_variance': null_loss},
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    latest_checkpoint = tf.train.latest_checkpoint(model_path)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint).expect_partial()

    return model


def preprocess_data_bert(data, max_length: int = 48, batch_size: int = 2048):
    input_ids, attention_masks, labels = bert_preprocess(data, max_length=max_length)
    data_tf = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        },
        labels
    ))
    data_tf = data_tf.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return data_tf


def bert_teacher_mc_dropout(model, eval_data, n=50) -> Results:
    total_logits = []
    total_probs = []
    total_mean_logits = []
    total_mean_log_variances = []
    total_labels = []

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
        total_labels.append(labels.numpy())

    y_prob_samples = np.concatenate(total_probs, axis=0)

    all_labels = np.concatenate(total_labels, axis=0)

    y_prob_mcd = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
    y_pred_mcd = y_prob_mcd.round(0).astype(int)
    y_pred_logits_mcd = np.array(total_mean_logits).reshape(all_labels.shape)
    var_mcd = tf.exp(total_mean_log_variances).numpy().reshape(all_labels.shape)
    y_true = all_labels

    acc = accuracy_score(y_true, y_pred_mcd)
    prec = precision_score(y_true, y_pred_mcd)
    rec = recall_score(y_true, y_pred_mcd)
    f1 = f1_score(y_true, y_pred_mcd)
    auc = auc_score(y_true, y_prob_mcd)
    nll = nll_score(y_true, y_prob_mcd)
    bs = brier_score(y_true, y_prob_mcd)
    ece = ece_score_l1_tfp(y_true, y_pred_logits_mcd)
    bald = bald_score(y_prob_samples)

    return Results(y_true, y_prob_mcd, var_mcd, acc, prec, rec, f1, auc, nll, bs, ece, bald)


def bert_student_monte_carlo(model, eval_data, n=50):
    total_logits = []
    total_log_variances = []
    total_labels = []

    # mc samples
    total_prob_samples = []

    # iterate over all batches in eval_data
    for batch in eval_data:
        features, labels = batch
        outputs = model.monte_carlo_sample(features, n=n)
        total_logits.extend(outputs['mean_logits'])
        total_log_variances.extend(outputs['log_variances'])
        total_labels.extend(labels.numpy())
        total_prob_samples.extend(outputs['prob_samples'])

    all_labels = np.array(total_labels)

    y_prob_mc = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
    y_pred_mc = y_prob_mc.round(0).astype(int)
    y_pred_logits_mc = np.array(total_logits).reshape(all_labels.shape)
    var_mc = tf.exp(total_log_variances).numpy().reshape(all_labels.shape)
    y_true = all_labels

    total_prob_samples_np = np.array(total_prob_samples)

    acc = accuracy_score(y_true, y_pred_mc)
    prec = precision_score(y_true, y_pred_mc)
    rec = recall_score(y_true, y_pred_mc)
    f1 = f1_score(y_true, y_pred_mc)
    auc = auc_score(y_true, y_prob_mc)
    nll = nll_score(y_true, y_prob_mc)
    bs = brier_score(y_true, y_prob_mc)
    ece = ece_score_l1_tfp(y_true, y_pred_logits_mc)
    bald = bald_score(total_prob_samples_np)

    return Results(y_true, y_prob_mc, var_mc, acc, prec, rec, f1, auc, nll, bs, ece, bald)


def perform_experiment_bert_teacher(model, dataset: tf.data.Dataset, n_trials: int):
    results = {}

    for i in range(n_trials):
        trial_results = bert_teacher_mc_dropout(model, dataset, n=50)
        if i == 0:
            results['y_true'] = trial_results.y_true
        for metric in Results.field_names:
            if i == 0:
                results[metric] = [getattr(trial_results, metric)]
            else:
                results[metric].append(getattr(trial_results, metric))

    for metric in Results.field_names:
        results[metric] = json_serialize(np.mean(results[metric], axis=0))

    # add y_pred
    results['y_pred'] = json_serialize(np.round(results['y_prob'], 0))

    logger.info("Finished experiment for teacher model.")
    return results


def perform_experiment_bert_student(model, dataset: tf.data.Dataset, n_trials: int):
    results = {}

    for i in range(n_trials):
        trial_results = bert_student_monte_carlo(model, dataset, n=50)
        if i == 0:
            results['y_true'] = trial_results.y_true
        for metric in Results.field_names:
            if i == 0:
                results[metric] = [getattr(trial_results, metric)]
            else:
                results[metric].append(getattr(trial_results, metric))

    for metric in Results.field_names:
        results[metric] = json_serialize(np.mean(results[metric], axis=0))

    # add y_pred
    results['y_pred'] = json_serialize(np.round(results['y_prob'], 0))

    logger.info("Finished experiment for student model.")
    return results


def main(args):
    logger.info(f"Loading data from {args.data_dir}")
    data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), sep='\t', index_col=0)
    data_tf = preprocess_data_bert(data)

    logger.info("Performing experiment...")
    if args.run_for_teacher:
        bert_teacher = load_bert_model(args.teacher_model_path)
        logger.info("Teacher model: MC dropout sampling")
        results_bert_teacher = perform_experiment_bert_teacher(bert_teacher, data_tf, n_trials=args.n_trials)
        with open(os.path.join(args.output_dir, 'results_bert_teacher.json'), 'w') as f:
            json.dump(results_bert_teacher, f)

    if args.run_for_student:
        bert_student = load_bert_model(args.student_model_path)
        logger.info("Student model: MC sampling from logit space")
        results_bert_student = perform_experiment_bert_student(bert_student, data_tf, n_trials=args.n_trials)
        with open(os.path.join(args.output_dir, 'results_bert_student.json'), 'w') as f:
            json.dump(results_bert_student, f)

    if args.run_for_augmented_student:
        bert_augmented_student = load_bert_model(args.augmented_student_model_path)
        logger.info("Augmented student model: MC sampling from logit space")
        results_bert_augmented_student = perform_experiment_bert_student(bert_augmented_student, data_tf, n_trials=args.n_trials)
        with open(os.path.join(args.output_dir, 'results_bert_augmented_student.json'), 'w') as f:
            json.dump(results_bert_augmented_student, f)

    logger.info("Finished experiment")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model_path', type=str)
    parser.add_argument('--student_model_path', type=str)
    parser.add_argument('--augmented_student_model_path', type=str)
    parser.add_argument('--data_dir', type=str, help='Directory containing the preprocessed data set.')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--run_for_teacher', action='store_true')
    parser.add_argument('--run_for_student', action='store_true')
    parser.add_argument('--run_for_augmented_student', action='store_true')
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'ood_detection_log.txt')
    setup_logging(logger, log_file_path)

    main(args)
