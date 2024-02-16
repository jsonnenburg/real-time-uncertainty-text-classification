import json
import os
import numpy as np
import logging
import argparse
import tensorflow as tf

from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config
from src.utils.logger_config import setup_logging
from src.utils.loss_functions import bayesian_binary_crossentropy, null_loss
from src.utils.metrics import bald_score, f1_score, auc_score, json_serialize, accuracy_score, precision_score, \
    recall_score, nll_score, brier_score, ece_score
from src.utils.robustness_study import RobustnessStudyDataLoader


logger = logging.getLogger(__name__)


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


def preprocess_data_bert(data, max_length: int = 48, batch_size: int = 512):
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


def bert_teacher_mc_dropout(model, eval_data, n=50) -> dict:
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
    y_true = all_labels

    var_mcd = tf.exp(total_mean_log_variances).numpy().reshape(all_labels.shape)

    acc = accuracy_score(y_true, y_pred_mcd)
    prec = precision_score(y_true, y_pred_mcd)
    rec = recall_score(y_true, y_pred_mcd)
    f1 = f1_score(y_true, y_pred_mcd)
    auc = auc_score(y_true, y_prob_mcd)
    nll = nll_score(y_true, y_prob_mcd)
    bs = brier_score(y_true, y_prob_mcd)
    ece = ece_score(y_true, y_pred_mcd, y_prob_mcd)
    bald = bald_score(y_prob_samples)

    return {
        'y_true': y_true,
        'y_pred': y_pred_mcd,
        'y_prob': y_prob_mcd,
        'predictive_variance': var_mcd,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_score': auc,
        'nll_score': nll,
        'brier_score': bs,
        'ece_score': ece,
        'bald': bald,
    }


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
        total_logits.extend(outputs['logits'])
        total_log_variances.extend(outputs['log_variances'])
        total_labels.extend(labels.numpy())
        total_prob_samples.extend(outputs['prob_samples'])

    all_labels = np.array(total_labels)

    y_prob_mc = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
    y_pred_mc = y_prob_mc.round(0).astype(int)
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
    ece = ece_score(y_true, y_pred_mc, y_prob_mc)
    bald = bald_score(total_prob_samples_np)

    return {
        'y_true': y_true,
        'y_pred': y_pred_mc,
        'y_prob': y_prob_mc,
        'predictive_variance': var_mc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_score': auc,
        'nll_score': nll,
        'brier_score': bs,
        'ece_score': ece,
        'bald': bald,
    }


def perform_experiment_bert_teacher(model, preprocessed_data, n_trials):
    results = {}
    for typ in preprocessed_data:
        for level in preprocessed_data[typ]:
            logger.info(f"Computing results for {typ} - {level}")
            data_tf = preprocess_data_bert(preprocessed_data[typ][level][0]['data'])
            init_results_storage(results, typ, level)

            for i in range(n_trials):
                trial_results = bert_teacher_mc_dropout(model, data_tf, n=50)
                update_trial_results(results, typ, level, trial_results, i)

            finalize_results(results, typ, level)
            logger.info(f"Successfully computed results for {typ} - {level}")

    logger.info("Finished experiment for teacher model.")
    return results


def perform_experiment_bert_student(model, preprocessed_data, n_trials):
    results = {}
    for typ in preprocessed_data:
        for level in preprocessed_data[typ]:
            logger.info(f"Computing results for {typ} - {level}")
            data_tf = preprocess_data_bert(preprocessed_data[typ][level][0]['data'])
            init_results_storage(results, typ, level)

            for i in range(n_trials):
                trial_results = bert_student_monte_carlo(model, data_tf, n=50)
                update_trial_results(results, typ, level, trial_results, i)

            finalize_results(results, typ, level)
            logger.info(f"Successfully computed results for {typ} - {level}")

    logger.info("Finished experiment for student model.")
    return results


def init_results_storage(results, typ, level):
    if typ not in results:
        results[typ] = {}
    if level not in results[typ]:
        results[typ][level] = {}
    scalar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 'nll_score', 'brier_score', 'ece_score']
    per_input_metrics = ['y_true', 'y_pred', 'y_prob', 'predictive_variance', 'bald']
    for metric in scalar_metrics + per_input_metrics:
        results[typ][level][metric] = []
        if metric not in ['y_true', 'y_pred', 'y_prob']:
            results[typ][level][metric + '_std'] = []


def update_trial_results(results, typ, level, trial_results, trial_index):
    if trial_index == 0:
        # y_true is the same for all trials
        results[typ][level]['y_true'] = trial_results['y_true']
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 'nll_score', 'brier_score', 'ece_score']:
        results[typ][level][metric].append(trial_results[metric])
    for metric in ['y_prob', 'predictive_variance', 'bald']:
        if trial_index == 0:
            results[typ][level][metric] = [trial_results[metric]]
        else:
            results[typ][level][metric].append(trial_results[metric])


def finalize_results(results, typ, level):
    # Average scalar metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 'nll_score', 'brier_score', 'ece_score']:
        results[typ][level][metric + '_std'] = json_serialize(np.std(results[typ][level][metric], axis=0))
        results[typ][level][metric] = json_serialize(np.mean(results[typ][level][metric], axis=0))
    # Handle per-input metrics
    for metric in ['predictive_variance', 'bald']:
        results[typ][level][metric + '_std'] = np.std(np.array(results[typ][level][metric]), axis=0)
        results[typ][level][metric] = np.mean(np.array(results[typ][level][metric]), axis=0)
    results[typ][level]['y_prob'] = np.mean(results[typ][level]['y_prob'], axis=0)
    results[typ][level]['y_pred'] = np.array(results[typ][level]['y_prob']).round(0).astype(int)
    for metric in ['y_true', 'y_pred', 'y_prob', 'predictive_variance', 'bald']:
        results[typ][level][metric] = json_serialize(results[typ][level][metric])
        if metric not in ['y_true', 'y_pred', 'y_prob']:
            results[typ][level][metric + '_std'] = json_serialize(results[typ][level][metric + '_std'])


def main(args):
    # load data from data/robustness_study/noisy
    logger.info(f"Loading data from {args.data_dir}")
    data_loader = RobustnessStudyDataLoader(args.data_dir)
    data_loader.load_data()

    test_data = data_loader.data

    logger.info(f"Loading models from {args.teacher_model_path} and {args.student_model_path}")
    # load models
    bert_teacher = load_bert_model(args.teacher_model_path)
    bert_student = load_bert_model(args.student_model_path)

    # perform experiments
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Performing experiment...")
    logger.info("Teacher model: MC dropout sampling")
    results_bert_teacher = perform_experiment_bert_teacher(bert_teacher, test_data, n_trials=10)
    with open(os.path.join(args.output_dir, 'results_bert_teacher.json'), 'w') as f:
        json.dump(results_bert_teacher, f)

    logger.info("\nStudent model: MC sampling from logit space")
    results_bert_student = perform_experiment_bert_student(bert_student, test_data, n_trials=10)
    with open(os.path.join(args.output_dir, 'results_bert_student.json'), 'w') as f:
        json.dump(results_bert_student, f)

    logger.info("Finished experiment")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model_path', type=str)
    parser.add_argument('--student_model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'robustness_study_log.txt')
    setup_logging(logger, log_file_path)

    main(args)
