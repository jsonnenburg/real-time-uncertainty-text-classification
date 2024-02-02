import json
import os

import numpy as np
from tqdm import tqdm

from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config
from src.utils.loss_functions import bayesian_binary_crossentropy, null_loss
from src.utils.metrics import bald_score, f1_score, serialize_metric
from src.utils.robustness_study import RobustnessStudyDataLoader

import argparse
import tensorflow as tf


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

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
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

    bald = bald_score(y_prob_samples)
    avg_bald = np.mean(bald)

    f1 = f1_score(y_true, y_pred_mcd)

    return {
        'y_true': y_true,
        'y_pred': y_pred_mcd,
        'y_prob': y_prob_mcd,
        'predictive_variance': var_mcd,
        'avg_bald': avg_bald,
        'f1_score': f1,
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
    variances_np = tf.exp(total_log_variances).numpy().reshape(all_labels.shape)
    y_true = all_labels

    total_prob_samples_np = np.array(total_prob_samples)
    bald = bald_score(total_prob_samples_np)
    avg_bald = np.mean(bald)

    f1 = f1_score(y_true, y_pred_mc)

    results = {
        'y_true': y_true,
        'y_pred': y_pred_mc,
        'y_prob': y_prob_mc,
        'predictive_variance': variances_np,
        'avg_bald': avg_bald,
        'f1_score': f1,
       }

    return results


def perform_experiment_bert_teacher(model, preprocessed_data, n_trials):
    results = {}
    for typ in preprocessed_data:
        if typ not in results:
            results[typ] = {}
        for level in preprocessed_data[typ]:
            if level not in results[typ]:
                results[typ][level] = {}
            data_tf = preprocess_data_bert(preprocessed_data[typ][level][0]['data'])

            result_dict = {'y_true': [], 'y_pred': [], 'y_prob': [], 'predictive_variance': [], 'avg_bald': [],
                           'f1_score': []}

            for _ in tqdm(range(n_trials), desc=f'Performing inference for {typ} {level}'):
                trial_results = bert_teacher_mc_dropout(model, data_tf, n=30)
                for key in result_dict.keys():
                    result_dict[key].extend(np.ravel(trial_results[key]))

            f1_mean = np.mean(result_dict['f1_score'])
            f1_std = np.std(result_dict['f1_score'])
            avg_bald = np.mean(result_dict['avg_bald'])

            results[typ][level] = {
                'f1_mean': serialize_metric(f1_mean),
                'f1_std': serialize_metric(f1_std),
                'avg_bald': serialize_metric(avg_bald),
            }

    return results


def perform_experiment_bert_student(model, preprocessed_data, n_trials):
    results = {}
    for typ in preprocessed_data:
        if typ not in results:
            results[typ] = {}
        for level in preprocessed_data[typ]:
            if level not in results[typ]:
                results[typ][level] = {}
            data_tf = preprocess_data_bert(preprocessed_data[typ][level][0]['data'])

            result_dict = {'y_true': [], 'y_pred': [], 'y_prob': [], 'predictive_variance': [], 'avg_bald': [], 'f1_score': []}

            for _ in tqdm(range(n_trials), desc=f'Performing inference for {typ} {level}'):
                trial_results = bert_student_monte_carlo(model, data_tf, n=30)
                for key in result_dict.keys():
                    result_dict[key].extend(np.ravel(trial_results[key]))

            f1_mean = np.mean(result_dict['f1_score'])
            f1_std = np.std(result_dict['f1_score'])
            avg_bald = np.mean(result_dict['avg_bald'])

            results[typ][level] = {
                'f1_mean': serialize_metric(f1_mean),
                'f1_std': serialize_metric(f1_std),
                'avg_bald': serialize_metric(avg_bald),
            }

    return results


def main(args):
    # load data from data/robustness_study/noisy
    data_loader = RobustnessStudyDataLoader(args.data_dir)
    data_loader.load_data()

    test_data = data_loader.data

    # load models
    bert_teacher = load_bert_model(args.teacher_model_path)
    bert_student = load_bert_model(args.student_model_path)

    # perform experiments
    os.makedirs(args.output_dir, exist_ok=True)

    results_bert_teacher = perform_experiment_bert_teacher(bert_teacher, test_data, n_trials=20)
    with open(os.path.join(args.output_dir, 'results_bert_teacher.json'), 'w') as f:
        json.dump(results_bert_teacher, f)

    results_bert_student = perform_experiment_bert_student(bert_student, test_data, n_trials=20)
    with open(os.path.join(args.output_dir, 'results_bert_student.json'), 'w') as f:
        json.dump(results_bert_student, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model_path', type=str)
    parser.add_argument('--student_model_path', type=str)
    parser.add_argument('--bilstm_model_path', type=str)
    parser.add_argument('--cnn_model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    main(args)





