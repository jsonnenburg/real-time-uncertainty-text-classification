import json
import os

import numpy as np
from tqdm import tqdm

from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.models.bert_model import AleatoricMCDropoutBERT, create_bert_config
from src.utils.loss_functions import bayesian_binary_crossentropy, null_loss
from src.utils.metrics import bald_score
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
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )

    latest_checkpoint = tf.train.latest_checkpoint(model_path)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint).expect_partial()

    return model


def preprocess_data_bert(data, max_length: int = 48, batch_size: int = 32):
    input_ids, attention_masks, labels = bert_preprocess(data, max_length=max_length)
    data_tf = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        },
        labels
    ))
    data_tf.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return data_tf


def bert_teacher_mc_dropout(model, eval_data, n=30) -> dict:
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
        total_labels.extend(labels.numpy())

    y_prob_samples = np.concatenate(total_probs, axis=0)

    all_labels = np.concatenate(total_labels, axis=0)

    y_prob_mcd = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
    y_pred_mcd = y_prob_mcd.round(0).astype(int)

    var_mcd = tf.exp(total_mean_log_variances).numpy().reshape(all_labels.shape)

    bald = bald_score(y_prob_samples)
    return {
        "y_pred": y_pred_mcd.tolist(),
        "y_prob": y_prob_mcd.tolist(),
        "predictive_variance": var_mcd.tolist(),
        "bald_score": bald.tolist()
    }


def perform_experiment_bert_teacher(model, preprocessed_data, n_trials):
    # preprocess data for model
    # each model might require a custom method for preprocessing and inference
    # this could then be called in a loop over each perturbed test set

    # create dict in shape of preprocessed data dict

    results = {}

    # for each key in preprocessed data dict, iterate over subdicts and preprocess data
    for typ in preprocessed_data:
        for level in preprocessed_data[typ]:
            data = preprocessed_data[typ][level]
            data_tf = preprocess_data_bert(data)

            result_dict = {'y_pred': [], 'y_prob': []}

            for _ in tqdm(range(n_trials), desc=f'Performing inference for {typ} {level}'):
                results = bert_teacher_mc_dropout(model, data_tf, n=30)
                result_dict['y_pred'].append(results['y_pred'])
                result_dict['y_prob'].append(results['y_prob'])
                result_dict['predictive_variance'].append(results['predictive_variance'])
                result_dict['bald_score'].append(results['bald_score'])

            mean_y_pred = np.mean(result_dict['y_prob'], axis=0)
            std_y_pred = np.std(result_dict['y_prob'], axis=0)

            results[typ][level] = {
                'y_pred': mean_y_pred.tolist(),
                'y_prob': std_y_pred.tolist(),
            }

    return results


def perform_experiment_bert_student(model, preprocessed_data, n_trials):
    # loop over preprocessed data and perform inference for each n_type, n_level combination
    # -> monce carlo sampling
    # repeat n_trials times for each combination
    # compute mean and std of inference results for n_trials ->

    # what about BALD?
    pass


def main(args):
    # load teacher

    # load student

    # load data from data/robustness_study/noisy
    data_loader = RobustnessStudyDataLoader(args.data_dir)
    data_loader.load_data()

    test_data = data_loader.data

    bert_teacher = load_bert_model(args.teacher_model_path)
    bert_student = load_bert_model(args.student_model_path)

    # perform inference on data with models
    # - need uncertainty estimates -> MCD teacher model
    # - repeat multiple (10) times to compute mean and std of inference results (statistical significance)
    results_bert_teacher = perform_experiment_bert_teacher(bert_teacher, test_data, n_trials=20)

    # save results (what format?)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model_path', type=str, required=True)
    parser.add_argument('--student_model_path', type=str, required=True)
    parser.add_argument('--bilstm_model_path', type=str, required=True)
    parser.add_argument('--cnn_model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)





