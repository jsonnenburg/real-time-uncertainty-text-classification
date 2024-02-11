import argparse
import json
import logging
import os
import re
import shutil

import time
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from src.utils.logger_config import setup_logging
from src.models.bert_model import create_bert_config, AleatoricMCDropoutBERT
from src.data.robustness_study.bert_data_preprocessing import bert_preprocess, get_tf_dataset

from src.utils.metrics import (serialize_metric, accuracy_score, precision_score, recall_score, f1_score, auc_score,
                               nll_score, brier_score, ece_score, bald_score)
from src.utils.loss_functions import null_loss, bayesian_binary_crossentropy
from src.utils.data import SimpleDataLoader, Dataset
from src.utils.training import HistorySaver

logger = logging.getLogger()


def compute_metrics(model, eval_data):
    total_logits = []
    total_log_variances = []
    total_labels = []

    # iterate over all batches in eval_data
    start_time = time.time()
    for batch in eval_data:
        features, labels = batch
        predictions = model(features, training=False)
        total_logits.extend(predictions.logits)
        total_log_variances.extend(predictions.log_variances)
        total_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels) * 1000
    logger.info(f"Average inference time per sample: {average_inference_time:.0f} milliseconds.")

    all_labels = np.array(total_labels)

    y_prob = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
    y_pred = y_prob.round(0).astype(int)
    var = tf.exp(total_log_variances).numpy().reshape(all_labels.shape)
    y_true = all_labels

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = auc_score(y_true, y_prob)
    nll = nll_score(y_true, y_prob)
    bs = brier_score(y_true, y_prob)
    ece = ece_score(y_true, y_pred, y_prob)
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "predictive_variance": var.tolist(),
        "average_inference_time": serialize_metric(average_inference_time),
        "accuracy_score": serialize_metric(acc),
        "precision_score": serialize_metric(prec),
        "recall_score": serialize_metric(rec),
        "f1_score": serialize_metric(f1),
        "auc_score": serialize_metric(auc),
        "nll_score": serialize_metric(nll),
        "brier_score": serialize_metric(bs),
        "ece_score": serialize_metric(ece)
    }


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
    logger.info(f"Average inference time per sample: {average_inference_time:.0f} milliseconds.")

    y_prob_samples = np.concatenate(total_probs, axis=0)

    all_labels = np.array(total_labels)
    
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
        "y_true": y_true.tolist(),
        "y_pred": y_pred_mcd.tolist(),
        "y_prob": y_prob_mcd.tolist(),
        "predictive_variance": var_mcd.tolist(),
        "average_inference_time": serialize_metric(average_inference_time),
        "accuracy_score": serialize_metric(acc),
        "precision_score": serialize_metric(prec),
        "recall_score": serialize_metric(rec),
        "f1_score": serialize_metric(f1),
        "auc_score": serialize_metric(auc),
        "nll_score": serialize_metric(nll),
        "brier_score": serialize_metric(bs),
        "ece_score": serialize_metric(ece),
        "bald_score": bald.tolist()
    }


def generate_file_path(dir_name: str, identifier: str) -> str:
    """Generate unique directory for each model and save outputs in subdirectories."""
    subdir_name = os.path.join(dir_name, f'{identifier}')
    os.makedirs(subdir_name, exist_ok=True)
    return subdir_name


def setup_config_directories(base_dir: str, lr: float, n_epochs: int, config, final_model: bool) -> dict:
    """
    Creates a directory for each model configuration and returns a dictionary with the paths to the results and
    model directories.

    :param base_dir: Base directory for all model configurations.
    :param config: BERT configuration.
    :param final_model: If True, the model directory will be named 'final' instead of 'temp'.
    :return:
    """
    prefix = 'final' if final_model else 'temp'
    suffix = f'e{int(n_epochs)}_lr{int(lr * 100000)}_hd{int(config.hidden_dropout_prob * 100):03d}_ad{int(config.attention_probs_dropout_prob * 100):03d}_cd{int(config.classifier_dropout * 100):03d}'
    config_dir = os.path.join(base_dir, f'{prefix}_{suffix}')

    paths = {
        "results_dir": os.path.join(config_dir, 'results'),
        "model_dir": os.path.join(config_dir, 'model'),
        "log_dir": os.path.join(base_dir, 'logs')
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def prepare_data(dataset: Dataset, max_length: int = 48, batch_size: int = 32):
    tokenized_dataset = {
        'train': bert_preprocess(dataset.train, max_length=max_length),
        'val': bert_preprocess(dataset.val, max_length=max_length) if dataset.val is not None else None,
        'test': bert_preprocess(dataset.test, max_length=max_length)
    }

    train_data = get_tf_dataset(tokenized_dataset, 'train')
    train_data = train_data.shuffle(buffer_size=10000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    # handle case where we group train and val together (best model config) and fine-tune on both
    if tokenized_dataset['val'] is not None:
        val_data = get_tf_dataset(tokenized_dataset, 'val')
        val_data = val_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        val_data = None
    test_data = get_tf_dataset(tokenized_dataset, 'test')
    test_data = test_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return train_data, val_data, test_data


def train_model(paths: dict, data: dict, config, batch_size: int, learning_rate: float, epochs: int, max_length: int,
                save_model: bool = False):
    """
    Trains a teacher BERT model and records the validation set performance, either for one stochastic forward pass or
    for M stochastic forward passes, with dropout enabled (MC dropout).

    :param paths: Dictionary with paths to the log, results, and model directories.
    :param config:
    :param data:
    :param batch_size:
    :param learning_rate:
    :param epochs:
    :param max_length:
    :param save_model:
    :return: eval_metrics
    """
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']

    model = AleatoricMCDropoutBERT(config=config, custom_loss_fn=bayesian_binary_crossentropy(50))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={'classifier': bayesian_binary_crossentropy(50), 'log_variance': null_loss},
        metrics=[{'classifier': 'binary_crossentropy'}, tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )

    model_config_info = {
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
        "classifier_dropout": config.classifier_dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_length": max_length
    }
    with open(os.path.join(paths['model_dir'], 'config.json'), 'w') as f:
        json.dump(model_config_info, f)

    checkpoint_path = os.path.join(paths['model_dir'], 'cp-{epoch:02d}.ckpt')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history_callback = HistorySaver(file_path=os.path.join(paths['log_dir'], 'grid_search_log.txt'))

    latest_checkpoint = tf.train.latest_checkpoint(paths['model_dir'])
    if latest_checkpoint:
        model.load_weights(latest_checkpoint).expect_partial()
        logger.info(f"Found model files, loaded weights from {latest_checkpoint}")
        logger.info('Skipping training.')
    else:
        model.fit(
            train_data,
            validation_data=val_data if val_data is not None else test_data,
            epochs=epochs,
            callbacks=[cp_callback, history_callback]
        )

    if not save_model:
        files = os.listdir(paths['model_dir'])
        for file in files:
            if file.startswith('cp-'):
                os.remove(os.path.join(paths['model_dir'], file))

    eval_data = val_data if val_data is not None else test_data
    eval_metrics = compute_metrics(model, eval_data)
    with open(os.path.join(paths['results_dir'], 'results_stochastic_pass.json'), 'w') as f:
        json.dump(eval_metrics, f)
    logger.info(f"\n==== Classification report  (weight averaging) ====\n {classification_report(eval_metrics['y_true'], eval_metrics['y_pred'], zero_division=0)}")

    logger.info("Computing MC dropout metrics.")
    mc_dropout_metrics = compute_mc_dropout_metrics(model, eval_data)
    with open(os.path.join(paths['results_dir'], 'results.json'), 'w') as f:
        json.dump(mc_dropout_metrics, f)
    logger.info(
        f"\n==== Classification report  (MC dropout) ====\n {classification_report(mc_dropout_metrics['y_true'], mc_dropout_metrics['y_pred'], zero_division=0)}")
    return mc_dropout_metrics


def run_bert_grid_search(data: dict,
                         lr: list,
                         n_epochs: list,
                         hidden_dropout_probs: list,
                         attention_dropout_probs: list,
                         classifier_dropout_probs: list,
                         args) -> Tuple[float, Tuple[float, float, float], float, int]:
    """
    Wrapper function to run a grid search over the dropout probabilities of the teacher BERT model.

    :param data:
    :param hidden_dropout_probs:
    :param attention_dropout_probs:
    :param classifier_dropout_probs:
    :param args:
    :return: best_f1, best_dropout_combination
    """
    best_dropout_combination = (None, None, None)
    best_lr = None
    best_n_epochs = None
    best_f1 = 0
    updated_best_combination = False
    for epochs in n_epochs:
        for learning_rate in lr:
            for hidden_dropout in hidden_dropout_probs:
                for attention_dropout in attention_dropout_probs:
                    for classifier_dropout in classifier_dropout_probs:
                        current_dropout_combination = (hidden_dropout, attention_dropout, classifier_dropout)
                        try:
                            logger.info(f"Training intermediate model for {epochs} epochs with "
                                        f"learning rate: {learning_rate} "
                                        f"and dropout combination: {current_dropout_combination}.")
                            config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
                            paths = setup_config_directories(args.output_dir, learning_rate, epochs, config, final_model=False)
                            eval_metrics = train_model(paths=paths, config=config, data=data, batch_size=args.batch_size,
                                                       learning_rate=learning_rate, epochs=epochs,
                                                       max_length=args.max_length, save_model=False)
                            f1 = eval_metrics['f1_score']  # note that eval_results is either eval_results or mc_dropout_results
                            if f1 > best_f1:
                                best_f1 = f1
                                best_lr = learning_rate
                                best_n_epochs = epochs
                                best_dropout_combination = current_dropout_combination
                                logger.info(f"New best f1 score: {best_f1:.3f} for "
                                            f"dropout combination: {best_dropout_combination} "
                                            f"with learning rate of {best_lr} and {best_n_epochs} epochs.")
                                updated_best_combination = True
                            logger.info(f"Finished current iteration.\n")
                        except Exception as e:
                            logger.error(f"Error for current iteration at epochs {epochs}, LR {learning_rate}, "
                                         f"and dropout combination {current_dropout_combination}: {e}.")
                        if not updated_best_combination:
                            best_dropout_combination = current_dropout_combination
    logger.info(f'Finished grid-search, best f1 score found at {best_f1:.3f} for combination {best_dropout_combination} '
                f'with learning rate of {best_lr} and {best_n_epochs} epochs.')
    return best_f1, best_dropout_combination, best_lr, best_n_epochs


########################################################################################################################


def infer_final_model_config(base_dir):
    config = None
    pattern = r'^final_e(\d{1})_lr(\d{1})_hd(\d{3})_ad(\d{3})_cd(\d{3})$'

    for dir_name in os.listdir(base_dir):
        if dir_name.startswith('final'):
            match = re.match(pattern, dir_name)
            if match:
                e, lr, hd, ad, cd = match.groups()
                config = {
                    'epochs': int(e),
                    'learning_rate': int(lr) / 100000,
                    'hidden_dropout_prob': int(hd) / 100.0,
                    'attention_probs_dropout_prob': int(ad) / 100.0,
                    'classifier_dropout': int(cd) / 100.0
                }

    return config


def main(args):
    logger.info("Starting grid search.")

    tf.random.set_seed(args.seed)

    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    try:
        data_loader.load_dataset()
    except FileNotFoundError:
        logger.error("No dataset found.")
        raise
    dataset = data_loader.get_dataset()

    train_data, val_data, test_data = prepare_data(dataset, max_length=args.max_length, batch_size=args.batch_size)

    data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }

    # define dropout probabilities for grid search
    learning_rates = [2e-5, 3e-5, 5e-5]
    n_epochs = [3]
    hidden_dropout_probs = [0.2, 0.3]
    attention_dropout_probs = [0.2, 0.3]
    classifier_dropout_probs = [0.2, 0.3]

    # if subdir with "final" prefix already exists, skip grid search and load best model
    final_model_trained = any([f.startswith("final") for f in os.listdir(args.output_dir)])
    if not final_model_trained:
        best_f1, best_dropout_combination, best_learning_rate, best_n_epochs = run_bert_grid_search(data=data,
                                                                                                    lr=learning_rates,
                                                                                                    n_epochs=n_epochs,
                                                                                                    hidden_dropout_probs=hidden_dropout_probs,
                                                                                                    attention_dropout_probs=attention_dropout_probs,
                                                                                                    classifier_dropout_probs=classifier_dropout_probs,
                                                                                                    args=args)
    else:
        logger.info("Final model already trained, skipping grid search.")
        # infer the best dropout combination from final model directory
        final_model_config = infer_final_model_config(args.output_dir)
        best_learning_rate = final_model_config['learning_rate']
        best_n_epochs = final_model_config['epochs']
        best_dropout_combination = (final_model_config['hidden_dropout_prob'], final_model_config['attention_probs_dropout_prob'], final_model_config['classifier_dropout'])

    # Retrain the best model on the combination of train and validation set
    # Update your dataset to include both training and validation data
    combined_training = pd.concat([dataset.train, dataset.val])
    combined_dataset = Dataset(train=combined_training, test=dataset.test)

    train_data, val_data, test_data = prepare_data(combined_dataset, max_length=args.max_length, batch_size=args.batch_size)

    combined_data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }

    if args.save_datasets:
        data_dir = os.path.join(args.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        # if any csv files already exist, raise an error
        if any([os.path.exists(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]):
            logger.warning("Dataset files already exist, not saving.")
        else:
            logger.info("Saving datasets.")
            dataset.train.to_csv(os.path.join(data_dir, 'train.csv'), sep='\t')
            dataset.val.to_csv(os.path.join(data_dir, 'val.csv'), sep='\t')
            dataset.test.to_csv(os.path.join(data_dir, 'test.csv'), sep='\t')
            combined_dataset.train.to_csv(os.path.join(data_dir, 'combined_train.csv'), sep='\t')
            combined_dataset.test.to_csv(os.path.join(data_dir, 'combined_test.csv'), sep='\t')

    trained_best_model = False
    if best_dropout_combination is None:
        logger.error("No best dropout combination saved.")
    else:
        best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1], best_dropout_combination[2])
        best_paths = setup_config_directories(args.output_dir, best_learning_rate, best_n_epochs, best_config, final_model=True)
        logger.info("Training final model with best dropout combination.")
        results = train_model(paths=best_paths, config=best_config, data=combined_data, batch_size=args.batch_size,
                              learning_rate=best_learning_rate, epochs=best_n_epochs, max_length=args.max_length,
                              save_model=True)
        if results is not None:
            trained_best_model = True
        f1 = results['f1_score']
        logger.info(f"Final f1 score of best model configuration: {f1:.3f}")
    if args.cleanup and trained_best_model:
        for directory in os.listdir(args.output_dir):
            if os.path.isdir(directory) and directory.startswith("temp"):
                shutil.rmtree(directory)
    logger.info("Finished grid search.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_datasets', action='store_true')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove all subdirectories with temp prefix from output dir.')
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'grid_search_log.txt')
    setup_logging(logger, log_file_path)

    main(args)
