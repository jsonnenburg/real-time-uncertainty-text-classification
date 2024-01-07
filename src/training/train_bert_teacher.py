import argparse
import json
import logging
import os
import shutil

import time
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report

from logger_config import setup_logging
from src.models.bert_model import create_bert_config, AleatoricMCDropoutBERT
from src.data.robustness_study.bert_data_preprocessing import bert_preprocess, get_tf_dataset

from src.utils.inference import mc_dropout_predict
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               pred_entropy_score, ece_score)
from src.utils.loss_functions import aleatoric_loss, null_loss
from src.utils.data import SimpleDataLoader, Dataset
from src.utils.training import HistorySaver

logger = logging.getLogger()


def serialize_metric(value):
    if np.isscalar(value):
        if np.isnan(value):
            return 'NaN'
        elif isinstance(value, np.ndarray) or isinstance(value, tf.Tensor):
            return value.item()
        elif type(value) is np.float32:
            return value.item()
        else:
            return value


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

    if total_logits is not None and total_labels is not None:
        prob_predictions_np = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
        class_predictions_np = prob_predictions_np.round(0).astype(int)
        variances = tf.exp(total_log_variances).numpy().reshape(all_labels.shape)
        labels_np = all_labels

        acc = accuracy_score(labels_np, class_predictions_np)
        prec = precision_score(labels_np, class_predictions_np)
        rec = recall_score(labels_np, class_predictions_np)
        f1 = f1_score(labels_np, class_predictions_np)
        nll = nll_score(labels_np, prob_predictions_np)
        bs = brier_score(labels_np, prob_predictions_np)
        ece = ece_score(labels_np, class_predictions_np, prob_predictions_np)
        return {
            "y_true": labels_np.tolist(),
            "y_pred": class_predictions_np.tolist(),
            "y_prob": prob_predictions_np.tolist(),
            "variance": variances.tolist(),
            "average_inference_time": serialize_metric(average_inference_time),
            "accuracy_score": serialize_metric(acc),
            "precision_score": serialize_metric(prec),
            "recall_score": serialize_metric(rec),
            "f1_score": serialize_metric(f1),
            "nll_score": serialize_metric(nll),
            "brier_score": serialize_metric(bs),
            "ece_score": serialize_metric(ece)
        }
    else:
        logger.error("Metrics could not be computed successfully.")


def compute_mc_dropout_metrics(model, eval_data, n=20) -> dict:
    total_logits = []
    total_mean_variances = []  # mean of the variances (log variance predictor output) over the MC samples
    total_mean_logits = []  # mean of the logits (classifier output) over the MC samples
    total_variances = []  # variance of the logits (classifier output) over the MC samples
    total_labels = []

    total_uncertainties = []

    start_time = time.time()
    for batch in eval_data:
        features, labels = batch
        logits, mean_variances, mean_predictions, var_predictions, total_uncertainty = mc_dropout_predict(model, features, n=n)
        total_logits.append(logits.numpy())
        total_mean_variances.extend(mean_variances.numpy())
        total_mean_logits.extend(mean_predictions.numpy())
        total_variances.extend(var_predictions.numpy())
        total_uncertainties.extend(total_uncertainty.numpy())
        total_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels) * 1000
    logger.info(f"Average inference time per sample: {average_inference_time:.0f} milliseconds.")

    total_logits = np.concatenate(total_logits, axis=1)  # concatenate along the batch dimension

    if total_mean_logits and total_labels:
        all_labels = np.array(total_labels)
        mean_prob_predictions_np = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
        mean_class_predictions_np = mean_prob_predictions_np.round(0).astype(int)
        mean_variances_np = np.array(total_mean_variances).reshape(all_labels.shape)
        total_uncertainties_np = np.array(total_uncertainties).reshape(all_labels.shape)
        labels_np = all_labels

        sigmoid_logits = tf.nn.sigmoid(total_logits)
        reshaped_logits = tf.reshape(sigmoid_logits, [-1, sigmoid_logits.shape[-1]])
        batch_entropy_scores = pred_entropy_score(reshaped_logits)
        avg_entropy = np.mean(batch_entropy_scores)

        acc = accuracy_score(labels_np, mean_class_predictions_np)
        prec = precision_score(labels_np, mean_class_predictions_np)
        rec = recall_score(labels_np, mean_class_predictions_np)
        f1 = f1_score(labels_np, mean_class_predictions_np)
        nll = nll_score(labels_np, mean_prob_predictions_np)
        bs = brier_score(labels_np, mean_prob_predictions_np)
        ece = ece_score(labels_np, mean_class_predictions_np, mean_prob_predictions_np)
        return {
            "y_true": labels_np.tolist(),
            "y_pred": mean_class_predictions_np.tolist(),
            "y_prob": mean_prob_predictions_np.tolist(),
            "variance": mean_variances_np.tolist(),
            "total_uncertainty": total_uncertainties_np.tolist(),
            "average_inference_time": serialize_metric(average_inference_time),
            "accuracy_score": serialize_metric(acc),
            "precision_score": serialize_metric(prec),
            "recall_score": serialize_metric(rec),
            "f1_score": serialize_metric(f1),
            "nll_score": serialize_metric(nll),
            "brier_score": serialize_metric(bs),
            "avg_pred_entropy_score": serialize_metric(avg_entropy),
            "ece_score": serialize_metric(ece),
        }
    else:
        logger.error("MC dropout metrics could not be computed successfully.")


def generate_file_path(dir_name: str, identifier: str) -> str:
    """Generate unique directory for each model and save outputs in subdirectories."""
    subdir_name = os.path.join(dir_name, f'{identifier}')
    os.makedirs(subdir_name, exist_ok=True)
    return subdir_name


def setup_config_directories(base_dir: str, config, final_model: bool) -> dict:
    """
    Creates a directory for each model configuration and returns a dictionary with the paths to the results and
    model directories.

    :param base_dir: Base directory for all model configurations.
    :param config: BERT configuration.
    :param final_model: If True, the model directory will be named 'final' instead of 'temp'.
    :return:
    """
    prefix = 'final' if final_model else 'temp'
    suffix = f'hd{int(config.hidden_dropout_prob * 100):03d}_ad{int(config.attention_probs_dropout_prob * 100):03d}_cd{int(config.classifier_dropout * 100):03d}'
    config_dir = os.path.join(base_dir, f'{prefix}_{suffix}')

    paths = {
        "results_dir": os.path.join(config_dir, 'results'),
        "model_dir": os.path.join(config_dir, 'model'),
        "log_dir": os.path.join(base_dir, 'logs')
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def train_model(paths: dict, config, dataset: Dataset, batch_size: int, learning_rate: float, epochs: int,
                max_length: int = 48, mc_dropout_inference: bool = True, save_model: bool = False):
    """
    Trains a teacher BERT model and records the validation set performance, either for one stochastic forward pass or
    for M stochastic forward passes, with dropout enabled (MC dropout).

    :param paths: Dictionary with paths to the log, results, and model directories.
    :param config:
    :param dataset:
    :param batch_size:
    :param learning_rate:
    :param epochs:
    :param max_length:
    :param mc_dropout_inference:
    :param save_model:
    :return: eval_metrics
    """

    model = AleatoricMCDropoutBERT(config=config, custom_loss_fn=aleatoric_loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={'classifier': aleatoric_loss, 'log_variance': null_loss},
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )

    # TODO: move data stuff outside of training method -> want to do this only once in grid-search loop
    tokenized_dataset = {
        'train': bert_preprocess(dataset.train, max_length=max_length),
        'val': bert_preprocess(dataset.val, max_length=max_length) if dataset.val is not None else None,
        'test': bert_preprocess(dataset.test, max_length=max_length)
    }

    train_data = get_tf_dataset(tokenized_dataset, 'train')
    train_data = train_data.shuffle(buffer_size=1024).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    # handle case where we group train and val together (best model config) and fine-tune on both
    if tokenized_dataset['val'] is not None:
        val_data = get_tf_dataset(tokenized_dataset, 'val')
        val_data = val_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        val_data = None
    test_data = get_tf_dataset(tokenized_dataset, 'test')
    test_data = test_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # TODO: individual output subdir for each run
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

    tensorboard_callback = TensorBoard(log_dir=paths['log_dir'], histogram_freq=1)

    history_callback = HistorySaver(file_path=os.path.join(paths['log_dir'], 'grid_search_log.txt'))

    model.fit(
        train_data,
        validation_data=val_data if val_data is not None else test_data,
        epochs=epochs,
        callbacks=[cp_callback, history_callback, tensorboard_callback]
    )

    if not save_model:
        files = os.listdir(paths['model_dir'])
        for file in files:
            if file.startswith('cp.'):
                os.remove(os.path.join(paths['model_dir'], file))

    eval_data = val_data if val_data is not None else test_data
    if isinstance(eval_data, tf.data.Dataset):
        eval_metrics = compute_metrics(model, eval_data)
    else:
        logger.error("Eval data is not in TensorFlow-conforming dataset format.")
    with open(os.path.join(paths['results_dir'], 'eval_results.json'), 'w') as f:
        json.dump(eval_metrics, f)
    logger.info(f"\n==== Classification report  (weight averaging) ====\n {classification_report(eval_metrics['y_true'], eval_metrics['y_pred'])}")

    if mc_dropout_inference:
        logger.info("Computing MC dropout metrics.")
        mc_dropout_metrics = compute_mc_dropout_metrics(model, eval_data)
        with open(os.path.join(paths['results_dir'], 'mc_dropout_results.json'), 'w') as f:
            json.dump(mc_dropout_metrics, f)
        logger.info(
            f"\n==== Classification report  (MC dropout) ====\n {classification_report(mc_dropout_metrics['y_true'], mc_dropout_metrics['y_pred'])}")
        return mc_dropout_metrics

    return eval_metrics


def run_bert_grid_search(dataset: Dataset, hidden_dropout_probs: list, attention_dropout_probs: list, classifier_dropout_probs: list, args) -> Tuple[float, Tuple[float, float, float]]:
    """
    Wrapper function to run a grid search over the dropout probabilities of the teacher BERT model.

    :param dataset:
    :param hidden_dropout_probs:
    :param attention_dropout_probs:
    :param classifier_dropout_probs:
    :param args:
    :return: best_f1, best_dropout_combination
    """
    best_dropout_combination = (None, None, None)
    best_f1 = 0
    updated_best_combination = False
    for hidden_dropout in hidden_dropout_probs:
        for attention_dropout in attention_dropout_probs:
            for classifier_dropout in classifier_dropout_probs:
                current_dropout_combination = (hidden_dropout, attention_dropout, classifier_dropout)
                try:
                    logger.info(f"Training intermediate model with dropout combination {current_dropout_combination}.")
                    config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
                    paths = setup_config_directories(args.output_dir, config, final_model=False)
                    eval_metrics = train_model(paths=paths, config=config, dataset=dataset, batch_size=args.batch_size,
                                               learning_rate=args.learning_rate, epochs=args.epochs,
                                               max_length=args.max_length,
                                               mc_dropout_inference=args.mc_dropout_inference, save_model=False)
                    f1 = eval_metrics['f1_score']  # note that eval_results is either eval_results or mc_dropout_results
                    if f1 > best_f1:
                        best_f1 = f1
                        best_dropout_combination = current_dropout_combination
                        logger.info(f"New best dropout combination: {best_dropout_combination}\n"
                                    f"New best f1 score: {best_f1:.3f}")
                        updated_best_combination = True
                    logger.info(f"Finished current iteration.\n")
                except Exception as e:
                    logger.error(f"Error with dropout combination {current_dropout_combination}: {e}.")
                if not updated_best_combination:
                    best_dropout_combination = current_dropout_combination
    logger.info(f'Finished grid-search, best f1 score found at {best_f1:.3f} for combination {best_dropout_combination}.')
    return best_f1, best_dropout_combination


########################################################################################################################
    

def main(args):
    logger.info("Starting grid search.")

    tf.random.set_seed(args.seed)

    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    data_loader.load_dataset()
    dataset = data_loader.get_dataset()

    # define dropout probabilities for grid search
    hidden_dropout_probs = [0.1, 0.2, 0.3]
    attention_dropout_probs = [0.05, 0.2, 0.35]
    classifier_dropout_probs = [0.05, 0.2, 0.35]

    best_f1, best_dropout_combination = run_bert_grid_search(dataset, hidden_dropout_probs, attention_dropout_probs, classifier_dropout_probs, args)

    # Retrain the best model on the combination of train and validation set
    # Update your dataset to include both training and validation data
    combined_training = pd.concat([dataset.train, dataset.val])
    combined_dataset = Dataset(train=combined_training, test=dataset.test)

    if args.save_datasets:
        logger.info("Saving datasets.")
        data_dir = os.path.join(args.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        # if any csv files already exist, raise an error
        if any([os.path.exists(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]):
            logger.warning("Dataset files already exist, not saving.")
        else:
            dataset.train.to_csv(os.path.join(data_dir, 'train.csv'), sep='\t')
            dataset.val.to_csv(os.path.join(data_dir, 'val.csv'), sep='\t')
            dataset.test.to_csv(os.path.join(data_dir, 'test.csv'), sep='\t')
            combined_dataset.train.to_csv(os.path.join(data_dir, 'combined_train.csv'), sep='\t')
            combined_dataset.test.to_csv(os.path.join(data_dir, 'combined_test.csv'), sep='\t')

    if best_dropout_combination is None:
        logger.error("No best dropout combination saved.")
    else:
        best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1], best_dropout_combination[2])
        best_paths = setup_config_directories(args.output_dir, best_config, final_model=True)
        logger.info("Training final model with best dropout combination.")
        eval_metrics = train_model(paths=best_paths, config=best_config, dataset=combined_dataset,
                                   batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs,
                                   max_length=args.max_length, mc_dropout_inference=args.mc_dropout_inference,
                                   save_model=True)
        f1 = eval_metrics['f1_score']
        logger.info(f"Final f1 score of best model configuration: {f1:.3f}")
    if args.cleanup:
        for directory in os.listdir("."):
            if os.path.isdir(directory) and directory.startswith("temp"):
                shutil.rmtree(directory)
    logger.info("Finished grid search.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument('-mcd', '--mc_dropout_inference', action='store_true', help='Enable MC dropout inference.')
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
