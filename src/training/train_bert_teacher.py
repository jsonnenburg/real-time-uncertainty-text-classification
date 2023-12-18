import argparse
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard

import logging

from src.models.bert_model import create_bert_config, AleatoricMCDropoutBERT
from src.data.robustness_study.bert_data_preprocessing import bert_preprocess, get_tf_dataset

from src.utils.inference import mc_dropout_predict
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               pred_entropy_score, ece_score)
from src.utils.loss_functions import aleatoric_loss, null_loss
from src.utils.data import SimpleDataLoader, Dataset
from src.utils.training import HistorySaver


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
    total_labels = []

    # iterate over all batches in eval_data
    for batch in eval_data:
        features, labels = batch
        predictions = model.predict(features)
        total_logits.extend(predictions.logits)
        total_labels.extend(labels.numpy())

    all_logits = np.array(total_logits)
    all_labels = np.array(total_labels)

    if total_logits is not None and total_labels is not None:
        prob_predictions_np = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
        class_predictions_np = prob_predictions_np.round(0).astype(int)
        labels_np = all_labels

        acc = accuracy_score(labels_np, class_predictions_np)
        prec = precision_score(labels_np, class_predictions_np)
        rec = recall_score(labels_np, class_predictions_np)
        f1 = f1_score(labels_np, class_predictions_np)
        nll = nll_score(labels_np, prob_predictions_np)
        bs = brier_score(labels_np, prob_predictions_np)
        avg_entropy = np.mean(pred_entropy_score(prob_predictions_np))
        ece = ece_score(labels_np, class_predictions_np, prob_predictions_np)
        return {
            "accuracy_score": serialize_metric(acc),
            "precision_score": serialize_metric(prec),
            "recall_score": serialize_metric(rec),
            "f1_score": serialize_metric(f1),
            "nll_score": serialize_metric(nll),
            "brier_score": serialize_metric(bs),
            "avg_pred_entropy_score": serialize_metric(avg_entropy),
            "ece_score": serialize_metric(ece)
        }


def compute_mc_dropout_metrics(labels, mean_predictions):
    y_pred = tf.argmax(mean_predictions, axis=-1)
    y_prob = tf.nn.sigmoid(mean_predictions)[:, 1:2]  # index 0 and 1 in y_prob correspond to class 0 and 1
    # y_prob are the probs. of class with label 1

    labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else labels
    y_pred_np = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred
    y_prob_np = y_prob.numpy().flatten() if isinstance(y_prob, tf.Tensor) else y_prob

    acc = accuracy_score(labels_np, y_pred_np)
    prec = precision_score(labels_np, y_pred_np)
    rec = recall_score(labels_np, y_pred_np)
    f1 = f1_score(labels_np, y_pred_np)
    nll = nll_score(labels_np, y_prob_np)
    bs = brier_score(labels_np, y_prob_np)
    entropy = pred_entropy_score(y_prob_np)
    ece = ece_score(labels_np, y_pred_np, y_prob_np)

    return {
        "accuracy_score": acc,
        "precision_score": prec,
        "recall_score": rec,
        "f1_score": f1,
        "nll_score": nll,
        "brier_score": bs,
        "pred_entropy_score": entropy,
        "ece_score": ece,
    }


def train_model(config, dataset: Dataset, output_dir: str, batch_size: int, learning_rate: float, epochs: int,
                max_length: int = 48, mc_dropout_inference: bool = True, save_model: bool = False,
                training_final_model: bool = False):
    """
    Trains a teacher BERT model and records the validation set performance, either for one stochastic forward pass or
    for M stochastic forward passes, with dropout enabled (MC dropout).

    :param config:
    :param dataset:
    :param output_dir:
    :param batch_size:
    :param learning_rate:
    :param epochs:
    :param max_length:
    :param mc_dropout_inference:
    :param save_model:
    :param training_final_model:
    :return: eval_metrics
    """

    model = AleatoricMCDropoutBERT(config=config, custom_loss_fn=aleatoric_loss)

    suffix = f'hd{int(config.hidden_dropout_prob * 100):03d}_ad{int(config.attention_probs_dropout_prob * 100):03d}_cd{int(config.classifier_dropout * 100):03d}'
    dir_prefix = 'temp' if not training_final_model else 'final'

    def generate_file_path(identifier: str) -> str:
        """Generate unique directory for each model and save outputs in subdirectories."""
        dir_name = os.path.join(output_dir, f'{dir_prefix}_{suffix}')
        os.makedirs(dir_name, exist_ok=True)
        subdir_name = os.path.join(dir_name, f'{identifier}')
        os.makedirs(subdir_name, exist_ok=True)
        return subdir_name

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={'classifier': aleatoric_loss, 'log_variance': null_loss},
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )  # use internal model loss (defined above)

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

    log_dir = generate_file_path('logs')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history_log_file = os.path.join(log_dir, "training_history.txt")
    history_saver = HistorySaver(file_path=history_log_file)

    history = model.fit(
        train_data,
        validation_data=val_data if val_data is not None else test_data,
        epochs=epochs,
        callbacks=[history_saver, tensorboard_callback]
    )
    
    eval_data = val_data if val_data is not None else test_data

    if isinstance(eval_data, tf.data.Dataset):
        eval_metrics = compute_metrics(model, eval_data)
    else:
        raise "Eval data is not in TensorFlow-conforming dataset format."

    model_config_info = {
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
        "classifier_dropout": config.classifier_dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_length": max_length
    }
    eval_metrics["model_config"] = model_config_info
    with open(generate_file_path('results') + '/eval_results.json', 'w') as f:
        json.dump(eval_metrics, f)
    
    if save_model:
        model.save_pretrained(generate_file_path('model'))

    if mc_dropout_inference:
        mean_predictions, var_predictions = mc_dropout_predict(model, eval_data)
        mc_dropout_metrics = compute_mc_dropout_metrics(labels, mean_predictions)
        mc_dropout_metrics["model_config"] = model_config_info
        with open(generate_file_path('results') + '/mc_dropout_metrics.json', 'w') as f:
            json.dump(mc_dropout_metrics, f)
        return mc_dropout_metrics

    return eval_metrics


def run_bert_grid_search(dataset, hidden_dropout_probs, attention_dropout_probs, classifier_dropout_probs, args):
    """

    :param dataset:
    :param hidden_dropout_probs:
    :param attention_dropout_probs:
    :param classifier_dropout_probs:
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
                    logging.info(f"Training model with dropout combination {current_dropout_combination}.")
                    config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
                    logging.info('Created BERT config.')
                    eval_metrics = train_model(config=config, dataset=dataset, output_dir=args.output_dir,
                                               batch_size=args.batch_size, learning_rate=args.learning_rate,
                                               epochs=args.epochs, max_length=args.max_length,
                                               mc_dropout_inference=args.mc_dropout_inference, save_model=False,
                                               training_final_model=False)
                    f1 = eval_metrics['eval_f1_score']  # note that eval_results is either eval_results or mc_dropout_results
                    if f1 > best_f1:
                        best_f1 = f1
                        best_dropout_combination = current_dropout_combination
                        logging.info(f"New best dropout combination: {best_dropout_combination}\n"
                                    f"New best f1 score: {best_f1}")
                        updated_best_combination = True
                    logging.info(f"Finished current iteration.\n")
                except Exception as e:
                    logging.error(f"Error with dropout combination {current_dropout_combination}: {e}.")
                if not updated_best_combination:
                    best_dropout_combination = current_dropout_combination
    logging.info(f'Finished grid-search, best f1 score found at {best_f1} for combination {best_dropout_combination}.')
    return best_f1, best_dropout_combination


########################################################################################################################
    

def main(args):

    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    data_loader.load_dataset()
    dataset = data_loader.get_dataset()

    # define dropout probabilities for grid search
    hidden_dropout_probs = [0.1, 0.2, 0.3]
    attention_dropout_probs = [0.1, 0.2, 0.3]
    classifier_dropout_probs = [0.1, 0.2, 0.3]

    best_f1, best_dropout_combination = run_bert_grid_search(dataset, hidden_dropout_probs, attention_dropout_probs, classifier_dropout_probs, args)

    # Retrain the best model on the combination of train and validation set
    # Update your dataset to include both training and validation data
    combined_training = dataset.train + dataset.val
    combined_dataset = Dataset(train=combined_training, test=dataset.test)

    if best_dropout_combination is None:
        raise ValueError("No best dropout combination saved.")
    else:
        best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1], best_dropout_combination[2])
        # train model, save results
        eval_metrics = train_model(best_config, combined_dataset, args.output_dir, args.batch_size, args.learning_rate,
                                   args.epochs, args.max_length, mc_dropout_inference=True,
                                   save_model=True, training_final_model=True)
        f1 = eval_metrics['eval_f1_score']
        logging.info(f"Final f1 score of best model configuration: {f1}")
    if args.cleanup:
        for directory in os.listdir("."):
            if os.path.isdir(directory) and directory.startswith("temp"):
                shutil.rmtree(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--mc_dropout_inference", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cleanup", type=bool, default=False)
    args = parser.parse_args()

    main(args)
