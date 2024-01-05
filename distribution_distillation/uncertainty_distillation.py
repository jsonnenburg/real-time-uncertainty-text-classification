import argparse
import json
import os

import numpy as np
import pandas as pd

import time

import logging

from sklearn.metrics import classification_report

from logger_config import setup_logging
from src.data.robustness_study.bert_data_preprocessing import transfer_data_bert_preprocess, transfer_get_tf_dataset
from src.models.bert_model import create_bert_config, AleatoricMCDropoutBERT
from src.training.train_bert_teacher import serialize_metric
from src.utils.loss_functions import shen_loss, null_loss
from src.utils.data import Dataset
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               pred_entropy_score, ece_score)

import tensorflow as tf

logger = logging.getLogger()


def main(args):
    """
    Student knowledge distillation pipeline. Setup similar to the grid search testing pipeline.

    # TODO: Add grid-search for optimal dropout rate.
    """
    logger.info("Starting distribution distillation.")

    with open(os.path.join(args.teacher_model_save_dir, 'config.json'), 'r') as f:
        teacher_config = json.load(f)

    dataset = Dataset()
    dataset.train = pd.read_csv(os.path.join(args.transfer_data_dir, 'transfer_train.csv'), sep='\t')
    dataset.test = pd.read_csv(os.path.join(args.transfer_data_dir, 'transfer_test.csv'), sep='\t')

    subset_size = 100
    dataset.train = dataset.train.sample(n=min(subset_size, len(dataset.train)), random_state=args.seed)
    if dataset.val is not None:
        dataset.val = dataset.val.sample(n=min(subset_size, len(dataset.val)), random_state=args.seed)
    dataset.test = dataset.test.sample(n=min(subset_size, len(dataset.test)), random_state=args.seed)

    # prepare data for transfer learning
    tokenized_dataset = {
        'train': transfer_data_bert_preprocess(dataset.train, max_length=args.max_length),
        'val': transfer_data_bert_preprocess(dataset.val, max_length=args.max_length) if dataset.val is not None else None,
        'test': transfer_data_bert_preprocess(dataset.test, max_length=args.max_length)
    }

    train_data = transfer_get_tf_dataset(tokenized_dataset, 'train')
    train_data = train_data.shuffle(buffer_size=1024).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    # handle case where we group train and val together (best model config) and fine-tune on both
    if tokenized_dataset['val'] is not None:
        val_data = transfer_get_tf_dataset(tokenized_dataset, 'val')
        val_data = val_data.batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        val_data = None
    test_data = transfer_get_tf_dataset(tokenized_dataset, 'test')
    test_data = test_data.batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    student_model_config = create_bert_config(teacher_config['hidden_dropout_prob'],
                                              teacher_config['attention_probs_dropout_prob'],
                                              teacher_config['classifier_dropout'])

    # initialize student model
    student_model = AleatoricMCDropoutBERT(student_model_config, custom_loss_fn=shen_loss)

    # load teacher model weights into student model
    student_model.load_weights(args.teacher_model_save_dir).expect_partial()

    # compile with shen loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    student_model.compile(
        optimizer=optimizer,
        loss={'classifier': shen_loss, 'log_variance': null_loss},
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )
    logger.info('Student model successfully compiled.')

    student_model.fit(
        train_data,
        validation_data=val_data if val_data is not None else test_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    logger.info('Student model successfully fine-tuned.')

    # save fine-tuned student model
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    student_model.save(model_dir,  save_format='tf')
    model_config_info = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "dropout_rate": args.dropout_rate,
        "n": args.n
    }
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(model_config_info, f)
    logger.info('Student model successfully saved.')

    # obtain "cached" MC dropout predictions on test set
    logger.info('Computing MC dropout metrics...')
    total_logits = []
    total_probs = []
    total_log_variances = []
    total_mean_logits = []
    total_variances = []
    total_labels = []

    start_time = time.time()
    for batch in test_data:
        features, (labels, predictions) = batch
        outputs = student_model.cached_mc_dropout_predict(features, n=args.n, dropout_rate=args.dropout_rate)
        logits = outputs['logits']
        probs = outputs['probs']
        log_variances = outputs['log_variances']
        mean_predictions = outputs['mean_predictions']
        var_predictions = outputs['var_predictions']
        total_logits.append(logits.numpy())
        total_probs.append(probs.numpy())
        total_log_variances.append(log_variances.numpy())
        total_mean_logits.extend(mean_predictions.numpy())
        total_variances.extend(var_predictions.numpy())
        total_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels)
    logger.info(f"Average inference time per sample: {average_inference_time:.4f} seconds.")

    total_logits = np.concatenate(total_logits, axis=1)  # concatenate along the batch dimension
    total_probs = np.concatenate(total_probs, axis=1)
    total_log_variances = np.concatenate(total_log_variances, axis=1)

    if total_mean_logits and total_labels:
        all_labels = np.array(total_labels)
        mean_prob_predictions_np = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
        mean_class_predictions_np = mean_prob_predictions_np.round(0).astype(int)
        labels_np = all_labels

        acc = accuracy_score(labels_np, mean_class_predictions_np)
        prec = precision_score(labels_np, mean_class_predictions_np)
        rec = recall_score(labels_np, mean_class_predictions_np)
        f1 = f1_score(labels_np, mean_class_predictions_np)
        nll = nll_score(labels_np, mean_prob_predictions_np)
        bs = brier_score(labels_np, mean_prob_predictions_np)
        avg_entropy = np.mean(
            [pred_entropy_score(tf.nn.sigmoid(total_logits[:, i, :].squeeze())) for i in range(total_logits.shape[1])])
        ece = ece_score(labels_np, mean_class_predictions_np, mean_prob_predictions_np)
        metrics = {
            "logits": total_logits.tolist(),
            "probs": total_probs.tolist(),
            "log_variances": total_log_variances.tolist(),
            "y_true": labels_np.astype(int).tolist(),
            "y_pred": mean_class_predictions_np.tolist(),
            "y_prob": mean_prob_predictions_np.tolist(),
            "average_inference_time": serialize_metric(average_inference_time),
            "accuracy_score": serialize_metric(acc),
            "precision_score": serialize_metric(prec),
            "recall_score": serialize_metric(rec),
            "f1_score": serialize_metric(f1),
            "nll_score": serialize_metric(nll),
            "brier_score": serialize_metric(bs),
            "avg_pred_entropy_score": serialize_metric(avg_entropy),
            "ece_score": serialize_metric(ece)
        }
        logger.info(
            f"\n==== Classification report  (MC dropout) ====\n {classification_report(metrics['y_true'], metrics['y_pred'])}")
        result_dir = os.path.join(args.output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, 'student_mc_dropout_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        logger.info("MC dropout metrics successfully computed and saved.")
    else:
        logger.error("Failed to compute MC dropout metrics.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer_data_dir', type=str)
    parser.add_argument('--teacher_model_save_dir', type=str)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)  # Reduced for testing
    parser.add_argument('--epochs', type=int, default=1)  # Reduced for testing
    parser.add_argument('--max_length', type=int, default=48)
    parser.add_argument('--n', type=int, default=20, help="Number of MC dropout samples to compute for student MCD metrics.")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for MC dropout student model.")
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'student_uncertainty_distillation_log.txt')
    setup_logging(log_file_path)

    main(args)