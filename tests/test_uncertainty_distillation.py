import argparse
import json
import os

import numpy as np
import pandas as pd

import time

import logging

from sklearn.metrics import classification_report

from src.distribution_distillation.uncertainty_distillation import get_predictive_distributions
from src.utils.logger_config import setup_logging
from src.data.robustness_study.bert_data_preprocessing import transfer_data_bert_preprocess, transfer_get_tf_dataset, \
    bert_preprocess, get_tf_dataset
from src.models.bert_model import create_bert_config, AleatoricMCDropoutBERT
from src.training.train_bert_teacher import serialize_metric
from src.utils.loss_functions import shen_loss, null_loss
from src.utils.data import Dataset
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               ece_score, bald_score)
from src.utils.training import HistorySaver

import tensorflow as tf

logger = logging.getLogger()


def get_predictions(model, eval_data):
    total_logits = []
    total_log_variances = []
    total_labels = []

    # mc sampling
    total_prob_samples = []

    # iterate over all batches in eval_data
    start_time = time.time()
    for batch in eval_data:
        features, labels = batch
        outputs = model.monte_carlo_sample(features, n=50)
        total_logits.extend(outputs['logits'])
        total_log_variances.extend(outputs['log_variances'])
        total_labels.extend(labels.numpy())
        total_prob_samples.extend(outputs['prob_samples'])
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels) * 1000
    logger.info(f"Average inference time per sample: {average_inference_time:.0f} milliseconds.")

    all_labels = np.array(total_labels)

    prob_predictions_np = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
    class_predictions_np = prob_predictions_np.round(0).astype(int)
    variances_np = tf.exp(total_log_variances).numpy().reshape(all_labels.shape)
    labels_np = all_labels

    total_prob_samples_np = np.array(total_prob_samples)

    results = {'y_true': labels_np,
               'y_pred': class_predictions_np,
               'y_prob': prob_predictions_np,
               'predictive_variance': variances_np,
               'y_prob_mc': total_prob_samples_np,
               'average_inference_time': average_inference_time
               }

    return results


def compute_student_metrics(model, eval_data):
    predictions = get_predictions(model, eval_data)

    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    y_prob = predictions['y_prob']
    predictive_variance = predictions['predictive_variance']
    y_prob_mc = predictions['y_prob_mc']
    average_inference_time = predictions['average_inference_time']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    nll = nll_score(y_true, y_prob)
    bs = brier_score(y_true, y_prob)
    ece = ece_score(y_true, y_pred, y_prob)
    bald = bald_score(y_prob_mc)
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "predictive_variance": predictive_variance.tolist(),
        "average_inference_time": serialize_metric(average_inference_time),
        "accuracy_score": serialize_metric(acc),
        "precision_score": serialize_metric(prec),
        "recall_score": serialize_metric(rec),
        "f1_score": serialize_metric(f1),
        "nll_score": serialize_metric(nll),
        "brier_score": serialize_metric(bs),
        "ece_score": serialize_metric(ece),
        "bald_score": bald.tolist()
    }


def main(args):
    """
    Student knowledge distillation pipeline. Setup similar to the grid search testing pipeline.
    """
    logger.info("Starting distribution distillation.")

    with open(os.path.join(args.teacher_model_save_dir, 'config.json'), 'r') as f:
        teacher_config = json.load(f)

    student_model_config = create_bert_config(teacher_config['hidden_dropout_prob'],
                                              teacher_config['attention_probs_dropout_prob'],
                                              teacher_config['classifier_dropout'])

    # initialize student model
    student_model = AleatoricMCDropoutBERT(student_model_config, custom_loss_fn=shen_loss(n_samples=args.m*args.k))
    # compile with shen loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    student_model.compile(
        optimizer=optimizer,
        loss={'classifier': shen_loss(n_samples=args.m*args.k), 'log_variance': null_loss},
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        run_eagerly=True
    )
    logger.info('Student model compiled.')

    model_config_info = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "n": args.n
    }
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(model_config_info, f)

    if args.epistemic_only:
        logger.info('Distilling epistemic uncertainty only.')
        data_dir = os.path.join(args.transfer_data_dir, f'm{args.m}')
    else:
        logger.info('Distilling both epistemic and aleatoric uncertainty.')
        data_dir = os.path.join(args.transfer_data_dir, f'm{args.m}_k{args.k}')

    dataset = Dataset()
    dataset.train = pd.read_csv(os.path.join(data_dir, 'transfer_train.csv'), sep='\t')
    dataset.val = pd.read_csv(os.path.join(data_dir, 'transfer_test.csv'), sep='\t')
    dataset.test = pd.read_csv(os.path.join(data_dir, 'test.csv'), sep='\t')  # ADDED ORIGINAL TEST SET HERE

    subset_size = 10000
    dataset.train = dataset.train.sample(n=min(subset_size, len(dataset.train)), random_state=args.seed)
    if dataset.val is not None:
        dataset.val = dataset.val.sample(n=min(subset_size, len(dataset.val)), random_state=args.seed)
    dataset.test = dataset.test.sample(n=min(subset_size, len(dataset.test)), random_state=args.seed)

    # prepare data for transfer learning
    tokenized_dataset = {
        'train': transfer_data_bert_preprocess(dataset.train, max_length=args.max_length),
        'val': transfer_data_bert_preprocess(dataset.val, max_length=args.max_length) if dataset.val is not None else None,
        'test': bert_preprocess(dataset.test, max_length=args.max_length)
    }

    train_data = transfer_get_tf_dataset(tokenized_dataset, 'train')
    train_data = train_data.shuffle(buffer_size=1024).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    # handle case where we group train and val together (best model config) and fine-tune on both
    if tokenized_dataset['val'] is not None:
        val_data = transfer_get_tf_dataset(tokenized_dataset, 'val')
        val_data = val_data.batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        val_data = None
    test_data = get_tf_dataset(tokenized_dataset, 'test')
    test_data = test_data.batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    teacher_checkpoint_path = os.path.join(args.teacher_model_save_dir, 'cp-{epoch:02d}.ckpt')
    teacher_checkpoint_dir = os.path.dirname(teacher_checkpoint_path)
    latest_teacher_checkpoint = tf.train.latest_checkpoint(teacher_checkpoint_dir)
    if latest_teacher_checkpoint:
        student_model.load_weights(latest_teacher_checkpoint).expect_partial()
        logger.info(f"Found teacher model files, loaded weights from {latest_teacher_checkpoint}")

    checkpoint_path = os.path.join(model_dir, 'cp-{epoch:02d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    log_dir = os.path.join(args.output_dir, 'logs')

    history_callback = HistorySaver(file_path=os.path.join(log_dir, 'student_uncertainty_distillation_log.txt'))

    teacher_checkpoint_path = os.path.join(args.teacher_model_save_dir, 'cp-{epoch:02d}.ckpt')
    teacher_checkpoint_dir = os.path.dirname(teacher_checkpoint_path)
    latest_teacher_checkpoint = tf.train.latest_checkpoint(teacher_checkpoint_dir)

    # if we have a checkpoint for max epoch , load it and skip training
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
        student_model.load_weights(latest_checkpoint).expect_partial()
        logger.info(f"Found student model files, loaded weights from {latest_checkpoint}")
        logger.info('Skipping training.')
    elif latest_teacher_checkpoint:
        student_model.load_weights(latest_teacher_checkpoint).expect_partial()
        logger.info(f"Found teacher model files, loaded weights from {latest_teacher_checkpoint}")
    else:
        logger.info('Starting fine-tuning...')
        student_model.fit(
            train_data,
            validation_data=val_data if val_data is not None else test_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[cp_callback, history_callback]
        )
        logger.info('Finished fine-tuning.')

    result_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    results = compute_student_metrics(student_model, test_data)
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    logger.info(
        f"\n==== Classification report ====\n {classification_report(results['y_true'], results['y_pred'], zero_division=0)}")

    f1 = results['f1_score']
    logger.info(f"Final f1 score of distilled student model: {f1:.3f}")

    if args.save_predictive_distributions:
        logger.info('Computing and saving predictive distributions...')
        logger.info('Initializing teacher model...')
        config = create_bert_config(teacher_config['hidden_dropout_prob'],
                                    teacher_config['attention_probs_dropout_prob'],
                                    teacher_config['classifier_dropout'])
        teacher_model = AleatoricMCDropoutBERT(config, custom_loss_fn=shen_loss)
        teacher_model.compile(
            optimizer=optimizer,
            loss={'classifier': shen_loss, 'log_variance': null_loss},
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            run_eagerly=True
        )
        teacher_model.load_weights(latest_teacher_checkpoint).expect_partial()
        logger.info(f"Found teacher model files, loaded weights from {latest_teacher_checkpoint}")
        logger.info('Teacher model initialized.')
        t_pred_dist_info, s_pred_dist_info = get_predictive_distributions(teacher_model,
                                                                          student_model,
                                                                          eval_data=test_data,
                                                                          n=args.n,
                                                                          num_samples=args.predictive_distribution_samples)
        with open(os.path.join(result_dir, 'teacher_predictive_distribution_info.json'), 'w') as f:
            json.dump(t_pred_dist_info, f)
        with open(os.path.join(result_dir, 'student_predictive_distribution_info.json'), 'w') as f:
            json.dump(s_pred_dist_info, f)
        logger.info('Predictive distributions successfully computed and saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer_data_dir', type=str)
    parser.add_argument('--teacher_model_save_dir', type=str)
    parser.add_argument('--m', type=int, default=5, help="Transfer sampling param to know which dataset to use.")
    parser.add_argument('--k', type=int, default=10, help="Transfer sampling param to know which dataset to use.")
    parser.add_argument('--epistemic_only', action='store_true')  # if true, only model epistemic uncertainty, else also model aleatoric uncertainty
    parser.add_argument('--save_predictive_distributions', action='store_true')
    parser.add_argument('--predictive_distribution_samples', type=int, default=500,
                        help="Number of samples to draw from predictive distribution.")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=48)
    parser.add_argument('--n', type=int, default=20, help="Number of MC dropout samples to compute for student MCD metrics.")
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.epistemic_only:
        args.output_dir = os.path.join(args.output_dir, f'm{args.m}')
    else:
        args.output_dir = os.path.join(args.output_dir, f'm{args.m}_k{args.k}')
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'student_uncertainty_distillation_log.txt')
    setup_logging(logger, log_file_path)

    main(args)
