import argparse
import json
import os

import pandas as pd

import logging

from sklearn.metrics import classification_report

from src.experiments.uncertainty_distillation.uncertainty_distillation import compute_student_metrics, \
    delete_all_but_latest_checkpoint
from src.utils.logger_config import setup_logging
from src.data.robustness_study.bert_data_preprocessing import transfer_data_bert_preprocess, transfer_get_tf_dataset, \
    bert_preprocess, get_tf_dataset
from src.models.bert_model import create_bert_config, AleatoricMCDropoutBERT
from src.utils.loss_functions import shen_loss, null_loss
from src.utils.data import Dataset
from src.utils.training import HistorySaver

import tensorflow as tf

logger = logging.getLogger()


def main(args):
    """
    Student knowledge distillation pipeline. Setup similar to the grid search testing pipeline.
    """
    logger.info("Starting distribution distillation with augmented transfer dataset.")

    with open(os.path.join(args.teacher_model_save_dir, 'config.json'), 'r') as f:
        teacher_config = json.load(f)

    if args.remove_dropout_layers:
        logger.info('Removing dropout layers.')
        student_model_config = create_bert_config(0.0, 0.0, 0.0)
    else:
        student_model_config = create_bert_config(teacher_config['hidden_dropout_prob'],
                                                  teacher_config['attention_probs_dropout_prob'],
                                                  teacher_config['classifier_dropout'])

    # initialize student model
    student_model = AleatoricMCDropoutBERT(student_model_config, custom_loss_fn=shen_loss(n_samples=args.m * args.k,
                                                                                          loss_weight=args.shen_loss_weight))
    # compile with shen loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    student_model.compile(
        optimizer=optimizer,
        loss={'classifier': shen_loss(n_samples=args.m * args.k, loss_weight=args.shen_loss_weight),
              'log_variance': null_loss},
        metrics=[{'classifier': 'binary_crossentropy'}, tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()],
        run_eagerly=True
    )
    logger.info('Student model compiled.')

    model_config_info = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "hidden_dropout_prob": teacher_config['hidden_dropout_prob'],
        "attention_probs_dropout_prob": teacher_config['attention_probs_dropout_prob'],
        "classifier_dropout": teacher_config['classifier_dropout']
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
    dataset.train = pd.read_csv(os.path.join(data_dir, 'transfer_train_augmented.csv'), sep='\t')
    dataset.val = pd.read_csv(os.path.join(data_dir, 'transfer_test.csv'), sep='\t')
    dataset.test = pd.read_csv(os.path.join(data_dir, 'test.csv'), sep='\t')  # ADDED ORIGINAL TEST SET HERE

    logger.info('Preprocessing data...')
    # prepare data for transfer learning
    tokenized_dataset = {
        'train': transfer_data_bert_preprocess(dataset.train, max_length=args.max_length),
        'val': transfer_data_bert_preprocess(dataset.val,
                                             max_length=args.max_length) if dataset.val is not None else None,
        'test': bert_preprocess(dataset.test, max_length=args.max_length)
    }

    train_data = transfer_get_tf_dataset(tokenized_dataset, 'train')
    train_data = train_data.shuffle(buffer_size=10000).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    # handle case where we group train and val together (best model config) and fine-tune on both
    if tokenized_dataset['val'] is not None:
        val_data = transfer_get_tf_dataset(tokenized_dataset, 'val')
        val_data = val_data.batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        val_data = None
    test_data = get_tf_dataset(tokenized_dataset, 'test')
    test_data = test_data.batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    logger.info('Data successfully preprocessed.')

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
    else:
        if latest_teacher_checkpoint:
            student_model.load_weights(latest_teacher_checkpoint).expect_partial()
            logger.info(f"Found teacher model files, loaded weights from {latest_teacher_checkpoint}")
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
    logger.info(f"Final f1 score of augmented distilled student model: {f1:.3f}")

    delete_all_but_latest_checkpoint(model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer_data_dir', type=str)
    parser.add_argument('--teacher_model_save_dir', type=str)
    parser.add_argument('--m', type=int, default=5,
                        help="Transfer sampling param to know which dataset to use.")
    parser.add_argument('--k', type=int, default=10,
                        help="Transfer sampling param to know which dataset to use.")
    parser.add_argument('--epistemic_only', action='store_true',
                        help="If true, only model epistemic uncertainty, else also model aleatoric uncertainty.")
    parser.add_argument('--version_identifier', type=str, default=None,
                        help="Version identifier for output dir.")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=48)
    parser.add_argument('--remove_dropout_layers', action='store_true',
                        help="No dropout layers (dropout prob. set to 0), as per Shen et al (2021).")
    parser.add_argument('--shen_loss_weight', type=float, default=1.0,
                        help="Weight for custom Shen et al. (2021) loss.")
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.epistemic_only:
        args.output_dir = os.path.join(args.output_dir, f'm{args.m}')
    else:
        args.output_dir = os.path.join(args.output_dir, f'm{args.m}_k{args.k}')
    if args.version_identifier is not None:
        # append version identifier to output dir (e.g., for experiments with different hyperparameters)
        args.output_dir = os.path.join(args.output_dir, args.version_identifier)
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'student_uncertainty_distillation_log.txt')
    setup_logging(logger, log_file_path)

    main(args)
