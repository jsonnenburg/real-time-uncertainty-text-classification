import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
import logging

from sklearn.metrics import classification_report

from src.utils.training import HistorySaver, BiLSTMConfig

logger = logging.getLogger(__name__)

from src.preprocessing.robustness_study.baselines_data_preprocessing import pad_sequences, get_embedding_matrix, \
    load_glove_embeddings, remove_stopwords
from src.utils.data import SimpleDataLoader
from src.utils.logger_config import setup_logging

from src.utils.metrics import (json_serialize, accuracy_score, precision_score, recall_score, f1_score, nll_score,
                               brier_score, ece_score, bald_score)

from src.models.bilstm_model import create_bilstm_config, BiLSTM


def compute_metrics(model, eval_data) -> dict:
    total_logits = []
    total_labels = []

    # iterate over all batches in eval_data
    start_time = time.time()
    for batch in eval_data:
        features, labels = batch
        logits = model(features, training=False)
        total_logits.extend(logits)
        total_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels) * 1000
    logger.info(f"Average inference time per sample: {average_inference_time:.0f} milliseconds.")

    all_labels = np.array(total_labels)

    y_prob = tf.nn.sigmoid(total_logits).numpy().reshape(all_labels.shape)
    y_pred = y_prob.round(0).astype(int)
    y_true = all_labels

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    nll = nll_score(y_true, y_prob)
    bs = brier_score(y_true, y_prob)
    ece = ece_score(y_true, y_pred, y_prob)
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "average_inference_time": json_serialize(average_inference_time),
        "accuracy_score": json_serialize(acc),
        "precision_score": json_serialize(prec),
        "recall_score": json_serialize(rec),
        "f1_score": json_serialize(f1),
        "nll_score": json_serialize(nll),
        "brier_score": json_serialize(bs),
        "ece_score": json_serialize(ece)
    }


def compute_mc_dropout_metrics(model, eval_data, n=50) -> dict:
    total_logits = []
    total_probs = []
    total_mean_logits = []
    total_var_logits = []
    total_labels = []

    start_time = time.time()
    for batch in eval_data:
        features, labels = batch

        samples = model.mc_dropout_sample(features, n=n)
        logits = samples['logit_samples']
        probs = samples['prob_samples']
        mean_logits = samples['mean_logits']
        var_logits = samples['var_logits']
        total_logits.append(logits.numpy())
        total_probs.append(probs.numpy())
        total_mean_logits.extend(mean_logits.numpy())
        total_var_logits.extend(var_logits.numpy())
        total_labels.extend(labels.numpy())
    total_time = time.time() - start_time
    average_inference_time = total_time / len(total_labels) * 1000
    logger.info(f"Average inference time per sample: {average_inference_time:.0f} milliseconds.")

    y_prob_samples = np.concatenate(total_probs, axis=0)

    all_labels = np.array(total_labels)

    y_prob_mcd = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
    y_pred_mcd = y_prob_mcd.round(0).astype(int)
    y_true = all_labels

    var_logits = np.array(total_var_logits).reshape(all_labels.shape)

    acc = accuracy_score(y_true, y_pred_mcd)
    prec = precision_score(y_true, y_pred_mcd)
    rec = recall_score(y_true, y_pred_mcd)
    f1 = f1_score(y_true, y_pred_mcd)
    nll = nll_score(y_true, y_prob_mcd)
    bs = brier_score(y_true, y_prob_mcd)
    ece = ece_score(y_true, y_pred_mcd, y_prob_mcd)
    bald = bald_score(y_prob_samples)
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred_mcd.tolist(),
        "y_prob": y_prob_mcd.tolist(),
        "var_logits": var_logits.tolist(),
        "average_inference_time": json_serialize(average_inference_time),
        "accuracy_score": json_serialize(acc),
        "precision_score": json_serialize(prec),
        "recall_score": json_serialize(rec),
        "f1_score": json_serialize(f1),
        "nll_score": json_serialize(nll),
        "brier_score": json_serialize(bs),
        "ece_score": json_serialize(ece),
        "bald_score": bald.tolist()
    }


def prepare_data(args):
    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    try:
        data_loader.load_dataset()
    except FileNotFoundError:
        logger.error("No dataset found.")
        raise
    dataset = data_loader.get_dataset()

    subset_size = 100
    dataset.train = dataset.train.sample(n=min(subset_size, len(dataset.train)), random_state=args.seed)
    dataset.val = dataset.val.sample(n=min(subset_size, len(dataset.val)), random_state=args.seed)
    dataset.test = dataset.test.sample(n=min(subset_size, len(dataset.test)), random_state=args.seed)

    dataset.train['text'] = dataset.train['text'].apply(remove_stopwords)
    dataset.val['text'] = dataset.val['text'].apply(remove_stopwords)
    dataset.test['text'] = dataset.test['text'].apply(remove_stopwords)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=args.max_vocab_size)
    tokenizer.fit_on_texts(dataset.train['text'])

    train_sequences = tokenizer.texts_to_sequences(dataset.train['text'])
    val_sequences = tokenizer.texts_to_sequences(dataset.val['text'])
    test_sequences = tokenizer.texts_to_sequences(dataset.test['text'])

    max_length = args.max_length

    X_train = pad_sequences(train_sequences, max_length=max_length)
    X_val = pad_sequences(val_sequences, max_length=max_length)
    X_test = pad_sequences(test_sequences, max_length=max_length)

    word_index = tokenizer.word_index

    glove_file = 'data/glove.6B.200d.txt'
    glove_embeddings = load_glove_embeddings(glove_file)
    embedding_matrix = get_embedding_matrix(word_index, glove_embeddings, 200)

    y_train = dataset.train['target'].values
    y_val = dataset.val['target'].values
    y_test = dataset.test['target'].values

    data = {'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'embedding_matrix': embedding_matrix,
            'max_length': max_length}

    return data, embedding_matrix, max_length


def train_bilstm(args, dataset: dict, embedding_matrix, max_length, config: BiLSTMConfig, save_model: bool = False):
    model = BiLSTM(config, embedding_matrix=embedding_matrix, sequence_length=max_length)

    config_info = {
        'embedding_dropout_rate': config.embedding_dropout_rate,
        'hidden_dropout_rate': config.hidden_dropout_rate,
        'lstm_units_1': config.lstm_units_1,
        'lstm_units_2': config.lstm_units_2,
        'max_length': max_length,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }

    if save_model:
        model_dir = os.path.join(args.output_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config_info, f)

        checkpoint_path = os.path.join(model_dir, 'cp-{epoch:02d}.ckpt')

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history_callback = HistorySaver(file_path=os.path.join(log_dir, 'bilstm_train.txt'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()],
                      run_eagerly=True)

        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint:
            model.load_weights(latest_checkpoint).expect_partial()
            logger.info(f"Found model files, loaded weights from {latest_checkpoint}")
            logger.info('Skipping training.')
        else:
            model.fit(dataset['train_data'],
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      validation_data=(dataset['test_data']),
                      callbacks=[cp_callback, history_callback])

        result_dir = os.path.join(args.output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)

        eval_metrics = compute_metrics(model, dataset['test_data'])
        with open(os.path.join(result_dir, 'results_stochastic_pass.json'), 'w') as f:
            json.dump(eval_metrics, f)
        logger.info(
            f"\n==== Classification report  (weight averaging) ====\n {classification_report(eval_metrics['y_true'], eval_metrics['y_pred'], zero_division=0)}")

        logger.info("Computing MC dropout metrics.")
        mc_dropout_metrics = compute_mc_dropout_metrics(model, dataset['test_data'])
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(mc_dropout_metrics, f)
        logger.info(
            f"\n==== Classification report  (MC dropout) ====\n {classification_report(mc_dropout_metrics['y_true'], mc_dropout_metrics['y_pred'], zero_division=0)}")
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        logger.info("Compiling model.")
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()],
                      run_eagerly=True)
        logger.info("Compiled model.")
        model.fit(dataset['train_data'],
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  validation_data=(dataset['val_data']))
        eval_metrics = compute_metrics(model, dataset['val_data'])
        logger.info(
            f"\n==== Classification report  (weight averaging) ====\n {classification_report(eval_metrics['y_true'], eval_metrics['y_pred'], zero_division=0)}")

        logger.info("Computing MC dropout metrics.")
        mc_dropout_metrics = compute_mc_dropout_metrics(model, dataset['val_data'])
        logger.info(
            f"\n==== Classification report  (MC dropout) ====\n {classification_report(mc_dropout_metrics['y_true'], mc_dropout_metrics['y_pred'], zero_division=0)}")

    return mc_dropout_metrics


def run_bilstm_grid_search(args, dataset: dict, embedding_matrix, max_length: int, embedding_dropout_rates: list,
                           hidden_dropout_rates: list, lstm_units_1: list, lstm_units_2: list):
    best_config = None
    best_f1 = 0
    for edr in embedding_dropout_rates:
        for hdr in hidden_dropout_rates:
            for lu1 in lstm_units_1:
                for lu2 in lstm_units_2:
                    config = create_bilstm_config(embedding_dropout_rate=edr,
                                                  hidden_dropout_rate=hdr,
                                                  lstm_units_1=lu1,
                                                  lstm_units_2=lu2)
                    try:
                        logger.info(f"Training BiLSTM with config: {config}")
                        result = train_bilstm(args, dataset=dataset, embedding_matrix=embedding_matrix,
                                              max_length=max_length, config=config, save_model=False)
                        f1 = result['f1_score']
                        if f1 > best_f1:
                            best_f1 = f1
                            best_config = config
                            logger.info(f"New best config: {config} with F1: {f1}")
                    except Exception as e:
                        logger.error(f"Failed to train BiLSTM with config: {config} due to: {e}")
    logger.info("Finished grid search.")
    logger.info(f"Best config: {best_config} with F1: {best_f1}")
    return best_f1, best_config


def main(args):
    # load and preprocess data
    data, embedding_matrix, max_length = prepare_data(args)

    train_data = tf.data.Dataset.from_tensor_slices((data['X_train'], data['y_train']))
    val_data = tf.data.Dataset.from_tensor_slices((data['X_val'], data['y_val']))
    test_data = tf.data.Dataset.from_tensor_slices((data['X_test'], data['y_test']))

    train_data = train_data.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = {'train_data': train_data,
               'val_data': val_data,
               'test_data': test_data
               }

    # grid search for best hyperparameters
    embedding_dropout_rates = [0.1]
    hidden_dropout_rates = [0.1]
    lstm_units_1 = [32]
    lstm_units_2 = [16]
    best_f1, best_config = run_bilstm_grid_search(args,
                                                  dataset=dataset,
                                                  embedding_matrix=embedding_matrix,
                                                  max_length=max_length,
                                                  embedding_dropout_rates=embedding_dropout_rates,
                                                  hidden_dropout_rates=hidden_dropout_rates,
                                                  lstm_units_1=lstm_units_1,
                                                  lstm_units_2=lstm_units_2)

    # train BiLSTM with best hyperparameters
    # prepare combined training and validation data
    data['X_train'] = tf.concat([data['X_train'], data['X_val']], axis=0)
    data['y_train'] = tf.concat([data['y_train'], data['y_val']], axis=0)

    train_data = tf.data.Dataset.from_tensor_slices((data['X_train'], data['y_train']))
    test_data = tf.data.Dataset.from_tensor_slices((data['X_test'], data['y_test']))

    train_data = train_data.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = {'train_data': train_data,
               'test_data': test_data,
               }

    best_config = BiLSTMConfig(embedding_dropout_rate=best_config.embedding_dropout_rate,
                               hidden_dropout_rate=best_config.hidden_dropout_rate,
                               lstm_units_1=best_config.lstm_units_1,
                               lstm_units_2=best_config.lstm_units_2)

    results = train_bilstm(args,
                           dataset=dataset,
                           embedding_matrix=embedding_matrix,
                           max_length=max_length,
                           config=best_config,
                           save_model=True)

    logger.info(f"Final F1: {results['f1_score']}")
    logger.info("Finished training.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_data_dir", type=str, help="Path to the input data directory.")
    ap.add_argument("--output_dir", type=str, help="Path to the output directory.")
    ap.add_argument("--max_length", type=int, help="Maximum length of the input sequences.")
    ap.add_argument("--max_vocab_size", type=int, default=10000, help="Maximum vocabulary size.")
    ap.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    ap.add_argument("--batch_size", type=int, help="Batch size.")
    ap.add_argument("--epochs", type=int, help="Number of epochs.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'bilstm_train.txt')
    setup_logging(logger, log_file_path)

    main(args)
