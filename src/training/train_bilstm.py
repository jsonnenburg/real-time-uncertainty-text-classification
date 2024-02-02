import argparse
import json
import os

import tensorflow as tf
import logging

from sklearn.metrics import classification_report

from src.utils.training import HistorySaver

logger = logging.getLogger(__name__)

from src.data.robustness_study.baselines_data_preprocessing import pad_sequences, get_embedding_matrix, load_glove_embeddings, remove_stopwords
from src.utils.data import SimpleDataLoader
from src.utils.logger_config import setup_logging

from src.models.bilstm_model import create_bilstm_config, BiLSTM


def compute_metrics(model, eval_data):
    pass


def compute_mc_dropout_metrics(model, eval_data):
    pass


def main(args):
    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    try:
        data_loader.load_dataset()
    except FileNotFoundError:
        logger.error("No dataset found.")
        raise
    dataset = data_loader.get_dataset()

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

    glove_file = 'data/glove.6B.100d.txt'
    glove_embeddings = load_glove_embeddings(glove_file)
    embedding_matrix = get_embedding_matrix(word_index, glove_embeddings, args.glove_vector_size)

    y_train = dataset.train['target'].values

    config = create_bilstm_config(embedding_dropout_rate=args.embedding_dropout_rate,
                                    hidden_dropout_rate=args.hidden_dropout_rate,
                                    lstm_units_1=args.lstm_units_1,
                                     lstm_units_2=args.lstm_units_2)

    model = BiLSTM(config, embedding_matrix=embedding_matrix, sequence_length=max_length)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                  run_eagerly=True)

    config_info = {
        'embedding_dropout_rate': args.embedding_dropout_rate,
        'hidden_dropout_rate': args.hidden_dropout_rate,
        'lstm_units_1': args.lstm_units_1,
        'lstm_units_2': args.lstm_units_2,
        'max_vocab_size': args.max_vocab_size,
        'glove_vector_size': args.glove_vector_size,
        'max_length': args.max_length,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    model_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f)

    checkpoint_path = os.path.join(model_dir, 'cp-{epoch:02d}.ckpt')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history_callback = HistorySaver(file_path=os.path.join(log_dir, 'bilstm_train.txt'))

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint).expect_partial()
        logger.info(f"Found model files, loaded weights from {latest_checkpoint}")
        logger.info('Skipping training.')
    else:
        model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, dataset.val['target'].values))

    eval_data = val_data if val_data is not None else test_data
    eval_metrics = compute_metrics(model, eval_data)
    with open(os.path.join(paths['results_dir'], 'results_stochastic_pass.json'), 'w') as f:
        json.dump(eval_metrics, f)
    logger.info(
        f"\n==== Classification report  (weight averaging) ====\n {classification_report(eval_metrics['y_true'], eval_metrics['y_pred'], zero_division=0)}")

    logger.info("Computing MC dropout metrics.")
    mc_dropout_metrics = compute_mc_dropout_metrics(model, eval_data)
    with open(os.path.join(paths['results_dir'], 'results.json'), 'w') as f:
        json.dump(mc_dropout_metrics, f)
    logger.info(
        f"\n==== Classification report  (MC dropout) ====\n {classification_report(mc_dropout_metrics['y_true'], mc_dropout_metrics['y_pred'], zero_division=0)}")
    return mc_dropout_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_data_dir", type=str, help="Path to the input data directory.")
    ap.add_argument("--output_dir", type=str, help="Path to the output directory.")
    ap.add_argument("--max_vocab_size", type=int, help="Maximum vocabulary size.")
    ap.add_argument("--glove_vector_size", type=int, help="GloVe vector size.")
    ap.add_argument("--max_length", type=int, help="Maximum length of the input sequences.")
    ap.add_argument("--embedding_dropout_rate", type=float, help="Embedding dropout rate.")
    ap.add_argument("--hidden_dropout_rate", type=float, help="Hidden dropout rate.")
    ap.add_argument("--lstm_units_1", type=int, help="Number of units in the first LSTM layer.")
    ap.add_argument("--lstm_units_2", type=int, help="Number of units in the second LSTM layer.")
    ap.add_argument("--learning_rate", type=float, help="Learning rate.")
    ap.add_argument("--batch_size", type=int, help="Batch size.")
    ap.add_argument("--epochs", type=int, help="Number of epochs.")
    args = ap.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'bilstm_train.txt')
    setup_logging(logger, log_file_path)

    main(args)
