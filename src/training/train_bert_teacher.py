import argparse
import json
import os
import shutil

import tensorflow as tf
from keras.callbacks import TensorBoard
from transformers import BertConfig

import logging

from src.models.bert_model import create_bert_config, MCDropoutBERT, CustomTFSequenceClassifierOutput
from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.utils.inference import mc_dropout_predict
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               pred_entropy_score, ece_score)

from src.utils.loss_functions import aleatoric_loss
from src.utils.data import SimpleDataLoader, Dataset


class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super(HistorySaver, self).__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.file_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}: {logs}\n")


def compute_metrics(output):
    logits = None
    labels = None
    if isinstance(output, CustomTFSequenceClassifierOutput):
        logits = output.logits
        labels = output.labels
    elif isinstance(output, tuple):
        logits = output[1]
        labels = output[0]

    if logits and labels:
        class_predictions = logits.argmax(-1)
        prob_predictions = tf.nn.softmax(logits, axis=-1)
        labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else labels
        class_predictions_np = class_predictions.numpy() if isinstance(class_predictions,
                                                                       tf.Tensor) else class_predictions
        prob_predictions_np = prob_predictions.numpy() if isinstance(prob_predictions, tf.Tensor) else prob_predictions
        
        acc = accuracy_score(labels_np, class_predictions_np)
        prec = precision_score(labels_np, class_predictions_np)
        rec = recall_score(labels_np, class_predictions_np)
        f1 = f1_score(labels_np, class_predictions_np)
        nll = nll_score(labels_np, prob_predictions_np)
        bs = brier_score(labels_np, prob_predictions_np)
        entropy = pred_entropy_score(prob_predictions_np)
        ece = ece_score(labels_np, prob_predictions_np)
        return {"accuracy_score": acc,
                "precision_score": prec,
                "recall_score": rec,
                "f1_score": f1,
                "nll_score": nll,
                "brier_score": bs,
                "pred_entropy_score": entropy,
                "ece_score": ece,
                }


def compute_mc_dropout_metrics(labels, mean_predictions):
    y_pred = tf.argmax(mean_predictions, axis=-1)
    y_prob = tf.nn.sigmoid(mean_predictions)[:, 1:2]  # index 0 and 1 in y_prob correspond to class 0 and 1
    # y_prob are the probs. of class with label 1

    labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else labels
    y_pred_np = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred
    y_prob_np = y_prob.numpy().flatten() if isinstance(y_prob, tf.Tensor) else y_prob

    acc = accuracy_score(labels_np, y_pred_np)
    prec = precision_score(labels_np, y_pred)
    rec = recall_score(labels_np, y_pred)
    f1 = f1_score(labels_np, y_pred)
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


def train_model(config: BertConfig, dataset: Dataset, batch_size: int, learning_rate: float, epochs: int,
                max_length: int = 48, custom_loss=aleatoric_loss, mc_dropout_inference: bool = True,
                save_model: bool = False, training_final_model: bool = False):

    model = MCDropoutBERT.from_pretrained('bert-base-uncased', config=config, custom_loss_fn=custom_loss)

    if training_final_model:
        dir_prefix = "final"
    else:
        dir_prefix = "temp"

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, metrics=['accuracy'])  # use internal model loss (defined above)

    tokenized_dataset = {
        'train': bert_preprocess(dataset.train, max_length=max_length),
        'val': bert_preprocess(dataset.val, max_length=max_length) if dataset.val is not None else None,
        'test': bert_preprocess(dataset.test, max_length=max_length)
    }

    train_data = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tokenized_dataset['train'][0],
            'attention_mask': tokenized_dataset['train'][1]
        },
        tokenized_dataset['train'][2]  # labels
    ))
    train_data = train_data.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if tokenized_dataset['val'] is not None:
        val_data = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': tokenized_dataset['val'][0],
                'attention_mask': tokenized_dataset['val'][1]
            },
            tokenized_dataset['val'][2]
        ))
        val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_data = None

    test_data = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tokenized_dataset['test'][0],
            'attention_mask': tokenized_dataset['test'][1]
        },
        tokenized_dataset['test'][2]
    ))
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    log_dir = f'./{dir_prefix}_logs_hd{int(config.hidden_dropout_prob * 100):03d}_ad{int(config.attention_dropout * 100):03d}_cd{config.classifier_dropout}'
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

    predictions = model.predict(eval_data)

    logits = predictions.logits 
    labels = eval_data[1] 

    eval_metrics = compute_metrics((labels, logits))
    with open(f"./{dir_prefix}_results_hd{config.hidden_dropout}_ad{config.attention_dropout}_cd{config.classifier_dropout}/eval_results.json", 'w') as f:
        json.dump(eval_metrics, f)
    
    if save_model:
        model.save_pretrained(f"./{dir_prefix}_model_hd{config.hidden_dropout}_ad{config.attention_dropout}_cd{config.classifier_dropout}")

    if mc_dropout_inference:
        mean_predictions, var_predictions = mc_dropout_predict(model, eval_data)
        mc_dropout_metrics = compute_mc_dropout_metrics(labels, mean_predictions)
        with open(f"./{dir_prefix}_results_hd{config.hidden_dropout}_ad{config.attention_dropout}_cd{config.classifier_dropout}/mc_dropout_metrics.json", 'w') as f:
            json.dump(mc_dropout_metrics, f)
        return mc_dropout_metrics

    return eval_metrics


########################################################################################################################
    

def main(args):
    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    data_loader.load_dataset()
    dataset = data_loader.get_dataset()

    # define dropout probabilities for grid search
    hidden_dropout_probs = [0.1, 0.2, 0.3]
    attention_dropout_probs = [0.1, 0.2, 0.3]
    classifier_dropout_probs = [0.1, 0.2, 0.3]

    # grid search over dropout probabilities
    best_dropout_combination = (None, None, None)
    best_f1 = 0

    for hidden_dropout in hidden_dropout_probs:
        for attention_dropout in attention_dropout_probs:
            for classifier_dropout in classifier_dropout_probs:
                current_dropout_combination = (hidden_dropout, attention_dropout, classifier_dropout)
                try:
                    logging.info(f"Training model with dropout combination {current_dropout_combination}.")
                    config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
                    eval_metrics = train_model(config=config, dataset=dataset,
                                               batch_size=args.batch_size, learning_rate=args.learning_rate,
                                               epochs=args.epochs, max_length=args.max_length,
                                               custom_loss=aleatoric_loss,
                                               mc_dropout_inference=args.mc_dropout_inference, save_model=False,
                                               training_final_model=False)
                    f1 = eval_metrics['eval_f1_score']  # note that eval_results is either eval_results or mc_dropout_results
                    if f1 > best_f1:
                        best_f1 = f1
                        best_dropout_combination = current_dropout_combination
                        logging.info(f"New best dropout combination: {best_dropout_combination}\n"
                                    f"New best f1 score: {best_f1}")
                    logging.info(f"Finished current iteration.\n")
                except Exception as e:
                    logging.error(f"Error with dropout combination {current_dropout_combination}: {e}.")

    # Retrain the best model on the combination of train and validation set
    # Update your dataset to include both training and validation data
    combined_training = dataset.train + dataset.val
    combined_dataset = Dataset(train=combined_training, test=dataset.test)

    if best_dropout_combination is None:
        raise ValueError("No best dropout combination saved.")
    else:
        best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1], best_dropout_combination[2])
        # train model, save results
        eval_metrics = train_model(best_config, combined_dataset, args.batch_size, args.learning_rate,
                                   args.epochs, args.max_length, custom_loss=aleatoric_loss, mc_dropout_inference=True,
                                   save_model=True, training_final_model=True)
        f1 = eval_metrics['eval_f1_score']
        logging.info(f"Final f1 score of best model configuration: {f1}")
    if args.cleanup:
        # remove all dirs that start with "temp"
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
