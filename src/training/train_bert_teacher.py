import argparse
import os
import json
from typing import Dict

import tensorflow as tf
from transformers import TFTrainer, TFTrainingArguments, BertConfig

from src.models.bert_model import create_bert_config, MCDropoutBERTDoubleHead
from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               pred_entropy_score, ece_score)

from src.utils.loss_functions import aleatoric_loss
from src.utils.data import SimpleDataLoader


def compute_metrics(pred):
    # TODO: write unit tests, adapt to BERT outputs
    labels = pred.label_ids
    class_predictions = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, class_predictions)
    prec = precision_score(labels, class_predictions)
    rec = recall_score(labels, class_predictions)
    f1 = f1_score(labels, class_predictions)
    nll = nll_score(labels, pred.predictions)
    bs = brier_score(labels, pred.predictions)
    entropy = pred_entropy_score(pred.predictions)
    ece = ece_score(labels, pred.predictions)
    return {"accuracy_score": acc,
            "precision_score": prec,
            "recall_score": rec,
            "f1_score": f1,
            "nll_score": nll,
            "brier_score": bs,
            "pred_entropy_score": entropy,
            "ece_score": ece,
            }


class AleatoricLossTrainer(TFTrainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_variances = outputs.get("log_variances")
        loss = aleatoric_loss(y_true=labels, y_pred_logits=logits, y_pred_log_variance=log_variances)
        return loss


def train_model(config: BertConfig, dataset: Dict, batch_size: int, learning_rate: float, epochs: int,
                save_model: bool = False, training_final_model: bool = False):
    model = MCDropoutBERTDoubleHead.from_pretrained('bert-base-uncased', config=config)

    if training_final_model:
        dir_prefix = "final"
    else:
        dir_prefix = "temp"

    # TODO: Tokenize the dataset
    # settings?
    # see mozafari2020 section for details
    tokenized_dataset: Dict = dict(train=None, val=None, test=None)
    tokenized_dataset['train'] = bert_preprocess(dataset['train'])
    tokenized_dataset['val'] = bert_preprocess(dataset['val']) if dataset['val'] is not None else None
    tokenized_dataset['test'] = bert_preprocess(dataset['test'])

    training_args = TFTrainingArguments(
        output_dir=f"./{dir_prefix}_results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir=f'./{dir_prefix}_logs_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}',
    )

    trainer = AleatoricLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"] if tokenized_dataset["val"] is not None else tokenized_dataset["test"],
        optimizers=([tf.keras.optimizers.Adam(learning_rate=learning_rate)], []),  # TODO: which scheduler with which parameters?
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    if save_model:
        model.save_pretrained(f"./{dir_prefix}_model_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}")

    with open(f"./{dir_prefix}_results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}/eval_results.json", 'w') as f:
        json.dump(eval_results, f)

    return eval_results["f1_score"]

########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()


# placeholder for dataset
# TODO: implement dataset loader
data_loader = SimpleDataLoader(dataset_dir="data/robustness_study/preprocessed")
data_loader.load_dataset()
dataset = data_loader.get_dataset()

# define dropout probabilities for grid search
hidden_dropout_probs = [0.1, 0.2, 0.3]
attention_dropout_probs = [0.1, 0.2, 0.3]
classifier_dropout_probs = [0.1, 0.2, 0.3]

# grid search over dropout probabilities
best_f1 = 0
best_dropout = 0

best_dropout_combination = (None, None, None)

for hidden_dropout in hidden_dropout_probs:
    for attention_dropout in attention_dropout_probs:
        for classifier_dropout in classifier_dropout_probs:
            best_dropout_combination = (hidden_dropout, attention_dropout, classifier_dropout)
            try:
                config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
                f1 = train_model(config=config, dataset=dataset, batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs)
                if f1 > best_f1:
                    best_F1 = f1
                    best_dropout_combination = (hidden_dropout, attention_dropout, classifier_dropout)
            except Exception as e:
                print(f"Error with dropout combination {best_dropout_combination}: {e}")

# Retrain the best model on the combination of train and validation set
# Update your dataset to include both training and validation data
combined_dataset = dict(train=None, val=None, test=None)
combined_training = dataset['train'] + dataset['val']   # placeholder is None
combined_dataset['train'] = bert_preprocess(combined_training)
combined_dataset['val'] = None
combined_dataset['test'] = dataset['test']

if best_dropout_combination is None:
    raise ValueError("No best dropout combination saved.")
else:
    best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1], best_dropout_combination[2])
    # train model, save results
    best_f1 = train_model(best_config, combined_dataset, args.learning_rate, args.batch_size, args.epochs, save_model=True, training_final_model=True)

# clean up all temp files if necessary
