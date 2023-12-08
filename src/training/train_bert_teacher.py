import argparse
import json
import os
import shutil
from typing import Dict

from transformers import TFTrainer, TFTrainingArguments, BertConfig, logger

from src.models.bert_model import create_bert_config, MCDropoutBERTDoubleHead
from src.data.robustness_study.bert_data_preprocessing import bert_preprocess
from src.utils.inference import mc_dropout_predict
from src.utils.metrics import (accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
                               pred_entropy_score, ece_score)

from src.utils.loss_functions import aleatoric_loss
from src.utils.data import SimpleDataLoader, Dataset


def compute_metrics(pred):
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


def compute_metrics_for_mc_dropout(labels, predictions):
    class_predictions = predictions.argmax(-1)
    acc = accuracy_score(labels, class_predictions)
    prec = precision_score(labels, class_predictions)
    rec = recall_score(labels, class_predictions)
    f1 = f1_score(labels, class_predictions)
    nll = nll_score(labels, predictions)
    bs = brier_score(labels, predictions)
    entropy = pred_entropy_score(predictions)
    ece = ece_score(labels, predictions)

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


def train_model(config: BertConfig, dataset: Dataset, batch_size: int, learning_rate: float, epochs: int,
                max_length: int = 48, mc_dropout_inference: bool = True, save_model: bool = False, training_final_model: bool = False):
    model = MCDropoutBERTDoubleHead.from_pretrained('bert-base-uncased', config=config)

    if training_final_model:
        dir_prefix = "final"
    else:
        dir_prefix = "temp"

    tokenized_dataset: Dict = dict(train=None, val=None, test=None)
    tokenized_dataset['train'] = bert_preprocess(dataset.train, max_length=max_length)
    tokenized_dataset['val'] = bert_preprocess(dataset.val, max_length=max_length) if dataset.val is not None else None
    tokenized_dataset['test'] = bert_preprocess(dataset.test, max_length=max_length)

    # default training parameters follow Devlin et al. (2019)! -> overwrite only learning rate
    training_args = TFTrainingArguments(
        output_dir=f"./{dir_prefix}_results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=10000,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f'./{dir_prefix}_logs_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}',
    )

    trainer = AleatoricLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"] if tokenized_dataset["val"] is not None else tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    if save_model:
        model.save_pretrained(f"./{dir_prefix}_model_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}")

    with open(f"./{dir_prefix}_results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}/eval_results.json", 'w') as f:
        json.dump(eval_results, f)

    if mc_dropout_inference:
        mc_dropout_predictions = mc_dropout_predict(model, tokenized_dataset['val'])
        mc_dropout_results = compute_metrics_for_mc_dropout(tokenized_dataset['val']['labels'], mc_dropout_predictions)
        with open(f"./{dir_prefix}_results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}/mc_dropout_metrics.json", 'w') as f:
            json.dump(mc_dropout_results, f)
        return mc_dropout_results
    return eval_results

########################################################################################################################

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
            current_dropout_combination = (hidden_dropout, attention_dropout, classifier_dropout)
            try:
                logger.info(f"Training model with dropout combination {current_dropout_combination}")
                config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
                eval_results = train_model(config=config,
                                           dataset=dataset,
                                           batch_size=args.batch_size,
                                           learning_rate=args.learning_rate,
                                           epochs=args.epochs,
                                           max_length=args.max_length,
                                           mc_dropout_inference=args.mc_dropout_inference,
                                           save_model=False,
                                           training_final_model=False)
                if args.mc_dropout_inference:
                    f1 = ...
                else:
                    f1 = eval_results['eval_f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_dropout_combination = current_dropout_combination
            except Exception as e:
                print(f"Error with dropout combination {current_dropout_combination}: {e}")

# Retrain the best model on the combination of train and validation set
# Update your dataset to include both training and validation data
combined_training = dataset.train + dataset.val
combined_dataset = Dataset(train=combined_training, test=dataset.test)

if best_dropout_combination is None:
    raise ValueError("No best dropout combination saved.")
else:
    best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1], best_dropout_combination[2])
    # train model, save results
    eval_results = train_model(best_config,
                               combined_dataset,
                               args.batch_size,
                               args.learning_rate,
                               args.epochs,
                               args.max_length,
                               mc_dropout_inference=True,
                               save_model=True,
                               training_final_model=True)

if args.cleanup:
   # remove all dirs that start with "temp"
    for directory in os.listdir("."):
        if os.path.isdir(directory) and directory.startswith("temp"):
            shutil.rmtree(directory)
