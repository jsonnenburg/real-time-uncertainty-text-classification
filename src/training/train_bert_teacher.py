import argparse
import os
import json
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFTrainer, TFTrainingArguments

from src.models.bert_model import create_bert_config, MCDropoutBERTDoubleHead
from src.utils.metrics import accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score, pred_entropy_score, ece_score

from src.utils.loss_functions import aleatoric_loss


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


class AleatoricLossTrainer(TFTrainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_variances = outputs.get("log_variances")
        loss = aleatoric_loss(y_true=labels, y_pred_logits=logits, y_pred_log_variance=log_variances)
        return loss


def train_model(hidden_dropout, attention_dropout, classifier_dropout, batch_size, dataset, learning_rate=2e-5):
    config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MCDropoutBERTDoubleHead.from_pretrained('bert-base-uncased', config=config)

    # TODO: Tokenize the dataset
    # see mozafari2020 section for details


    training_args = TFTrainingArguments(
        output_dir=f"./results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}",
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        logging_dir=f'./logs_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}',
    )

    trainer = AleatoricLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        optimizers=([tf.keras.optimizers.Adam(learning_rate=learning_rate)], []),  # TODO: which scheduler?
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    model.save_pretrained(f"./model_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}")
    with open(f"./results_hd{hidden_dropout}_ad{attention_dropout}_cd{classifier_dropout}/eval_results.json", 'w') as f:
        json.dump(eval_results, f)

    return eval_results["f1_score"]

########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# define dropout probabilities for grid search
hidden_dropout_probs = [0.1, 0.2, 0.3]
attention_dropout_probs = [0.1, 0.2, 0.3]
classifier_dropout_probs = [0.1, 0.2, 0.3]

# grid search over dropout probabilities
best_f1 = 0
best_dropout = 0

for hidden_dropout in hidden_dropout_probs:
    for attention_dropout in attention_dropout_probs:
        for classifier_dropout in classifier_dropout_probs:
            try:
                f1 = train_model(dropout_prob, args.learning_rate, args.batch_size, dataset)  # TODO: adapt
                if f1 > best_f1:
                    best_F1 = f1
                    best_dropout = dropout_prob
            except Exception as e:
                print(f"Error with dropout {dropout_prob}: {e}")

# Retrain the best model on the combination of train and validation set
# Update your dataset to include both training and validation data
best_model = train_model(best_dropout, args.learning_rate, args.batch_size, combined_dataset)

# Save the final model
model.save_pretrained(f"./final_model_dropout_{best_dropout}")

# Clean-up code here (if needed)
