"""
Minimal run config test of grid searching BERT teacher model.
"""
import argparse
import os

import logging
import shutil

logger = logging.getLogger('__name__')

from src.models.bert_model import create_bert_config
from src.training.train_bert_teacher import run_bert_grid_search, train_model
from src.utils.data import SimpleDataLoader, Dataset


def main(args):
    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    data_loader.load_dataset()
    dataset = data_loader.get_dataset()

    # Use a smaller subset of the dataset for testing
    subset_size = 100
    dataset.train = dataset.train.sample(n=min(subset_size, len(dataset.train)), random_state=args.seed)
    dataset.val = dataset.val.sample(n=min(subset_size, len(dataset.val)), random_state=args.seed)
    dataset.test = dataset.test.sample(n=min(subset_size, len(dataset.test)), random_state=args.seed)

    # Limit the grid search to fewer combinations
    hidden_dropout_probs = [0.1]
    attention_dropout_probs = [0.1]
    classifier_dropout_probs = [0.1]

    best_f1, best_dropout_combination = run_bert_grid_search(dataset, hidden_dropout_probs, attention_dropout_probs,
                                                             classifier_dropout_probs, args)

    # Retrain the best model on a combination of train and validation set
    combined_training = dataset.train + dataset.val
    combined_dataset = Dataset(train=combined_training, test=dataset.test)

    if best_dropout_combination is None:
        raise ValueError("No best dropout combination saved.")
    else:
        best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1],
                                         best_dropout_combination[2])
        eval_metrics = train_model(best_config, combined_dataset, args.batch_size, args.learning_rate,
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
    parser.add_argument("--batch_size", type=int, default=16)  # Reduced for testing
    parser.add_argument("--epochs", type=int, default=1)  # Reduced for testing
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--mc_dropout_inference", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cleanup", type=bool, default=False)
    args = parser.parse_args()

    main(args)