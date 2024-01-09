"""
Minimal run config test of grid searching BERT teacher model.
"""
import argparse
import os

import shutil

import logging
import pandas as pd

from src.utils.logger_config import setup_logging

from src.models.bert_model import create_bert_config
from src.training.train_bert_teacher import run_bert_grid_search, train_model, setup_config_directories
from src.utils.data import SimpleDataLoader, Dataset

logger = logging.getLogger()


def main(args):
    logger.info("Starting grid search.")

    data_loader = SimpleDataLoader(dataset_dir=args.input_data_dir)
    data_loader.load_dataset()
    dataset = data_loader.get_dataset()

    # Use a smaller subset of the dataset for testing
    subset_size = 100
    dataset.train = dataset.train.sample(n=min(subset_size, len(dataset.train)), random_state=args.seed)
    dataset.val = dataset.val.sample(n=min(subset_size, len(dataset.val)), random_state=args.seed)
    dataset.test = dataset.test.sample(n=min(subset_size, len(dataset.test)), random_state=args.seed)

    # Limit the grid search to fewer combinations
    hidden_dropout_probs = [0.7]
    attention_dropout_probs = [0.7]
    classifier_dropout_probs = [0.7]

    best_f1, best_dropout_combination = run_bert_grid_search(dataset, hidden_dropout_probs, attention_dropout_probs,
                                                             classifier_dropout_probs, args)

    # Retrain the best model on a combination of train and validation set
    combined_training = pd.concat([dataset.train, dataset.val])
    combined_dataset = Dataset(train=combined_training, test=dataset.test)

    if args.save_datasets:
        logger.info("Saving datasets.")
        data_dir = os.path.join(args.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        # if any csv files already exist, raise an error
        if any([os.path.exists(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]):
            logger.warning("Dataset files already exist.")
        else:
            dataset.train.to_csv(os.path.join(data_dir, 'train.csv'), sep='\t')
            dataset.val.to_csv(os.path.join(data_dir, 'val.csv'), sep='\t')
            dataset.test.to_csv(os.path.join(data_dir, 'test.csv'), sep='\t')
            combined_dataset.train.to_csv(os.path.join(data_dir, 'combined_train.csv'), sep='\t')
            combined_dataset.test.to_csv(os.path.join(data_dir, 'combined_test.csv'), sep='\t')

    if best_dropout_combination is None:
        raise ValueError("No best dropout combination saved.")
    else:
        best_config = create_bert_config(best_dropout_combination[0], best_dropout_combination[1],
                                         best_dropout_combination[2])
        best_paths = setup_config_directories(args.output_dir, best_config, final_model=True)
        eval_metrics = train_model(paths=best_paths, config=best_config, dataset=combined_dataset,
                                   batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs,
                                   max_length=args.max_length, mc_dropout_inference=args.mc_dropout_inference,
                                   save_model=True)
        f1 = eval_metrics['f1_score']
        logger.info(f"Final f1 score of best model configuration: {f1}")
    if args.cleanup:
        for directory in os.listdir("."):
            if os.path.isdir(directory) and directory.startswith("temp"):
                shutil.rmtree(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)  # Reduced for testing
    parser.add_argument('--epochs', type=int, default=1)  # Reduced for testing
    parser.add_argument('--max_length', type=int, default=48)
    parser.add_argument('-mcd', '--mc_dropout_inference', action='store_true', help='Enable MC dropout inference.')
    parser.add_argument('--output_dir', type=str, default="out")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_datasets', action='store_true')
    parser.add_argument('--cleanup', action='store_true', help='Remove all subdirectories with temp prefix from output dir.')
    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'grid_search_log.txt')
    setup_logging(logger, log_file_path)

    main(args)
