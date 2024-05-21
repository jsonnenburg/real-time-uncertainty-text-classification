"""
Script for adding correctly calculated ECE score to the results json file.
"""
import argparse
import json
from scipy.special import logit

from src.utils.metrics import ece_score_l1_tfp, json_serialize


def main(args):
    results_path = args.results_path

    with open(results_path, 'r') as f:
        results = json.load(f)

    y_true = results['y_true']

    y_pred_logits = logit(results['y_prob'])

    ece_l1 = ece_score_l1_tfp(y_true, y_pred_logits, n_bins=10)

    results['ece_score_l1'] = json_serialize(ece_l1)

    with open(results_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str)
    args = parser.parse_args()

    main(args)
