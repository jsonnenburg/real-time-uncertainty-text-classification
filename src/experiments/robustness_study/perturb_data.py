import os
import argparse
from noise import introduce_noise, WordDistributionByPOSTag
import pandas as pd
import logging

logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True, type=str, help="Path to raw test data.")
ap.add_argument("--output_dir", required=True, type=str, help="Where to save modified test data.")
args = ap.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

p_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def generate_noisy_test_data(input_file, output_dir, p_sr, p_pr, p_ri, p_rs, p_rd):
    print(f"Generating noisy sequences... \n"
          f"p_sr: {p_sr}, p_pr: {p_pr}, p_ri: {p_ri}, p_rs: {p_rs}, p_rd: {p_rd}.")
    input_data = pd.read_csv(input_file, sep='\t')

    if 'Unnamed: 0' in input_data.columns:
        input_data.drop('Unnamed: 0', axis=1, inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    format_params = f"_psr{int(p_sr * 100):03d}_ppr{int(p_pr * 100):03d}_pri{int(p_ri * 100):03d}_prs{int(p_rs * 100):03d}_prd{int(p_rd * 100):03d}"
    output_file_name = os.path.basename(input_file).split('.')[0] + format_params + '.csv'

    if p_pr > 0:
        word_distribution = WordDistributionByPOSTag(input_data['text'])
    else:
        word_distribution = None

    try:
        input_data['text'] = input_data['text'].apply(
            lambda x: introduce_noise(x, word_distribution, p_sr=p_sr, p_pr=p_pr, p_ri=p_ri, p_rs=p_rs, p_rd=p_rd))
    except Exception as e:
        logger.error("Failed to generate noisy sequences: " + str(e))

    input_data.to_csv(os.path.join(output_dir, output_file_name), sep='\t', index=False)
    print(f"Successfully generated noisy sequences for {input_file} and saved them to {output_dir}.")


def main():
    input_file = os.path.join(input_dir, 'test.csv')
    for p in p_values:
        generate_noisy_test_data(input_file, output_dir, p_sr=p, p_pr=0, p_ri=0,
                                 p_rs=0, p_rd=0)
        generate_noisy_test_data(input_file, output_dir, p_sr=0, p_pr=p, p_ri=0,
                                 p_rs=0, p_rd=0)
        generate_noisy_test_data(input_file, output_dir, p_sr=0, p_pr=0, p_ri=p,
                                 p_rs=0, p_rd=0)
        generate_noisy_test_data(input_file, output_dir, p_sr=0, p_pr=0, p_ri=0,
                                 p_rs=p, p_rd=0)
        generate_noisy_test_data(input_file, output_dir, p_sr=0, p_pr=0, p_ri=0,
                                 p_rs=0, p_rd=p)


if __name__ == "__main__":
    main()