import os
import argparse
from src.experiments.robustness_study.noise import introduce_noise, WordDistributionByPOSTag
from src.utils.processing import parallel_apply
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("--input_file_path", required=True, type=str, help="Path to raw test data.")
ap.add_argument("--output_dir", required=True, type=str, help="Where to save modified data.")
args = ap.parse_args()

input_file_path = args.input_file_path
output_dir = args.output_dir


def generate_noisy_test_data(input_file, output_dir, p_sr, p_pr, p_ri, p_rs, p_rd):
    logger.info(f"Generating noisy sequences... \n"
          f"p_sr: {p_sr}, p_pr: {p_pr}, p_ri: {p_ri}, p_rs: {p_rs}, p_rd: {p_rd}.")
    input_data = pd.read_csv(input_file, sep='\t')

    if 'Unnamed: 0' in input_data.columns:
        input_data.drop('Unnamed: 0', axis=1, inplace=True)

    if os.path.exists(output_dir):
        logger.info(f"Output directory {output_dir} already exists. \n"
                    f"Overwriting existing files.")
    else:
        os.makedirs(output_dir, exist_ok=True)

    format_params = f"_psr{int(p_sr * 100):03d}_ppr{int(p_pr * 100):03d}_pri{int(p_ri * 100):03d}_prs{int(p_rs * 100):03d}_prd{int(p_rd * 100):03d}"
    output_file_name = os.path.basename(input_file).split('.')[0] + format_params + '.csv'

    if p_pr > 0:
        word_distribution = WordDistributionByPOSTag(input_data['sequences'])
    else:
        word_distribution = None

    try:
        input_data['sequences'] = parallel_apply(input_data['sequences'], introduce_noise, word_distribution=word_distribution,
                                            p_sr=p_sr, p_pr=p_pr, p_ri=p_ri, p_rs=p_rs, p_rd=p_rd)
    except Exception as e:
        logger.error("Failed to generate noisy sequences: " + str(e))

    input_data.to_csv(os.path.join(output_dir, output_file_name), sep='\t', index=False)
    logger.info(f"Successfully generated noisy sequences for {input_file} and saved them to {output_dir}.")


def main():
    logger.info(f"Generating noisy sequences for {input_file_path}.")
    generate_noisy_test_data(input_file_path, output_dir, p_sr=0.1, p_pr=0, p_ri=0,
                                 p_rs=0.1, p_rd=0)


if __name__ == "__main__":
    main()
