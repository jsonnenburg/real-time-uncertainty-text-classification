# adapt natural and synthetic noise from https://github.com/jasonwei20/eda_nlp/blob/master/code/augment.py
import os
import argparse
from noise import introduce_noise, WordDistributionByPOSTag

import logging
logger = logging.getLogger(__name__)


ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True, type=str, help="Path to raw test data.")
ap.add_argument("--output_dir", required=True, type=str, help="Where to save modified test data.")
ap.add_argument("--p_sr", required=False, type=float, help="Percent of words in each sentence to be replaced by synonyms.")
ap.add_argument("--p_pr", required=False, type=float, help="Percent of words in each sentence to be replaced by similar POS-tagged words.")
ap.add_argument("--p_ri", required=False, type=float, help="Percent of words in each sentence to be inserted.")
ap.add_argument("--p_rs", required=False, type=float, help="Percent of words in each sentence to be swapped.")
ap.add_argument("--p_rd", required=False, type=float, help="Percent of words in each sentence to be deleted.")
args = ap.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

if args.p_sr is not None:
    p_sr = args.p_sr
else:
    p_sr = 0

if args.p_pr is not None:
    p_pr = args.p_pr
else:
    p_pr = 0

if args.p_ri is not None:
    p_ri = args.p_ri
else:
    p_ri = 0

if args.p_rs is not None:
    p_rs = args.p_rs
else:
    p_rs = 0

if args.p_rd is not None:
    p_rd = args.p_rd
else:
    p_rd = 0

if p_sr == p_pr == p_ri == p_rs == p_rd == 0:
    ap.error('At least one alpha should be greater than zero')

# we only allow one augmentation at a time
if p_sr > 0:
    assert p_pr == 0 and p_ri == 0 and p_rs == 0 and p_rd == 0
elif p_pr > 0:
    assert p_sr == 0 and p_ri == 0 and p_rs == 0 and p_rd == 0
elif p_ri > 0:
    assert p_sr == 0 and p_pr == 0 and p_rs == 0 and p_rd == 0
elif p_rs > 0:
    assert p_sr == 0 and p_pr == 0 and p_ri == 0 and p_rd == 0
elif p_rd > 0:
    assert p_sr == 0 and p_pr == 0 and p_ri == 0 and p_rs == 0


def generate_noisy_test_data(input_data, output_dir, p_sr, p_pr, p_ri, p_rs, p_rd):

    os.makedirs(output_dir, exist_ok=True)

    format_params = f"_psr{int(p_sr*1000):03d}_ppr{int(p_pr*1000):03d}_pri{int(p_ri*1000):03d}_prs{int(p_rs*1000):03d}_prd{int(p_rd*1000):03d}"

    output_file_name = os.path.basename(input_data).split('.')[0] + format_params + '.csv'

    if p_pr > 0:
        word_distribution = WordDistributionByPOSTag(input_data['text'])
    else:
        word_distribution = None

    try:
        input_data['text'].apply(lambda x: introduce_noise(x, word_distribution, p_sr=p_sr, p_pr=p_pr, p_ri=p_ri, p_rs=p_rs, p_rd=p_rd))
    except Exception as e:
        logger.error("Failed to generate noisy sentences: " + str(e))

    input_data.to_csv(os.path.join(output_dir, output_file_name), sep='\t')

    print("Successfully generated noisy sentences for " + input_data + " and saved them to " + output_dir + ".")


if __name__ == "__main__":

    generate_noisy_test_data(args.input, output_dir, p_sr=p_sr, p_pr=p_pr, p_ri=p_ri, p_rs=p_rs, p_rd=p_rd)
