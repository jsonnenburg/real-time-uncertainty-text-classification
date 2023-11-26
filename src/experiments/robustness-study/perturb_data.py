# adapt natural and synthetic noise from https://github.com/jasonwei20/eda_nlp/blob/master/code/augment.py
import os
import argparse
from noise import introduce_noise

import logging
logger = logging.getLogger(__name__)


ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True, type=str, help="Path to raw test data.")
ap.add_argument("--output_dir", required=True, type=str, help="Where to save modified test data.")
ap.add_argument("--p_sr", required=False, type=float, help="Percent of words in each sentence to be replaced by synonyms.")
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

if p_sr == p_ri == p_rs == p_rd == 0:
    ap.error('At least one alpha should be greater than zero')

# we only allow one augmentation at a time
if p_sr > 0:
    assert p_ri == 0 and p_rs == 0 and p_rd == 0
elif p_ri > 0:
    assert p_sr == 0 and p_rs == 0 and p_rd == 0
elif p_rs > 0:
    assert p_sr == 0 and p_ri == 0 and p_rd == 0
elif p_rd > 0:
    assert p_sr == 0 and p_ri == 0 and p_rs == 0


def generate_noisy_test_data(input_data, output_file, p_sr, p_ri, p_rs, p_rd):

    try:
        os.makedirs(output_dir, exist_ok=True)
        writer = open(output_file, 'w')
        lines = input_data['text'].tolist()
        for i, line in enumerate(lines):
            parts = line[:-1].split('\t')
            label = parts[0]
            sentence = parts[1]
            aug_sentences = introduce_noise(sentence, p_sr=p_sr, p_ri=p_ri, p_rs=p_rs, p_rd=p_rd)
            for aug_sentence in aug_sentences:
                writer.write(label + "\t" + aug_sentence + '\n')

        writer.close()
    except Exception as e:
        logger.error("Failed to generate noisy sentences: " + str(e))
    print("Successfully generated noisy sentences for " + input_data + " and saved them to " + output_file + ".")


if __name__ == "__main__":

    generate_noisy_test_data(args.input, output_dir, p_sr=p_sr, p_ri=p_ri, p_rs=p_rs, p_rd=p_rd)
