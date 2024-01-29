import argparse
import pandas as pd
import os


def main(args):
    def transform_df(args, subset: str = 'train'):
        df = pd.read_csv(os.path.join(args.input_data_dir, f'transfer_{subset}.csv'), sep='\t')
        grouped_df = df.groupby(['sequences', 'labels'])['predictions'].agg(list).reset_index()
        grouped_df.columns = ['sequences', 'labels', 'predictions']
        grouped_df.to_csv(os.path.join(args.output_data_dir, f'transfer_{subset}_grouped.csv'), sep='\t', index=False)
    try:
        transform_df(args, 'train')
        transform_df(args, 'test')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", default="data", type=str, help="Path to input data directory.")
    parser.add_argument("--output_data_dir", default="out", type=str, help="Path to output data directory.")
    args = parser.parse_args()
    main(args)
