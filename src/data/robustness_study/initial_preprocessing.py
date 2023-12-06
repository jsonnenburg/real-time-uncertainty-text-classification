import os

from shared_data_preprocessing import DataLoader, preprocess

from joblib import Parallel, delayed


DATA_PATH = "../../../data/robustness-study/raw/labeled_data.csv"
OUTPUT_PATH = "../../../data/robustness-study/preprocessed/"

# load data
data_loader = DataLoader(DATA_PATH)
data_loader.load_data()

# split data, splits following mozafari2020
df_train, df_val, df_test = data_loader.split(0.8, 0.1, 0.1)


# iterate preprocess over all rows
def parallel_apply(df, func, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(func)(row) for row in df)


df_train['text'] = parallel_apply(df_train['text'], preprocess)
df_val['text'] = parallel_apply(df_val['text'], preprocess)
df_test['text'] = parallel_apply(df_test['text'], preprocess)

# save preprocessed data
df_train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), sep='\t')
df_val.to_csv(os.path.join(OUTPUT_PATH, "val.csv"), sep='\t')
df_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), sep='\t')


def main():
    pass


if __name__ == '__main__':
    main()
