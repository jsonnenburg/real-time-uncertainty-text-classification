import os

from shared_data_preprocessing import DataLoader, preprocess

DATA_PATH = "../../../data/robustness-study/raw/labeled_data.csv"
OUTPUT_PATH = "../../../data/robustness-study/preprocessed/"

# load data
data_loader = DataLoader(DATA_PATH)
data_loader.load_data()

# split data
df_train, df_val, df_test = data_loader.split(0.7, 0.15, 0.15)

# general preprocessing
# iterate preprocess over all rows

df_train['text'] = df_train['text'].apply(lambda x: preprocess(x))
df_val['text'] = df_val['text'].apply(lambda x: preprocess(x))
df_test['text'] = df_test['text'].apply(lambda x: preprocess(x))

# save preprocessed data
df_train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), sep='\t')
df_val.to_csv(os.path.join(OUTPUT_PATH, "val.csv"), sep='\t')
df_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), sep='\t')


def main():
    pass


if __name__ == '__main__':
    main()
