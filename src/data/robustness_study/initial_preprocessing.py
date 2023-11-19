import os

from shared_data_preprocessing import DataLoader, GeneralTextPreprocessor

DATA_PATH = "../../../data/robustness-study/raw/labeled_data.csv"
OUTPUT_PATH = "../../../data/robustness-study/preprocessed/"

# load data
data_loader = DataLoader(DATA_PATH)
data_loader.load_data()

# split data
df_train, df_val, df_test = data_loader.split(0.7, 0.15, 0.15)

# general preprocessing
preprocessor = GeneralTextPreprocessor()
# iterate preprocess over all rows


df_train['text'] = df_train['text'].apply(preprocessor.preprocess)
df_val['text'] = df_val['text'].apply(preprocessor.preprocess)
df_test['text'] = df_test['text'].apply(preprocessor.preprocess)

# save preprocessed data
df_train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"))
df_val.to_csv(os.path.join(OUTPUT_PATH, "val.csv"))
df_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"))


def main():
    pass


if __name__ == '__main__':
    main()
