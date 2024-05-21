import os

from src.preprocessing.robustness_study.shared_data_preprocessing import remove_stopwords
from src.utils.data import SimpleDataLoader
from src.utils.processing import parallel_apply


def main():
    input_path = "../../../data/robustness_study/preprocessed/"
    output_path = "../../../data/robustness_study/preprocessed_no_stopwords/"

    data_loader = SimpleDataLoader(input_path)
    data_loader.load_dataset()

    data = data_loader.get_dataset()

    # iterate stopword removal over all rows
    data.train['text'] = parallel_apply(data.train['text'], remove_stopwords)
    data.val['text'] = parallel_apply(data.val['text'], remove_stopwords)
    data.test['text'] = parallel_apply(data.test['text'], remove_stopwords)

    # save preprocessed data
    data.train.to_csv(os.path.join(output_path, "train.csv"), sep='\t')
    data.val.to_csv(os.path.join(output_path, "val.csv"), sep='\t')
    data.test.to_csv(os.path.join(output_path, "test.csv"), sep='\t')


if __name__ == '__main__':
    main()
