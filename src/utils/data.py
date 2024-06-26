import pandas as pd
import os
from dataclasses import dataclass


@dataclass
class Dataset:
    """
    Dataclass to hold the dataset.
    """
    train: pd.DataFrame = None
    val: pd.DataFrame = None
    test: pd.DataFrame = None


class SimpleDataLoader:
    """
    Loads the data subsets from the given directory.
    """
    def __init__(self, dataset_dir: str):
        self.dataset: Dataset = Dataset()
        self.dataset_dir: str = dataset_dir

    def load_data(self, subset: str) -> pd.DataFrame:
        subset_file = os.path.join(self.dataset_dir, f'{subset}.csv')
        df = pd.read_csv(subset_file, sep='\t', index_col=0)
        return df

    def load_dataset(self):
        self.dataset.train = self.load_data('train')
        self.dataset.val = self.load_data('val')
        self.dataset.test = self.load_data('test')

    def get_dataset(self) -> Dataset:
        return self.dataset
