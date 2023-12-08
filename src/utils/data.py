import pandas as pd
import os


class SimpleDataLoader:
    """
    Loads the data subsets from the given directory.
    """
    def __init__(self, dataset_dir: str):
        self.dataset: dict = {}
        self.dataset_dir: str = dataset_dir

    def load_data(self, subset: str) -> pd.DataFrame:
        subset_file = os.path.join(self.dataset_dir, f'{subset}.csv')
        df = pd.read_csv(subset_file, sep='\t', index_col=0)
        return df

    def load_dataset(self):
        self.dataset['train'] = self.load_data('train')
        self.dataset['val'] = self.load_data('val')
        self.dataset['test'] = self.load_data('test')

    def get_dataset(self) -> dict:
        return self.dataset


class RobustnessStudyDataLoader(SimpleDataLoader):
    """
    Takes as additional arguments the test set perturbation parameters.
    """
    def __init__(self, dataset_dir: str, noisy_dataset_dir: str, perturbation_params: dict):
        super().__init__(dataset_dir)
        self.noisy_dataset_dir: str = noisy_dataset_dir
        self.perturbation_params: dict = perturbation_params

    def load_perturbed_test_data(self) -> pd.DataFrame:
        """
        Loads the perturbed test set from the given directory.
        TODO: adapt to actual format of perturbation param dict
        """
        perturbed_test_file = os.path.join(self.noisy_dataset_dir, f'test_{self.perturbation_params["perturbation_type"]}_'
                                                             f'{self.perturbation_params["perturbation_level"]}.csv')
        df = pd.read_csv(perturbed_test_file, sep='\t', index_col=0)
        return df

    def load_dataset(self):
        super().load_dataset()
        self.dataset['test'] = self.load_perturbed_test_data()
