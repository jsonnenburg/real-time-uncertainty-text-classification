import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class NoiseParams:
    """
    Dataclass to hold the noise parameters for the perturbed test set.
    """
    psr_level: float = 0.0
    ppr_level: float = 0.0
    pri_level: float = 0.0
    prs_level: float = 0.0
    prd_level: float = 0.0

    # TODO: incorporate checks (only one > 0, all >= 0, all <= 1, sum <= 1)
    def __set__(self, instance, value):
        # if setting a value, set all others to 0
        if value > 0:
            self.psr_level = 0.0
            self.ppr_level = 0.0
            self.pri_level = 0.0
            self.prs_level = 0.0
            self.prd_level = 0.0


def generate_noisy_test_data_filepath(noise_params: NoiseParams) -> str:
    """
    Generates the filepath for the perturbed test set with the given noise parameters.
    :param noise_params:
    :return: string with filepath
    """
    psr_level = noise_params.psr_level
    ppr_level = noise_params.ppr_level
    pri_level = noise_params.pri_level
    prs_level = noise_params.prs_level
    prd_level = noise_params.prd_level
    return f'test_psr{int(psr_level * 100):03d}_ppr{int(ppr_level * 100):03d}_pri{int(pri_level * 100):03d}_prs{int(prs_level * 100):03d}_prd{int(prd_level * 100):03d}.csv'


class RobustnessStudyTestSetLoader():
    """
    Takes as additional arguments the test set perturbation parameters.
    """
    def __init__(self, noisy_dataset_dir: str, noise_params: NoiseParams):
        self.noisy_dataset_dir: str = noisy_dataset_dir
        self.dataset: dict = dict(params=noise_params, data=None)

    def load_perturbed_test_dataset(self) -> None:
        """
        Loads the perturbed test set from the given directory.
        """
        perturbed_test_file = os.path.join(self.noisy_dataset_dir, generate_noisy_test_data_filepath(self.dataset['params']))
        self.dataset['data'] = pd.read_csv(perturbed_test_file, sep='\t', index_col=0)

    def get_dataset(self) -> dict:
        """
        Returns the perturbed test dataset.

        :return: dict with perturbation parameters and perturbed test dataset, keys: 'params', 'data'
        """
        return self.dataset


# usage:
# import NoiseParams, set the noise parameters in each iteration
