import os
import re
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


class RobustnessStudyDataLoader:
    """
    Holds the perturbed test set from the given directory.
    """
    def __init__(self, data_dir: str):
        self.data_dir: str = data_dir
        self.data = None

    def load_data(self) -> None:
        """
        Loads the perturbed test set from the given directory.
        """
        self.data = self._load_data()

    def _load_data(self):
        datasets = {}

        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                # extract noise levels from filename
                noise_levels = re.findall(r'_(psr|ppr|pri|prs|prd)(\d{3})', file)
                for typ, level in noise_levels:
                    level = int(level) / 100  # Convert level to a more readable format

                    if level > 0:
                        # read the CSV file
                        df = pd.read_csv(os.path.join(self.data_dir, file), sep='\t')

                        # check if the noise type is already in the dictionary
                        if typ not in datasets:
                            datasets[typ] = {}
                        # check if the noise level is already under this noise type
                        if level not in datasets[typ]:
                            datasets[typ][level] = []

                        # append the dataframe and file name to the list under the specific noise type and level
                        datasets[typ][level].append({'file': file, 'data': df})

        return datasets
