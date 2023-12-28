import logging
from typing import Optional


def setup_logging(log_file_path: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    if log_file_path is not None:
        f_handler = logging.FileHandler(log_file_path)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
