from joblib import Parallel, delayed
import pandas as pd


def parallel_apply(df: pd.DataFrame, func, n_jobs=-1, **kwargs):
    """
    Apply a function to each row in a DataFrame in parallel.
    :param df: The DataFrame to apply the function to.
    :param func: The function to apply.
    :param n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    :param kwargs: Additional keyword arguments to pass to the function.
    """
    return Parallel(n_jobs=n_jobs)(delayed(func)(row, **kwargs) for row in df)
