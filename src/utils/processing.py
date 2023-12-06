from joblib import Parallel, delayed
import pandas as pd


def parallel_apply(df: pd.DataFrame, func, n_jobs=-1, **kwargs):
    return Parallel(n_jobs=n_jobs)(delayed(func)(row, **kwargs) for row in df)
