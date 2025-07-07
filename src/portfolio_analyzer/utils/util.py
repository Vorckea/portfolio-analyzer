import numpy as np
import pandas as pd


def calculate_log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns from closing prices.

    Args:
        close_df (pd.DataFrame): DataFrame containing closing prices of assets.

    Returns:
        pd.DataFrame: DataFrame containing daily log returns, with the same index as close_df.

    """
    return np.log(close_df / close_df.shift(1)).dropna()
