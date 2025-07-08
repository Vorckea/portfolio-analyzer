import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def calculate_log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns from closing prices.

    Args:
        close_df (pd.DataFrame): DataFrame containing closing prices of assets.

    Returns:
        pd.DataFrame: DataFrame containing daily log returns, with the same index as close_df.

    """
    return np.log(close_df / close_df.shift(1)).dropna()


def calculate_annualized_covariance(
    log_returns: pd.DataFrame, trading_days: int = 252
) -> pd.DataFrame:
    """Calculate the annualized covariance matrix using Ledoit-Wolf shrinkage.

    Args:
        log_returns (pd.DataFrame): DataFrame of daily log returns for assets.
        trading_days (int, optional): Number of trading days in a year for annualization.
            This is used to scale the covariance matrix to an annual basis. Defaults to 252.

    Returns:
        pd.DataFrame: Annualized covariance matrix of asset returns.

    """
    lw = LedoitWolf()
    lw.fit(log_returns)
    return pd.DataFrame(
        data=lw.covariance_ * trading_days,
        index=log_returns.columns,
        columns=log_returns.columns,
    )
