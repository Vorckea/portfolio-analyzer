import numpy as np
import pandas as pd


def calculate_performance_summary(
    value_series: pd.Series, risk_free_rate: float, periods_per_year: int
) -> dict:
    """Calculate key performance metrics for a given portfolio value series.

    Args:
        value_series (pd.Series): A time series of portfolio values.
        risk_free_rate (float): The annualized risk-free rate.
        periods_per_year (int): The number of trading periods in a year
            (e.g., 252 for daily data).

    Returns:
        dict: A dictionary of performance metrics, including annualized return,
            volatility, Sharpe ratio, and max drawdown.

    """
    if value_series.empty or len(value_series) < 2:
        return {}

    metrics = {}
    metrics["Final Value"] = value_series.iloc[-1]
    metrics["Total Return"] = (value_series.iloc[-1] / value_series.iloc[0]) - 1

    returns = value_series.pct_change().dropna()
    if returns.empty:
        return metrics

    annualized_return = returns.mean() * periods_per_year
    metrics["Annualized Return"] = annualized_return

    volatility = returns.std() * np.sqrt(periods_per_year)
    metrics["Annualized Volatility"] = volatility

    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0.0
    metrics["Sharpe Ratio"] = sharpe_ratio

    peak = value_series.expanding(min_periods=1).max()
    drawdown = (value_series - peak) / peak
    metrics["Max Drawdown"] = drawdown.min()

    return metrics


def calculate_relative_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float,
    strategy_annualized_return: float,
    benchmark_annualized_return: float,
) -> dict:
    """Calculate metrics that compare a strategy's performance to a benchmark.

    Args:
        strategy_returns (pd.Series): The periodic returns of the strategy.
        benchmark_returns (pd.Series): The periodic returns of the benchmark.
        risk_free_rate (float): The annualized risk-free rate.
        strategy_annualized_return (float): The pre-calculated annualized
            return of the strategy.
        benchmark_annualized_return (float): The pre-calculated annualized
            return of the benchmark.

    Returns:
        dict: A dictionary containing relative metrics like Beta and Alpha.

    """
    aligned_df = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned_df.columns = ["Strategy", "Benchmark"]

    if len(aligned_df) < 2 or aligned_df["Benchmark"].var() == 0:
        return {"Alpha": np.nan, "Beta": np.nan}

    cov_matrix = aligned_df.cov()
    beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]

    alpha = strategy_annualized_return - (
        risk_free_rate + beta * (benchmark_annualized_return - risk_free_rate)
    )

    return {"Alpha": alpha, "Beta": beta}
