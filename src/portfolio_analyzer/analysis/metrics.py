import numpy as np
import pandas as pd


def calculate_performance_summary(
    value_series: pd.Series, risk_free_rate: float, periods_per_year: int
) -> dict:
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
