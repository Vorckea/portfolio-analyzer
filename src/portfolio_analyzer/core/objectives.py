"""Objective functions for portfolio optimization.

These functions are designed to be used with optimizers like `scipy.optimize.minimize`.
They typically take an array of weights and other parameters (returns, covariance)
and return a scalar value to be minimized.
"""

import numpy as np


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate the expected return of a portfolio given weights and mean returns.

    Args:
        weights (np.ndarray): Portfolio weights.
        mean_returns (np.ndarray): Mean returns of assets.

    Returns:
        float: Expected portfolio return.

    """
    return np.sum(weights * mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate the portfolio volatility given weights and covariance matrix.

    Args:
        weights (np.ndarray): Portfolio weights.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Portfolio volatility.

    """
    return np.sqrt(weights.T @ cov_matrix @ weights)


def sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Sharpe ratio of a portfolio.

    Args:
        weights (np.ndarray): Portfolio weights.
        mean_returns (np.ndarray): Mean returns of assets.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

    Returns:
        float: Sharpe ratio of the portfolio.

    """
    p_return = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    risk_free_rate_log = np.log(1 + risk_free_rate)

    if p_vol == 0:
        return -np.inf if (p_return - risk_free_rate_log) < 0 else np.inf

    return (p_return - risk_free_rate_log) / p_vol


def negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    lambda_reg: float = 0.0,
) -> float:
    """Calculate the negative Sharpe ratio with L2 regularization.

    Args:
        weights (np.ndarray): Portfolio weights.
        mean_returns (np.ndarray): Mean returns of assets.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
        lambda_reg (float): Regularization parameter for L2 penalty.

    Returns:
        float: Negative Sharpe ratio with L2 penalty.

    """
    s_ratio = sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    l2_penalty = lambda_reg * np.sum(weights**2)
    return -s_ratio + l2_penalty
