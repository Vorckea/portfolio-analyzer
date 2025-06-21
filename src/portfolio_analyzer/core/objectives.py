"""Objective functions for portfolio optimization.

These functions are designed to be used with optimizers like `scipy.optimize.minimize`.
They typically take an array of weights and other parameters (returns, covariance)
and return a scalar value to be minimized.
"""

import numpy as np


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate the expected return of a portfolio.

    Args:
        weights (np.ndarray): The weights of the assets in the portfolio.
        mean_returns (np.ndarray): The expected returns of the assets.

    Returns:
        float: The expected portfolio return.

    """
    return np.sum(weights * mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate the volatility (standard deviation) of a portfolio.

    Args:
        weights (np.ndarray): The weights of the assets in the portfolio.
        cov_matrix (np.ndarray): The covariance matrix of the assets.

    Returns:
        float: The portfolio volatility.

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
        weights (np.ndarray): The weights of the assets in the portfolio.
        mean_returns (np.ndarray): The expected returns of the assets.
        cov_matrix (np.ndarray): The covariance matrix of the assets.
        risk_free_rate (float): The risk-free rate of return.

    Returns:
        float: The portfolio's Sharpe ratio.

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
    """Calculate the negative Sharpe ratio, suitable for minimization.

    Includes an L2 regularization term on weights to encourage diversification.

    Args:
        weights (np.ndarray): The weights of the assets in the portfolio.
        mean_returns (np.ndarray): The expected returns of the assets.
        cov_matrix (np.ndarray): The covariance matrix of the assets.
        risk_free_rate (float): The risk-free rate of return.
        lambda_reg (float): The L2 regularization penalty coefficient.

    Returns:
        float: The negative Sharpe ratio, including the regularization penalty.

    """
    # The negative of the Sharpe ratio is minimized
    # We add a regularization term to the volatility
    s_ratio = sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
    l2_penalty = lambda_reg * np.sum(weights**2)
    return -s_ratio + l2_penalty
