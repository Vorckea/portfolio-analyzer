import numpy as np


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate the expected return of a portfolio.

    Args:
        weights (np.ndarray): The weights of the assets in the portfolio.
        mean_returns (np.ndarray): The expected returns of the assets.

    Returns:
        float: The expected portfolio return.

    """
    return float(np.dot(weights, mean_returns))


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
    p_return = portfolio_return(weights=weights, mean_returns=mean_returns)
    p_vol = portfolio_volatility(weights=weights, cov_matrix=cov_matrix)
    risk_free_rate_log = np.log(1 + risk_free_rate)

    if p_vol == 0:
        return -np.inf if (p_return - risk_free_rate_log) < 0 else np.inf

    return (p_return - risk_free_rate_log) / p_vol
