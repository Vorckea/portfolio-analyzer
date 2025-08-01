from abc import ABC, abstractmethod

import numpy as np


class PortfolioObjective(ABC):
    @abstractmethod
    def __call__(self, weights: np.ndarray) -> float:
        """Evaluate the objective function.

        Args:
            weights (np.ndarray): Portfolio weights.

        Returns:
            float: Objective function value for the given weights.

        """
        pass


class NegativeSharpeRatio(PortfolioObjective):
    def __init__(self, mean_returns, cov_matrix, risk_free_rate, lambda_reg):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.lambda_reg = lambda_reg

    def __call__(self, weights: np.ndarray) -> float:
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
        log_risk_free_rate = np.log(1 + self.risk_free_rate)

        if portfolio_volatility == 0:
            return -np.inf if (portfolio_return - log_risk_free_rate) < 0 else np.inf

        sharpe_ratio = (portfolio_return - log_risk_free_rate) / portfolio_volatility
        penalty = self.lambda_reg * np.sum(np.abs(weights) ** 2)

        return -sharpe_ratio + penalty


class VolatilityObjective(PortfolioObjective):
    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix

    def __call__(self, weights: np.ndarray) -> float:
        return np.sqrt(weights.T @ self.cov_matrix @ weights)
