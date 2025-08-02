from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.exceptions import OptimizationError


class PortfolioObjective(ABC):
    @abstractmethod
    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective function.

        Args:
            weights (np.ndarray): Portfolio weights.

        Returns:
            float: Objective function value for the given weights.

        """
        pass


class NegativeSharpeRatio(PortfolioObjective):
    def __init__(
        self,
        mean_returns: npt.NDArray[np.float64] | pd.Series,
        cov_matrix: npt.NDArray[np.float64] | pd.DataFrame,
        risk_free_rate: float,
        lambda_reg: float,
    ) -> None:
        self.mean_returns = np.asarray(mean_returns, dtype=np.float64)
        self.cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        self.risk_free_rate = risk_free_rate
        self.lambda_reg = lambda_reg

    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        if weights is None or np.sum(weights) == 0:
            raise OptimizationError("Weights cannot be None or sum to zero.")

        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
        log_risk_free_rate = np.log(1 + self.risk_free_rate)

        if portfolio_volatility == 0:
            return -np.inf if (portfolio_return - log_risk_free_rate) < 0 else np.inf

        sharpe_ratio = (portfolio_return - log_risk_free_rate) / portfolio_volatility
        penalty = self.lambda_reg * np.sum(np.abs(weights) ** 2)

        return -sharpe_ratio + penalty


class VolatilityObjective(PortfolioObjective):
    def __init__(self, cov_matrix: npt.NDArray[np.float64] | pd.DataFrame):
        self.cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        if weights is None or np.sum(weights) == 0:
            raise OptimizationError("Weights cannot be None or sum to zero.")

        return np.sqrt(weights.T @ self.cov_matrix @ weights).item()
