"""Portfolio optimization utilities and classes.

This module provides the PortfolioOptimizer class and supporting functions
for optimizing asset allocations using mean-variance analysis and regularization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config.config import AppConfig
from ..core.utils import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)
from ..data.models import PortfolioResult
from ..utils.exceptions import InputAlignmentError, OptimizationError
from .objectives import PortfolioObjective


class PortfolioOptimizer:
    """Performs mean-variance portfolio optimization."""

    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        config: AppConfig,
        objective: PortfolioObjective,
    ):
        """Initialize the PortfolioOptimizer.

        Aligns tickers between mean_returns and cov_matrix to ensure consistency.

        Args:
            mean_returns (pd.Series): A series of mean expected returns for each asset.
            cov_matrix (pd.DataFrame): The covariance matrix of asset returns.

        Raises:
            InputAlignmentError: If the inputs have no common tickers.

        """
        common_tickers = mean_returns.index.intersection(cov_matrix.index)
        if common_tickers.empty:
            raise InputAlignmentError("Mean returns and covariance matrix have no common tickers.")

        if not mean_returns.index.equals(cov_matrix.index):
            mean_returns = mean_returns.loc[common_tickers]
            cov_matrix = cov_matrix.loc[common_tickers, common_tickers]

        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.tickers = list(mean_returns.index)
        self.config = config
        self.objective = objective

    def optimize(self) -> PortfolioResult | None:
        """Perform portfolio optimization to find the tangency portfolio.

        This method seeks to maximize the Sharpe ratio, potentially with L2
        regularization to control overfitting and encourage diversification.

        Args:
            lambda_reg (float): The L2 regularization coefficient. Higher values
                result in more diversified, less concentrated portfolios.

        Returns:
            Optional[PortfolioResult]: A data object containing the results of the
                optimization, or None if it fails.

        """
        return self._perform_core_optimization()

    def _perform_core_optimization(self) -> PortfolioResult:
        num_asset = len(self.tickers)
        if num_asset == 0:
            raise OptimizationError("No assets to optimize.")

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, self.config.optimization.max_weight_per_asset) for _ in range(num_asset))
        initial_weights = np.array([1.0 / num_asset] * num_asset)

        opt_result = minimize(
            self.objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not opt_result.success:
            raise OptimizationError(f"Portfolio optimization failed: {opt_result.message}")

        return self._create_result_from_weights(opt_result.x)

    def _create_result_from_weights(self, weights: np.ndarray) -> PortfolioResult:
        final_weights = pd.Series(weights, index=self.tickers)
        final_weights = final_weights[final_weights > self.config.optimization.min_weight_per_asset]
        if final_weights.empty:
            return PortfolioResult.failure()

        final_weights /= final_weights.sum()
        mean_returns_filtered = self.mean_returns.loc[final_weights.index]
        cov_matrix_filtered = self.cov_matrix.loc[final_weights.index, final_weights.index]

        log_return = portfolio_return(final_weights.values, mean_returns_filtered.values)
        std_dev = portfolio_volatility(final_weights.values, cov_matrix_filtered.values)
        s_ratio = sharpe_ratio(
            final_weights.values,
            mean_returns_filtered.values,
            cov_matrix_filtered.values,
            self.config.risk_free_rate,
        )
        arithmetic_return = np.exp(log_return) - 1
        display_sharpe = (
            (arithmetic_return - self.config.risk_free_rate) / std_dev if std_dev != 0 else 0
        )

        return PortfolioResult(
            success=True,
            opt_weights=final_weights,
            mean_returns=mean_returns_filtered,
            cov_matrix=cov_matrix_filtered,
            log_return=log_return,
            std_dev=std_dev,
            sharpe_ratio=s_ratio,
            arithmetic_return=arithmetic_return,
            display_sharpe=display_sharpe,
        )
