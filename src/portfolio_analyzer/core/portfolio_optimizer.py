"""Portfolio optimization utilities and classes.

This module provides the PortfolioOptimizer class and supporting functions
for optimizing asset allocations using mean-variance analysis and regularization.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.core.objectives import (
    negative_sharpe_ratio,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)
from portfolio_analyzer.data.models import PortfolioResult
from portfolio_analyzer.utils.exceptions import InputAlignmentError, OptimizationError


class PortfolioOptimizer:
    """Performs mean-variance portfolio optimization."""

    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, config: AppConfig):
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

    def optimize(self, lambda_reg: float) -> Optional[PortfolioResult]:
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
        return self._perform_core_optimization(lambda_reg)

    def _perform_core_optimization(self, lambda_val: float) -> PortfolioResult:
        num_asset = len(self.tickers)
        if num_asset == 0:
            raise OptimizationError("No assets to optimize.")

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, self.config.optimization.max_weight_per_asset) for _ in range(num_asset))
        initial_weights = np.array([1.0 / num_asset] * num_asset)

        opt_result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(
                self.mean_returns.values,
                self.cov_matrix.values,
                self.config.risk_free_rate,
                lambda_val,
            ),
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
            return PortfolioResult(success=False)

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

    def calculate_efficient_frontier(
        self, num_points: int = 100
    ) -> tuple[pd.DataFrame, PortfolioResult, PortfolioResult]:
        """Calculate the efficient frontier for the given assets.

        The efficient frontier is the set of optimal portfolios that offer the
        highest expected return for a given level of risk (volatility).

        Args:
            num_points (int): The number of points to calculate along the frontier.

        Returns:
            tuple[pd.DataFrame, PortfolioResult, PortfolioResult]: A tuple containing:
                - A DataFrame with the 'Return' and 'Volatility' for each point
                  on the frontier.
                - The result for the Maximum Sharpe Ratio portfolio.
                - The result for the Minimum Volatility portfolio.

        Raises:
            OptimizationError: If the optimization fails for key points on the
                frontier (e.g., min volatility or max Sharpe).

        """
        num_asset = len(self.tickers)
        bounds = tuple((0, 1.0) for _ in range(num_asset))
        initial_weights = np.array([1.0 / num_asset] * num_asset)

        # 1. Find Min Volatility portfolio
        min_vol_constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        min_vol_opt = minimize(
            portfolio_volatility,
            initial_weights,
            args=(self.cov_matrix.values,),
            method="SLSQP",
            bounds=bounds,
            constraints=min_vol_constraints,
        )
        if not min_vol_opt.success:
            raise OptimizationError("Could not find the minimum volatility portfolio.")
        min_vol_result = self._create_result_from_weights(min_vol_opt.x)

        # 2. Find the Max Sharpe Ratio portfolio directly using the optimizer
        # We use lambda_reg=0 to find the theoretical max Sharpe on the frontier
        max_sharpe_result = self.optimize(lambda_reg=0.0)
        if not max_sharpe_result or not max_sharpe_result.success:
            raise OptimizationError("Could not find the maximum Sharpe ratio portfolio.")

        min_return_log = min_vol_result.log_return
        max_return_log = max(max_sharpe_result.log_return, self.mean_returns.max())
        target_log_returns = np.linspace(min_return_log, max_return_log, num_points)

        # 4. Calculate frontier points by minimizing volatility for each target return
        frontier_portfolios: List[dict] = []
        for target_return in target_log_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {
                    "type": "eq",
                    "fun": lambda w: portfolio_return(w, self.mean_returns.values) - target_return,
                },
            ]
            opt = minimize(
                portfolio_volatility,
                initial_weights,
                args=(self.cov_matrix.values,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if opt.success:
                frontier_portfolios.append({"Return": target_return, "Volatility": opt.fun})

        if not frontier_portfolios:
            raise OptimizationError("Could not calculate any points for the efficient frontier.")

        frontier_df = pd.DataFrame(frontier_portfolios)
        log_risk_free_rate = np.log(1 + self.config.risk_free_rate)
        frontier_df["Sharpe"] = (frontier_df["Return"] - log_risk_free_rate) / frontier_df[
            "Volatility"
        ]

        return frontier_df, max_sharpe_result, min_vol_result
