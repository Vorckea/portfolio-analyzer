"""Portfolio optimization utilities and classes.

This module provides the PortfolioOptimizer class and supporting functions
for optimizing asset allocations using mean-variance analysis and regularization.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_analyzer.config import AppConfig


@dataclass
class PortfolioResult:
    """Holds the results of a portfolio optimization."""

    success: bool
    opt_weights: Optional[pd.DataFrame] = None
    mean_returns: Optional[pd.Series] = None
    cov_matrix: Optional[pd.DataFrame] = None
    log_return: float = 0.0
    std_dev: float = 0.0
    sharpe_ratio: float = 0.0
    arithmetic_return: float = 0.0
    display_sharpe: float = 0.0


def _standard_deviations(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)


def _expected_returns(weights: np.ndarray, annualized_mean_returns_vector: np.ndarray) -> float:
    return np.sum(annualized_mean_returns_vector * weights)


def _sharpe_ratio(
    weights: np.ndarray,
    annualized_mean_returns_vector: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    exp_return = _expected_returns(weights, annualized_mean_returns_vector)
    std_dev = _standard_deviations(weights, cov_matrix)

    risk_free_rate_log = np.log(1 + risk_free_rate)

    if std_dev == 0:
        return -np.inf if (exp_return - risk_free_rate) < 0 else np.inf
    return (exp_return - risk_free_rate_log) / std_dev


def _neg_sharpe_ratio_L2(
    weights: np.ndarray,
    annualized_mean_returns_vector: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    lambda_reg: float,
) -> float:
    s_ratio = _sharpe_ratio(weights, annualized_mean_returns_vector, cov_matrix, risk_free_rate)
    l2_penalty = lambda_reg * np.sum(weights**2)
    return -s_ratio + l2_penalty


class PortfolioOptimizer:
    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        config: AppConfig,
    ):
        if not mean_returns.index.equals(cov_matrix.index):
            common_tickers = sorted(list(set(mean_returns.index) & set(cov_matrix.index)))
            if not common_tickers:
                raise ValueError("Mean returns and covariance matrix have no common tickers.")

            self.mean_returns = mean_returns.loc[common_tickers]
            self.cov_matrix = cov_matrix.loc[common_tickers, common_tickers]
            self.tickers = common_tickers
            print(
                f"Warning: Input have mismatched tickers. Using {len(common_tickers)} common tickers."
            )
        else:
            self.mean_returns = mean_returns
            self.cov_matrix = cov_matrix
            self.tickers = list(mean_returns.index)

        self.config = config
        self.latest_result: Optional[PortfolioResult] = None

    def _create_result_from_weights(self, weights: np.ndarray) -> PortfolioResult:
        """Helper to create a PortfolioResult object from a given set of weights."""
        final_weights = pd.Series(weights, index=self.tickers)
        final_weights = final_weights[final_weights > 1e-5]  # Filter out negligible weights
        if final_weights.empty:
            return PortfolioResult(success=False)
        final_weights /= final_weights.sum()

        mean_returns_filtered = self.mean_returns.loc[final_weights.index]
        cov_matrix_filtered = self.cov_matrix.loc[final_weights.index, final_weights.index]

        log_return = _expected_returns(final_weights.values, mean_returns_filtered.values)
        std_dev = _standard_deviations(final_weights.values, cov_matrix_filtered.values)
        sharpe_ratio = _sharpe_ratio(
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
            sharpe_ratio=sharpe_ratio,
            arithmetic_return=arithmetic_return,
            display_sharpe=display_sharpe,
        )

    def optimize(self, lambda_reg: float) -> Optional[PortfolioResult]:
        """Performs the core portfolio optimization and stores the result."""
        self.latest_result = self._perform_core_optimization(lambda_val=lambda_reg)
        return self.latest_result

    def _perform_core_optimization(self, lambda_val: float) -> PortfolioResult:
        num_asset = len(self.tickers)
        if num_asset == 0:
            return PortfolioResult(success=False)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, self.config.optimization.max_weight_per_asset) for _ in range(num_asset))
        initial_weights = np.array([1.0 / num_asset] * num_asset)

        opt_result = minimize(
            _neg_sharpe_ratio_L2,
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
            return PortfolioResult(success=False)

        # Create a Series to easily filter both weights and tickers
        final_weights = pd.Series(opt_result.x, index=self.tickers)
        final_weights = final_weights[final_weights > self.config.optimization.min_weight_per_asset]

        if final_weights.empty:
            return PortfolioResult(success=False)

        # Normalize the filtered weights to ensure they sum to 1
        final_weights /= final_weights.sum()

        # Filter the model inputs to match the final assets
        mean_returns_filtered = self.mean_returns.loc[final_weights.index]
        cov_matrix_filtered = self.cov_matrix.loc[final_weights.index, final_weights.index]

        # Recalculate final portfolio metrics
        portfolio_return_log = _expected_returns(final_weights.values, mean_returns_filtered.values)
        portfolio_std_dev = _standard_deviations(final_weights.values, cov_matrix_filtered.values)
        portfolio_sharpe_log = _sharpe_ratio(
            final_weights.values,
            mean_returns_filtered.values,
            cov_matrix_filtered.values,
            self.config.risk_free_rate,
        )
        arithmetic_opt_return = np.exp(portfolio_return_log) - 1
        display_sharpe = (
            (arithmetic_opt_return - self.config.risk_free_rate) / portfolio_std_dev
            if portfolio_std_dev != 0
            else 0
        )

        return PortfolioResult(
            success=True,
            opt_weights=final_weights,
            mean_returns=mean_returns_filtered,
            cov_matrix=cov_matrix_filtered,
            log_return=portfolio_return_log,
            std_dev=portfolio_std_dev,
            sharpe_ratio=portfolio_sharpe_log,
            arithmetic_return=arithmetic_opt_return,
            display_sharpe=display_sharpe,
        )

    def calculate_efficient_frontier(
        self, num_points: int = 100
    ) -> tuple[pd.DataFrame, PortfolioResult, PortfolioResult]:
        """Calculates the efficient frontier.

        Returns a tuple containing:
        - A DataFrame with frontier points (Return, Volatility, Sharpe).
        - The PortfolioResult for the maximum Sharpe ratio portfolio.
        - The PortfolioResult for the minimum volatility portfolio.
        """
        num_asset = len(self.tickers)
        bounds = tuple((0, 1.0) for _ in range(num_asset))
        initial_weights = np.array([1.0 / num_asset] * num_asset)

        # 1. Find Min Volatility portfolio
        min_vol_constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        min_vol_opt = minimize(
            _standard_deviations,
            initial_weights,
            args=(self.cov_matrix.values,),
            method="SLSQP",
            bounds=bounds,
            constraints=min_vol_constraints,
        )
        if not min_vol_opt.success:
            raise RuntimeError("Could not find the minimum volatility portfolio.")
        min_vol_result = self._create_result_from_weights(min_vol_opt.x)

        # 2. Determine range of returns for the frontier (using log returns for optimization)
        min_return_log = min_vol_result.log_return
        max_return_log = self.mean_returns.max()
        target_log_returns = np.linspace(min_return_log, max_return_log, num_points)

        # 3. Calculate frontier points by minimizing volatility for each target return
        frontier_portfolios = []
        for target in target_log_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {
                    "type": "eq",
                    "fun": lambda w: _expected_returns(w, self.mean_returns.values) - target,
                },
            ]
            opt_result = minimize(
                _standard_deviations,
                initial_weights,
                args=(self.cov_matrix.values,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if opt_result.success:
                volatility = opt_result.fun
                arithmetic_return = np.exp(target) - 1
                frontier_portfolios.append(
                    {
                        "Return": arithmetic_return,
                        "Volatility": volatility,
                        "weights": opt_result.x,
                    }
                )

        if not frontier_portfolios:
            raise RuntimeError("Could not calculate any frontier points.")

        frontier_df = pd.DataFrame(frontier_portfolios)

        # 4. Calculate Sharpe for each point and find the max
        frontier_df["Sharpe"] = (frontier_df["Return"] - self.config.risk_free_rate) / frontier_df[
            "Volatility"
        ]
        max_sharpe_idx = frontier_df["Sharpe"].idxmax()
        max_sharpe_weights = frontier_df.loc[max_sharpe_idx, "weights"]
        max_sharpe_result = self._create_result_from_weights(max_sharpe_weights)

        return frontier_df.drop(columns="weights"), max_sharpe_result, min_vol_result
