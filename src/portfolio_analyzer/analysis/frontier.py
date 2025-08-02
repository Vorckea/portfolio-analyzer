import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config.config import AppConfig
from ..core.objectives import NegativeSharpeRatio, VolatilityObjective
from ..core.utils import portfolio_return
from ..data.models import PortfolioResult
from ..utils.exceptions import OptimizationError

OPTIMIZATION_METHOD = "SLSQP"


class EfficientFrontierAnalyzer:
    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        config: AppConfig,
        logger: logging.Logger | None = None,
    ):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.tickers = mean_returns.index.tolist()
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def calculate(
        self, num_points: int = 100
    ) -> tuple[pd.DataFrame, PortfolioResult, PortfolioResult]:
        num_assets = len(self.tickers)
        bounds = tuple((0, 1.0) for _ in range(num_assets))
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        min_vol_constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        min_vol_opt = self._optimize_portfolio(
            objective=VolatilityObjective(self.cov_matrix.values),
            initial_weights=initial_weights,
            bounds=bounds,
            constraints=min_vol_constraints,
        )
        if not min_vol_opt.success:
            self.logger.error(f"Minimum volatility optimization failed: {min_vol_opt.message}")
            raise OptimizationError("Could not find the minimum volatility portfolio.")
        min_vol_result = self._create_result_from_weights(min_vol_opt.x)

        max_sharpe_opt = self._optimize_portfolio(
            objective=NegativeSharpeRatio(
                self.mean_returns.values,
                self.cov_matrix.values,
                self.config.risk_free_rate,
                0.0,
            ),
            initial_weights=initial_weights,
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        max_sharpe_result = self._create_result_from_weights(max_sharpe_opt.x)
        if not max_sharpe_result or not max_sharpe_result.success:
            self.logger.error(f"Maximum Sharpe ratio optimization failed: {max_sharpe_opt.message}")
            raise OptimizationError("Could not find the maximum Sharpe ratio portfolio.")

        min_return_log = min_vol_result.log_return
        max_return_log = max(max_sharpe_result.log_return, self.mean_returns.max())
        target_log_returns = np.linspace(min_return_log, max_return_log, num_points)

        frontier_portfolios: list[dict] = []
        for target_return in target_log_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {
                    "type": "eq",
                    "fun": lambda w: portfolio_return(w, self.mean_returns.values) - target_return,
                },
            ]

            opt = self._optimize_portfolio(
                objective=VolatilityObjective(self.cov_matrix.values),
                initial_weights=initial_weights,
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

    def _create_result_from_weights(self, weights: np.ndarray) -> PortfolioResult:
        from portfolio_analyzer.core.optimizer import PortfolioOptimizer

        return PortfolioOptimizer(
            self.mean_returns, self.cov_matrix, self.config, None
        )._create_result_from_weights(weights)

    @staticmethod
    def _optimize_portfolio(objective, initial_weights, bounds, constraints):
        return minimize(
            objective,
            initial_weights,
            method=OPTIMIZATION_METHOD,
            bounds=bounds,
            constraints=constraints,
        )
