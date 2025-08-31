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

        # find minimum volatility portfolio
        min_vol_result = self._find_minimum_volatility(bounds, initial_weights)
        # find maximum Sharpe ratio portfolio
        max_sharpe_result = self._find_max_sharpe(bounds, initial_weights)

        # sanity checks
        if not min_vol_result:
            raise OptimizationError("Could not find the minimum volatility portfolio.")
        if not max_sharpe_result or not max_sharpe_result.success:
            self.logger.error(
                f"Maximum Sharpe ratio optimization failed: {getattr(max_sharpe_result, 'message', None)}"
            )
            raise OptimizationError("Could not find the maximum Sharpe ratio portfolio.")

        min_return_log = min_vol_result.log_return
        max_return_log = max(max_sharpe_result.log_return, self.mean_returns.max())

        if max_return_log <= min_return_log:
            # degenerate case: frontier collapses to a single point
            self.logger.warning(
                "Max return <= min return (max=%s, min=%s). Returning single-point frontier.",
                max_return_log,
                min_return_log,
            )
            frontier_df = pd.DataFrame(
                [{"Return": min_return_log, "Volatility": min_vol_result.volatility}],
                columns=["Return", "Volatility"],
            )
            return frontier_df, max_sharpe_result, min_vol_result

        frontier_df = self._build_frontier_points(
            min_return=min_return_log,
            max_return=max_return_log,
            num_points=num_points,
            bounds=bounds,
            initial_weights=initial_weights,
        )

        if frontier_df.empty:
            raise OptimizationError(
                "Efficient frontier calculation failed: no valid frontier points found."
            )

        return frontier_df, max_sharpe_result, min_vol_result

    def _build_frontier_points(
        self,
        min_return: float,
        max_return: float,
        num_points: int,
        bounds,
        initial_weights,
    ) -> pd.DataFrame:
        target_log_returns = np.linspace(min_return, max_return, num_points)

        frontier_portfolios: list[dict] = []
        # objective uses covariance matrix values directly for the volatility minimization
        objective = VolatilityObjective(self.cov_matrix.values)

        for target in target_log_returns:
            constraints = (self._sum_to_one_constraint(), self._return_target_constraint(target))
            opt = self._optimize_portfolio(
                objective=objective,
                initial_weights=initial_weights,
                bounds=bounds,
                constraints=constraints,
            )

            if opt is not None and getattr(opt, "success", False):
                frontier_portfolios.append({"Return": target, "Volatility": opt.fun})
            else:
                # Log debug message for failed point (keeps frontier generation robust)
                self.logger.debug(
                    "Skipping failed frontier point for target %s: %s",
                    target,
                    getattr(opt, "message", None),
                )

        if not frontier_portfolios:
            self.logger.error("No successful frontier points were generated.")
            return pd.DataFrame(columns=["Return", "Volatility"])

        frontier_df = pd.DataFrame(frontier_portfolios)
        return frontier_df

    def _find_minimum_volatility(self, bounds, initial_weights) -> PortfolioResult:
        constraints = self._sum_to_one_constraint()
        opt = self._optimize_portfolio(
            objective=VolatilityObjective(self.cov_matrix.values),
            initial_weights=initial_weights,
            bounds=bounds,
            constraints=constraints,
        )
        if not opt.success:
            self.logger.error(f"Minimum volatility optimization failed: {opt.message}")
            return None
        return self._create_result_from_weights(opt.x)

    def _find_max_sharpe(self, bounds, initial_weights) -> PortfolioResult:
        constraints = self._sum_to_one_constraint()
        opt = self._optimize_portfolio(
            objective=NegativeSharpeRatio(
                self.mean_returns.values,
                self.cov_matrix.values,
                self.config.risk_free_rate,
                0.0,
            ),
            initial_weights=initial_weights,
            bounds=bounds,
            constraints=constraints,
        )
        return self._create_result_from_weights(opt.x) if opt.success else opt

    def _sum_to_one_constraint(self):
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def _return_target_constraint(self, target_return: float):
        # bind target_return at creation time to avoid late-binding closure issues
        def _fun(w, tr=target_return):
            return portfolio_return(w, self.mean_returns.values) - tr

        return {"type": "eq", "fun": _fun}

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
