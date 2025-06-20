from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.portfolio_optimizer import PortfolioResult


@dataclass
class SimulationResult:
    """ "Holds the results of a Monte Carlo simulation."""

    stats: Dict[str, float]
    final_values: np.ndarray
    simulation_paths: np.ndarray
    num_simulations: int
    time_horizon_years: float
    dist_model_name: str
    initial_value: float


class MonteCarloSimulator:
    def __init__(self, config: AppConfig):
        self.mc_config = config.monte_carlo
        self.trading_days = config.trading_days_per_year

    def run(
        self,
        portfolio_result: PortfolioResult,
        num_simulations: int,
        time_horizon_years: float,
        df_t_distribution: int,
    ) -> SimulationResult:
        if (
            not portfolio_result.success
            or portfolio_result.opt_weights is None
            or portfolio_result.mean_returns is None
            or portfolio_result.cov_matrix is None
        ):
            raise ValueError("Portfolio result is invalid or incomplete.")

        num_days = int(time_horizon_years * self.trading_days)
        daily_mean_returns = portfolio_result.mean_returns / self.trading_days

        sim_paths = self._generate_simulation_paths(
            mean_return_arr=daily_mean_returns.values,
            cov_matrix_arr=portfolio_result.cov_matrix.values / self.trading_days,
            opt_weights=portfolio_result.opt_weights.values,
            num_simulations=num_simulations,
            num_days=num_days,
            df_t=df_t_distribution,
        )

        stats, final_values = self._calculate_statistics(sim_paths)
        dist_model_name = "Student's t" if df_t_distribution > 2 else "Normal"

        return SimulationResult(
            stats=stats,
            final_values=final_values,
            simulation_paths=sim_paths,
            num_simulations=num_simulations,
            time_horizon_years=time_horizon_years,
            dist_model_name=dist_model_name,
            initial_value=self.mc_config.initial_value,
        )

    def _generate_simulation_paths(
        self,
        mean_return_arr: np.ndarray,
        cov_matrix_arr: np.ndarray,
        opt_weights: np.ndarray,
        num_simulations: int,
        num_days: int,
        df_t: int,
    ) -> np.ndarray:
        if df_t > 2:  # Use Student's t-distribution
            # Scale covariance for t-distribution properties
            scale_matrix = (df_t - 2) / df_t * cov_matrix_arr
            rvs = multivariate_t.rvs(
                loc=mean_return_arr,
                shape=scale_matrix,
                df=df_t,
                size=(num_days, num_simulations),
            )
        else:  # Use Normal distribution
            rvs = multivariate_normal.rvs(
                mean=mean_return_arr,
                cov=cov_matrix_arr,
                size=(num_days, num_simulations),
            )

        portfolio_returns = rvs @ opt_weights
        compounded_returns = np.cumprod(1 + portfolio_returns, axis=0)
        return self.mc_config.initial_value * compounded_returns

    def _calculate_statistics(
        self, sim_paths: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        final_values = sim_paths[-1, :]
        stats = {
            "mean": np.mean(final_values),
            "median": np.median(final_values),
            "std_dev": np.std(final_values),
            "var_95": np.percentile(final_values, 5),
            "cvar_95": np.mean(
                final_values[final_values <= np.percentile(final_values, 5)]
            ),
            "prob_breakeven": np.mean(final_values > self.mc_config.initial_value),
        }
        return stats, final_values
