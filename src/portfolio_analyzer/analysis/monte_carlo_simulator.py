from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t

from portfolio_analyzer.analysis.metrics import conditional_value_at_risk, value_at_risk
from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.data.models import PortfolioResult, SimulationResult


class SimulationEngine(ABC):
    @abstractmethod
    def generate_paths(
        self,
        mean_return_arr: np.ndarray,
        cov_matrix_arr: np.ndarray,
        opt_weights: np.ndarray,
        num_simulations: int,
        num_days: int,
    ) -> np.ndarray:
        pass


class NormalSimulationEngine(SimulationEngine):
    def generate_paths(
        self,
        mean_return_arr: np.ndarray,
        cov_matrix_arr: np.ndarray,
        opt_weights: np.ndarray,
        num_simulations: int,
        num_days: int,
    ) -> np.ndarray:
        rvs = multivariate_normal.rvs(
            mean=mean_return_arr,
            cov=cov_matrix_arr,
            size=(num_days, num_simulations),
        )
        portfolio_returns = rvs @ opt_weights
        compounded_returns = np.cumprod(1 + portfolio_returns, axis=0)
        return compounded_returns


class StudentTSimulationEngine(SimulationEngine):
    def __init__(self, df_t: int):
        self.df_t = df_t

    def generate_paths(
        self,
        mean_return_arr: np.ndarray,
        cov_matrix_arr: np.ndarray,
        opt_weights: np.ndarray,
        num_simulations: int,
        num_days: int,
    ) -> np.ndarray:
        scale_matrix = (self.df_t - 2) / self.df_t * cov_matrix_arr
        rvs = multivariate_t.rvs(
            loc=mean_return_arr,
            shape=scale_matrix,
            df=self.df_t,
            size=(num_days, num_simulations),
        )
        portfolio_returns = rvs @ opt_weights
        compounded_returns = np.cumprod(1 + portfolio_returns, axis=0)
        return compounded_returns


class SimulationStatisticsCalculator:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def calculate(self, sim_paths: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        final_values = pd.Series(sim_paths[-1, :], name="Final Value")
        stats = {
            "mean": final_values.mean(),
            "median": final_values.median(),
            "std_dev": final_values.std(),
            "var_95": value_at_risk(final_values, percentile=5.0),
            "cvar_95": conditional_value_at_risk(final_values, percentile=5.0),
            "prob_breakeven": (final_values > self.initial_value).mean(),
            "ci_5": np.percentile(final_values, 5),
            "ci_95": np.percentile(final_values, 95),
        }
        return stats, final_values.values


class MonteCarloSimulator:
    """Runs Monte Carlo simulations to project portfolio performance."""

    def __init__(self, config: AppConfig, stats_calculator: SimulationStatisticsCalculator):
        """Initialize the MonteCarloSimulator."""
        self.mc_config = config.monte_carlo
        self.trading_days = config.trading_days_per_year
        self.stats_calculator = stats_calculator

    def _get_engine(self, df_t_distribution: int) -> SimulationEngine:
        if df_t_distribution > 2:
            return StudentTSimulationEngine(df_t=df_t_distribution)
        return NormalSimulationEngine()

    def run(
        self,
        portfolio_result: PortfolioResult,
        num_simulations: int,
        time_horizon_years: float,
        df_t_distribution: int,
    ) -> SimulationResult:
        """Run the Monte Carlo simulation for a given optimized portfolio.

        Args:
            portfolio_result (PortfolioResult): The result of a portfolio optimization.
            num_simulations (int): The number of simulation paths to generate.
            time_horizon_years (float): The simulation period in years.
            df_t_distribution (int): Degrees of freedom for the Student's t-distribution.
                If <= 2, a Normal distribution is used instead.

        Returns:
            SimulationResult: An object containing the simulation results, including
                summary statistics and the generated paths.

        Raises:
            ValueError: If the portfolio_result is invalid or incomplete.

        """
        if (
            not portfolio_result.success
            or portfolio_result.opt_weights is None
            or portfolio_result.mean_returns is None
            or portfolio_result.cov_matrix is None
        ):
            raise ValueError("Portfolio result is invalid or incomplete.")

        num_days = int(time_horizon_years * self.trading_days)
        daily_mean_returns = portfolio_result.mean_returns / self.trading_days
        engine = self._get_engine(df_t_distribution)
        sim_paths = engine.generate_paths(
            mean_return_arr=daily_mean_returns.values,
            cov_matrix_arr=portfolio_result.cov_matrix.values / self.trading_days,
            opt_weights=portfolio_result.opt_weights.values,
            num_simulations=num_simulations,
            num_days=num_days,
        )
        sim_paths = self.mc_config.initial_value * sim_paths
        stats, final_values = self.stats_calculator.calculate(sim_paths)
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
