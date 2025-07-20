from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ModelInputs:
    """Holds all the data required for the optimization and analysis steps.

    Attributes:
        mean_returns (pd.Series): The final mean return vector for optimization.
        cov_matrix (pd.DataFrame): The final covariance matrix for optimization.
        log_returns (pd.DataFrame): The historical log returns of the assets.
        close_df (pd.DataFrame): The historical closing prices of the assets.
        final_tickers (List[str]): The list of tickers included in the final model.
        hist_mean_returns (pd.Series): Historical mean returns before any blending.
        implied_equilibrium_returns (Optional[pd.Series]): Market-implied returns
            from the Black-Litterman model.

    """

    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    log_returns: pd.DataFrame
    close_df: pd.DataFrame
    final_tickers: list[str]
    hist_mean_returns: pd.Series | None = None
    implied_equilibrium_returns: pd.Series | None = None


@dataclass
class PortfolioResult:
    """Holds the results of a portfolio optimization.

    Attributes:
        success (bool): Whether the optimization converged successfully.
        opt_weights (Optional[pd.Series]): The optimal asset weights.
        mean_returns (Optional[pd.Series]): The mean returns vector used.
        cov_matrix (Optional[pd.DataFrame]): The covariance matrix used.
        log_return (Optional[float]): The expected logarithmic return of the portfolio.
        std_dev (Optional[float]): The expected volatility of the portfolio.
        sharpe_ratio (Optional[float]): The expected Sharpe ratio of the portfolio.
        arithmetic_return (Optional[float]): The expected arithmetic return.
        display_sharpe (Optional[float]): The Sharpe ratio for display purposes.

    """

    success: bool
    opt_weights: pd.DataFrame | None = None
    mean_returns: pd.Series | None = None
    cov_matrix: pd.DataFrame | None = None
    log_return: float = 0.0
    std_dev: float = 0.0
    sharpe_ratio: float = 0.0
    arithmetic_return: float = 0.0
    display_sharpe: float = 0.0


@dataclass
class SimulationResult:
    """Hold the results of a Monte Carlo simulation.

    Attributes:
        stats (Dict[str, float]): A dictionary of summary statistics.
        final_values (np.ndarray): An array of the final portfolio values from
            each simulation path.
        simulation_paths (np.ndarray): The full simulation paths.
        num_simulations (int): The number of simulations run.
        time_horizon_years (float): The simulation time horizon in years.
        dist_model_name (str): The name of the distribution used ('Normal' or
            'Student's t').
        initial_value (float): The initial portfolio value for the simulation.

    """

    stats: Dict[str, float]
    final_values: np.ndarray
    simulation_paths: np.ndarray
    num_simulations: int
    time_horizon_years: float
    dist_model_name: str
    initial_value: float
