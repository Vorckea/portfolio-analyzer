from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelInputs:
    """Holds all the data required for the optimization and analysis steps."""

    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    log_returns: pd.DataFrame
    close_df: pd.DataFrame
    final_tickers: list[str]
    w_mkt: pd.Series
    hist_mean_returns: Optional[pd.Series] = None
    implied_equilibrium_returns: Optional[pd.Series] = None


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
