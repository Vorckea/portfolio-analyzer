from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    success: bool
    opt_weights: pd.Series
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    log_return: float
    std_dev: float
    sharpe_ratio: float
    arithmetic_return: float
    display_sharpe: float

    @classmethod
    def failure(cls) -> "PortfolioResult":
        return cls(
            success=False,
            opt_weights=pd.Series(dtype=float),
            mean_returns=pd.Series(dtype=float),
            cov_matrix=pd.DataFrame(),
            log_return=0.0,
            std_dev=0.0,
            sharpe_ratio=0.0,
            arithmetic_return=0.0,
            display_sharpe=0.0,
        )


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

    stats: dict[str, float]
    final_values: np.ndarray
    simulation_paths: np.ndarray
    num_simulations: int
    time_horizon_years: float
    dist_model_name: str
    initial_value: float
