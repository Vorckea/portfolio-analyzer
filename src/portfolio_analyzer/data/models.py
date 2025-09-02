"""Pydantic data models for portfolio results and simulation outputs.

This module provides lightweight, validated containers for results returned
from optimizers and Monte Carlo simulators.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class PortfolioResult(BaseModel):
    """Result object for portfolio optimization.

    Fields accept pandas/numpy objects directly and simple defaults are
    provided for failure cases.
    """

    success: bool = False
    opt_weights: pd.Series = Field(default_factory=lambda: pd.Series(dtype=float))
    mean_returns: pd.Series = Field(default_factory=lambda: pd.Series(dtype=float))
    cov_matrix: pd.DataFrame = Field(default_factory=lambda: pd.DataFrame())
    log_return: float = 0.0
    std_dev: float = 0.0
    sharpe_ratio: float = 0.0
    arithmetic_return: float = 0.0
    display_sharpe: float = 0.0

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @classmethod
    def failure(cls) -> "PortfolioResult":
        """Return a failed optimization result."""
        return cls(success=False)


class SimulationResult(BaseModel):
    """Hold the results of a Monte Carlo simulation.

    final_values should be a 1-D array with length == num_simulations.
    simulation_paths should have shape (num_days, num_simulations) where the
    second axis corresponds to the individual simulation paths.
    """

    stats: dict[str, float]
    final_values: np.ndarray
    simulation_paths: np.ndarray
    num_simulations: int
    time_horizon_years: float
    dist_model_name: str
    initial_value: float

    model_config = {"frozen": True}

    # allow numpy arrays/pandas objects
    model_config.update({"arbitrary_types_allowed": True})

    @field_validator("num_simulations")
    def _num_sim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_simulations must be > 0")
        return v

    @model_validator(mode="after")
    def _validate_shapes(self) -> "SimulationResult":
        # final_values must be 1-D with length matching num_simulations
        if not isinstance(self.final_values, np.ndarray):
            raise TypeError("final_values must be a numpy.ndarray")
        if self.final_values.ndim != 1:
            raise ValueError("final_values must be a 1-D array")
        if len(self.final_values) != self.num_simulations:
            raise ValueError("final_values length must equal num_simulations")

        # simulation_paths must be at least 2-D with second dim == num_simulations
        if not isinstance(self.simulation_paths, np.ndarray):
            raise TypeError("simulation_paths must be a numpy.ndarray")
        if self.simulation_paths.ndim < 2:
            raise ValueError("simulation_paths must be at least 2-D (num_simulations, steps)")
        if self.simulation_paths.shape[1] != self.num_simulations:
            raise ValueError("simulation_paths second dimension must equal num_simulations")

        return self
