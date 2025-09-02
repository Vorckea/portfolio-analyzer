"""Objective factories and adapters used by the optimizer.

This module exposes pure, functional objective factories (weights-only
callables) and a small adapter to convert those callables to the
`ObjectiveProtocol` used by the refactored optimizer.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.exceptions import OptimizationError
from .types import ObjectiveProtocol

# A pluggable objective is any callable that takes a weights array and returns a float.
ObjectiveCallable = Callable[[npt.NDArray[np.float64]], float]


def negative_sharpe_ratio(
    weights: npt.NDArray[np.float64],
    mean_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float,
    lambda_reg: float,
) -> float:
    """Pure negative Sharpe ratio objective with L2 penalty.

    This function is pure and does not mutate inputs. Use the `make_negative_sharpe`
    factory to produce a weights-only callable suitable for optimizers.
    """
    if weights is None or np.sum(weights) == 0:
        raise OptimizationError("Weights cannot be None or sum to zero.")

    mean_returns = np.asarray(mean_returns, dtype=np.float64)
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
    log_risk_free_rate = np.log(1 + risk_free_rate)

    if portfolio_volatility == 0:
        return -np.inf if (portfolio_return - log_risk_free_rate) < 0 else np.inf

    sharpe_ratio = (portfolio_return - log_risk_free_rate) / portfolio_volatility
    penalty = lambda_reg * np.sum(np.abs(weights) ** 2)

    return -sharpe_ratio + penalty


def make_negative_sharpe(
    mean_returns: npt.NDArray[np.float64] | pd.Series,
    cov_matrix: npt.NDArray[np.float64] | pd.DataFrame,
    risk_free_rate: float,
    lambda_reg: float,
) -> ObjectiveProtocol:
    """Produce a weights-only callable (pure) capturing the data by value.

    The returned callable is safe to pass to optimizers and does not mutate
    the provided inputs.
    """
    mean_arr = np.asarray(mean_returns, dtype=np.float64)
    cov_arr = np.asarray(cov_matrix, dtype=np.float64)

    def _objective(weights: npt.NDArray[np.float64]) -> float:
        return negative_sharpe_ratio(weights, mean_arr, cov_arr, risk_free_rate, lambda_reg)

    return NegativeSharpeObjective(mean_arr, cov_arr, risk_free_rate, lambda_reg)


def volatility_objective(
    weights: npt.NDArray[np.float64], cov_matrix: npt.NDArray[np.float64]
) -> float:
    """Pure portfolio volatility objective (sqrt(w' C w))."""
    if weights is None or np.sum(weights) == 0:
        raise OptimizationError("Weights cannot be None or sum to zero.")

    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
    return float(np.sqrt(weights.T @ cov_matrix @ weights))


def make_volatility_objective(
    cov_matrix: npt.NDArray[np.float64] | pd.DataFrame,
) -> ObjectiveProtocol:
    """Create a weights-only volatility objective from a covariance matrix."""
    cov_arr = np.asarray(cov_matrix, dtype=np.float64)

    def _objective(weights: npt.NDArray[np.float64]) -> float:
        return volatility_objective(weights, cov_arr)

    return VolatilityObjective(cov_arr)


# Legacy wrapper classes removed in favor of ObjectiveProtocol adapter.
# If you need an object with state, implement ObjectiveProtocol directly.


# ---------------------------------------------------------------------------
# Concrete Objective implementations (new canonical API)
# ---------------------------------------------------------------------------


class NegativeSharpeObjective:
    """ObjectiveProtocol implementation for the negative Sharpe ratio.

    This object stores the data by value (numpy arrays) and exposes
    `to_callable()` and `__call__` for optimizer compatibility.
    """

    name = "negative_sharpe"

    def __init__(
        self,
        mean_returns: npt.NDArray[np.float64] | pd.Series,
        cov_matrix: npt.NDArray[np.float64] | pd.DataFrame,
        risk_free_rate: float,
        lambda_reg: float,
    ) -> None:
        self._mean = np.asarray(mean_returns, dtype=np.float64)
        self._cov = np.asarray(cov_matrix, dtype=np.float64)
        self._rfr = float(risk_free_rate)
        self._lambda = float(lambda_reg)

    def to_callable(self) -> ObjectiveCallable:
        def _objective(weights: npt.NDArray[np.float64]) -> float:
            return negative_sharpe_ratio(weights, self._mean, self._cov, self._rfr, self._lambda)

        return _objective

    def gradient(self) -> Callable[[np.ndarray], np.ndarray] | None:
        return None

    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        return self.to_callable()(weights)


class VolatilityObjective:
    """ObjectiveProtocol implementation for portfolio volatility (sqrt(w' C w))."""

    name = "volatility"

    def __init__(self, cov_matrix: npt.NDArray[np.float64] | pd.DataFrame) -> None:
        self._cov = np.asarray(cov_matrix, dtype=np.float64)

    def to_callable(self) -> ObjectiveCallable:
        def _objective(weights: npt.NDArray[np.float64]) -> float:
            return volatility_objective(weights, self._cov)

        return _objective

    def gradient(self) -> Callable[[np.ndarray], np.ndarray] | None:
        return None

    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        return self.to_callable()(weights)
