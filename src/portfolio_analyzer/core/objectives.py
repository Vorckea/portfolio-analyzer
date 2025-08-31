from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.exceptions import OptimizationError

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
) -> ObjectiveCallable:
    """Produce a weights-only callable (pure) capturing the data by value.

    The returned callable is safe to pass to optimizers and does not mutate
    the provided inputs.
    """
    mean_arr = np.asarray(mean_returns, dtype=np.float64)
    cov_arr = np.asarray(cov_matrix, dtype=np.float64)

    def _objective(weights: npt.NDArray[np.float64]) -> float:
        return negative_sharpe_ratio(weights, mean_arr, cov_arr, risk_free_rate, lambda_reg)

    return _objective


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
) -> ObjectiveCallable:
    """Create a weights-only volatility objective from a covariance matrix.

    Parameters
    ----------
    cov_matrix
        Covariance matrix (array-like or DataFrame).

    Returns
    -------
    ObjectiveCallable
        Callable that accepts a weights array and returns the portfolio volatility.
    """

    cov_arr = np.asarray(cov_matrix, dtype=np.float64)

    def _objective(weights: npt.NDArray[np.float64]) -> float:
        """Weights-only wrapper for `volatility_objective`."""

        return volatility_objective(weights, cov_arr)

    return _objective


# Backwards-compatible thin wrappers (keeps existing API that expects objects)


class PortfolioObjective:
    """Compatibility class: instances behave like callables but delegate to pure functions."""

    def __call__(
        self, weights: npt.NDArray[np.float64]
    ) -> float:  # pragma: no cover - abstract compatibility
        """Abstract call signature for compatibility with previous API."""

        raise NotImplementedError()


class NegativeSharpeRatio(PortfolioObjective):
    """Callable object implementing the negative Sharpe ratio objective.

    This is a thin wrapper around the pure `make_negative_sharpe` factory and
    exists for backwards compatibility with code that constructs objective
    instances.
    """

    def __init__(
        self,
        mean_returns: npt.NDArray[np.float64] | pd.Series,
        cov_matrix: npt.NDArray[np.float64] | pd.DataFrame,
        risk_free_rate: float,
        lambda_reg: float,
    ) -> None:
        """Create the callable by capturing copies of the inputs."""

        self._fn = make_negative_sharpe(mean_returns, cov_matrix, risk_free_rate, lambda_reg)

    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        """Evaluate the objective for the provided weights."""

        return self._fn(weights)


class VolatilityObjective(PortfolioObjective):
    """Callable object implementing the volatility-only objective (sqrt(w' C w))."""

    def __init__(self, cov_matrix: npt.NDArray[np.float64] | pd.DataFrame) -> None:
        """Capture the covariance matrix and produce a weights-only callable."""

        self._fn = make_volatility_objective(cov_matrix)

    def __call__(self, weights: npt.NDArray[np.float64]) -> float:
        """Evaluate the volatility objective for the provided weights."""

        return self._fn(weights)
