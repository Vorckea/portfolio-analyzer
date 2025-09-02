import numpy as np
import pytest

from portfolio_analyzer.core import objectives
from portfolio_analyzer.utils.exceptions import OptimizationError


def sample_inputs():
    mean_returns = np.array([0.01, 0.02, 0.015])
    cov_matrix = np.array(
        [
            [0.0001, 0.00005, 0.00002],
            [0.00005, 0.0002, 0.00003],
            [0.00002, 0.00003, 0.00015],
        ]
    )
    risk_free_rate = 0.01
    lambda_reg = 0.1
    weights = np.array([0.4, 0.4, 0.2])
    return mean_returns, cov_matrix, risk_free_rate, lambda_reg, weights


def test_negative_sharpe_ratio_returns_float():
    mean_returns, cov_matrix, risk_free_rate, lambda_reg, weights = sample_inputs()
    objective = objectives.make_negative_sharpe(
        mean_returns, cov_matrix, risk_free_rate, lambda_reg
    )
    result = objective(weights)
    assert isinstance(result, float)


def test_negative_sharpe_raises_on_zero_weights():
    mean_returns, cov_matrix, risk_free_rate, lambda_reg, _ = sample_inputs()
    zero_weights = np.array([0.0, 0.0, 0.0])
    objective = objectives.make_negative_sharpe(
        mean_returns, cov_matrix, risk_free_rate, lambda_reg
    )
    with pytest.raises(OptimizationError):
        objective(zero_weights)


def test_volatility_objective_non_negative():
    _, cov_matrix, _, _, weights = sample_inputs()
    objective = objectives.make_volatility_objective(cov_matrix)
    result = objective(weights)
    assert isinstance(result, float)
    assert result >= 0


def test_volatility_raises_on_zero_weights():
    _, cov_matrix, _, _, _ = sample_inputs()
    zero_weights = np.array([0.0, 0.0, 0.0])
    objective = objectives.make_volatility_objective(cov_matrix)
    with pytest.raises(OptimizationError):
        objective(zero_weights)
