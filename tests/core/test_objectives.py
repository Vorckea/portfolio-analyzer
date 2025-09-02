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


import unittest

import numpy as np

from portfolio_analyzer.core import objectives
from portfolio_analyzer.utils.exceptions import OptimizationError


class TestPortfolioObjectives(unittest.TestCase):
    def setUp(self):
        self.mean_returns = np.array([0.01, 0.02, 0.015])
        self.cov_matrix = np.array(
            [
                [0.0001, 0.00005, 0.00002],
                [0.00005, 0.0002, 0.00003],
                [0.00002, 0.00003, 0.00015],
            ]
        )
        self.risk_free_rate = 0.01
        self.lambda_reg = 0.1
        self.weights = np.array([0.4, 0.4, 0.2])

    def test_negative_sharpe_ratio(self):
        objective = objectives.make_negative_sharpe(
            self.mean_returns, self.cov_matrix, self.risk_free_rate, self.lambda_reg
        )
        result = objective(self.weights)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1e10)

    def test_negative_sharpe_ratio_with_zero_weights(self):
        zero_weights = np.array([0.0, 0.0, 0.0])
        objective = objectives.make_negative_sharpe(
            self.mean_returns, self.cov_matrix, self.risk_free_rate, self.lambda_reg
        )
        with self.assertRaises(OptimizationError):
            objective(zero_weights)

    def test_volatility_objective(self):
        objective = objectives.make_volatility_objective(self.cov_matrix)
        result = objective(self.weights)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_volatility_objective_with_zero_weights(self):
        zero_weights = np.array([0.0, 0.0, 0.0])
        objective = objectives.make_volatility_objective(self.cov_matrix)
        with self.assertRaises(OptimizationError):
            objective(zero_weights)


if __name__ == "__main__":
    unittest.main()
