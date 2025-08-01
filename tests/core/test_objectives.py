import unittest

import numpy as np

from portfolio_analyzer.core import objectives


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
        objective = objectives.NegativeSharpeRatio(
            self.mean_returns, self.cov_matrix, self.risk_free_rate, self.lambda_reg
        )
        result = objective(self.weights)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1e10)

    def test_negative_sharpe_ratio_with_zero_weights(self):
        zero_weights = np.array([0.0, 0.0, 0.0])
        objective = objectives.NegativeSharpeRatio(
            self.mean_returns, self.cov_matrix, self.risk_free_rate, self.lambda_reg
        )
        result = objective(zero_weights)
        self.assertEqual(result, float("inf"))

    def test_volatility_objective(self):
        objective = objectives.VolatilityObjective(self.cov_matrix)
        result = objective(self.weights)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_volatility_objective_with_zero_weights(self):
        zero_weights = np.array([0.0, 0.0, 0.0])
        objective = objectives.VolatilityObjective(self.cov_matrix)
        result = objective(zero_weights)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
