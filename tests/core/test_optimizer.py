import numpy as np
import pandas as pd

from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.core.optimizer import PortfolioOptimizer
from portfolio_analyzer.data.models import PortfolioResult


def make_simple_inputs():
    mean_returns = pd.Series([0.01, 0.02], index=["A", "B"])
    cov = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0002]], index=["A", "B"], columns=["A", "B"])
    config = AppConfig.default()
    return mean_returns, cov, config


def test_create_result_from_weights_all_filtered_out():
    mean_returns, cov, config = make_simple_inputs()
    weights = np.array([0.0, 0.0])
    result = PortfolioOptimizer.create_result_from_weights(mean_returns, cov, config, weights)
    assert isinstance(result, PortfolioResult)
    assert result.success is False


def test_create_result_from_weights_success():
    mean_returns, cov, config = make_simple_inputs()
    weights = np.array([0.6, 0.4])
    result = PortfolioOptimizer.create_result_from_weights(mean_returns, cov, config, weights)
    assert result.success is True
    assert set(result.opt_weights.index) <= set(mean_returns.index)
    assert result.std_dev >= 0
