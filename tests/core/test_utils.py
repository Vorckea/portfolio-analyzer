import numpy as np

from portfolio_analyzer.core import utils


def test_portfolio_return_simple():
    weights = np.array([0.5, 0.5])
    mean_returns = np.array([0.1, 0.2])
    assert np.isclose(utils.portfolio_return(weights, mean_returns), 0.15)


def test_portfolio_volatility_zero_cov():
    weights = np.array([0.5, 0.5])
    cov = np.zeros((2, 2))
    assert utils.portfolio_volatility(weights, cov) == 0.0


def test_sharpe_ratio_infinite_on_zero_vol():
    weights = np.array([1.0, 0.0])
    mean_returns = np.array([0.05, 0.0])
    cov = np.zeros((2, 2))
    # when volatility is zero and return is greater than risk_free, should be +inf
    sr = utils.sharpe_ratio(weights, mean_returns, cov, risk_free_rate=0.0)
    assert np.isposinf(sr)
