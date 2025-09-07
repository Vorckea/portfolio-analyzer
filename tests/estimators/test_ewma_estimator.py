import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from portfolio_analyzer.data.schema import PriceHistory
from portfolio_analyzer.estimators.new_estimators.ewma import EWMAReturnEstimator


def _make_price_df():
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.5, 103.0], "MSFT": [150.0, 151.2, 152.3, 153.0]},
        index=idx,
    )
    df.index.name = "Datetime"
    return df


def test_estimate_raises_before_fit():
    est = EWMAReturnEstimator(span=3)
    with pytest.raises(ValueError):
        est.estimate()


def test_fit_calls_ledoitwolf_and_estimate_returns_expected():
    prices = _make_price_df()
    ph = PriceHistory(
        prices=prices,
        start_date=prices.index[0].to_pydatetime(),
        end_date=prices.index[-1].to_pydatetime(),
        frequency="D",
    )

    span = 2
    est = EWMAReturnEstimator(span=span)

    # Replace the internal LedoitWolf with a stub to control covariance_ and record input
    class LWStub:
        def __init__(self):
            self.fitted_with = None
            # simple symmetric covariance
            self.covariance_ = np.array([[0.01, 0.002], [0.002, 0.02]])

        def fit(self, arr):
            # store a copy for inspection
            self.fitted_with = np.array(arr)

    lw = LWStub()
    est._lw = lw

    # Fit should call lw.fit with the log returns values (dropna applied in estimator)
    est.fit(ph)
    expected_values = ph.log_returns.dropna(how="all").values
    assert lw.fitted_with is not None
    np.testing.assert_allclose(lw.fitted_with, expected_values)

    # Estimate should compute EWMA returns and construct covariance DataFrame from lw.covariance_
    rf = est.estimate()
    expected_ewma = ph.log_returns.ewm(span=span).mean().iloc[-1]
    pdt.assert_series_equal(rf.returns.sort_index(), expected_ewma.sort_index())

    expected_cov = pd.DataFrame(
        data=lw.covariance_, index=ph.log_returns.columns, columns=ph.log_returns.columns
    )
    pdt.assert_frame_equal(rf.covariance, expected_cov)
    assert rf.frequency == ph.frequency


def test_estimate_is_cached_on_second_call():
    # Ensure repeated calls to estimate reuse computed values (no error, same objects)
    prices = _make_price_df()
    ph = PriceHistory(
        prices=prices,
        start_date=prices.index[0].to_pydatetime(),
        end_date=prices.index[-1].to_pydatetime(),
        frequency="D",
    )

    est = EWMAReturnEstimator(span=2)

    # stub lw so covariance is deterministic
    class LWStub:
        def __init__(self):
            self.covariance_ = np.eye(2) * 0.001

        def fit(self, arr):
            pass

    est._lw = LWStub()
    est.fit(ph)
    first = est.estimate()
    second = est.estimate()
    # returned frames should be equal
    pdt.assert_series_equal(first.returns, second.returns)
    pdt.assert_frame_equal(first.covariance, second.covariance)
