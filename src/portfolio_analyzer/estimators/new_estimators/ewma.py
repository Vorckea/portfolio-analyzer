import pandas as pd
from sklearn.covariance import LedoitWolf

from ...core.interfaces import BaseReturnEstimator
from ...core.models import ReturnsFrame
from ...data.schema import PriceHistory


class EWMAReturnEstimator(BaseReturnEstimator):
    def __init__(self, span: int = 20):
        self._span = span
        self._lw = LedoitWolf()

        self._price_history = None

        self._ewma_returns = None
        self._covariance = None

    def fit(self, prices: PriceHistory) -> None:
        self._price_history = prices
        self._lw.fit(prices.log_returns.dropna(how="all").values)

    def estimate(self) -> ReturnsFrame:
        if not self._price_history:
            raise ValueError("Estimator has not been fitted with price data.")

        if self._ewma_returns is None:
            self._ewma_returns = (
                self._price_history.log_returns.ewm(span=self._span).mean().iloc[-1]
            )
        if self._covariance is None:
            self._covariance = pd.DataFrame(
                data=self._lw.covariance_,
                index=self._price_history.log_returns.columns,
                columns=self._price_history.log_returns.columns,
            )
        return ReturnsFrame(
            returns=self._ewma_returns,
            covariance=self._covariance,
            frequency=self._price_history.frequency,
        )
