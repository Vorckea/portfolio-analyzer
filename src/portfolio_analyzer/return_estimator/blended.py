import pandas as pd

from .base import ReturnEstimator


class BlendedReturn(ReturnEstimator):
    def __init__(self, return_estimators: list[tuple[ReturnEstimator, float]]):
        self.return_estimators = return_estimators
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> pd.Series:
        all_tickers = set()
        for est, _ in self.return_estimators:
            all_tickers.update(est.get_returns().index)
        total_weight = sum(weight for _, weight in self.return_estimators)
        blended = sum(est.get_returns() * weight for est, weight in self.return_estimators)
        blended = blended.reindex(all_tickers).fillna(0)
        if total_weight != 0:
            blended = blended / total_weight
        return blended

    def get_returns(self) -> pd.Series:
        return self.returns
