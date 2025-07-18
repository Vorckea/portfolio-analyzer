from typing import List, Tuple

import pandas as pd

from .base import ReturnEstimator


class BlendedReturn(ReturnEstimator):
    """Combines multiple ReturnEstimators into a single blended return using specified weights.

    Args:
        weighted_estimators (list[tuple[ReturnEstimator, float]]):
            List of (estimator, weight) pairs.

    """

    def __init__(self, weighted_estimators: List[Tuple[ReturnEstimator, float]]):
        if not weighted_estimators:
            raise ValueError("At least one estimator must be provided.")
        self.weighted_estimators = weighted_estimators
        self._returns = None

    def _calculate_returns(self) -> pd.Series:
        """Calculate the weighted blended returns from all estimators.

        Returns:
            pd.Series: Blended returns series.

        """
        all_tickers = set()
        for est, _ in self.weighted_estimators:
            all_tickers.update(est.get_returns().index)
        total_weight = sum(weight for _, weight in self.weighted_estimators)
        if total_weight == 0:
            raise ValueError("Total weight for blended returns cannot be zero.")
        blended = sum(est.get_returns() * weight for est, weight in self.weighted_estimators)
        blended = blended.reindex(all_tickers).fillna(0)
        return blended / total_weight

    def get_returns(self) -> pd.Series:
        if self._returns is None:
            self._returns = self._calculate_returns()
        return self._returns
