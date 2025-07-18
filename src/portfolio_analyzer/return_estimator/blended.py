from collections.abc import Sequence
from typing import Tuple

import pandas as pd

from .base import ReturnEstimator


class BlendedReturn(ReturnEstimator):
    """Combines multiple ReturnEstimators into a single blended return using specified weights.

    Args:
        weighted_estimators (list[tuple[ReturnEstimator, float]]):
            List of (estimator, weight) pairs.

    """

    def __init__(self, weighted_estimators: Sequence[Tuple[ReturnEstimator, float]]):
        if not weighted_estimators:
            raise ValueError("At least one estimator must be provided.")

        for _, weight in weighted_estimators:
            if weight < 0:
                raise ValueError("Weights must be positive or zero.")

        self.weighted_estimators = weighted_estimators
        self._returns = None

    def _calculate_returns(self) -> pd.Series:
        """Calculate the weighted blended returns from all estimators.

        Returns:
            pd.Series: Blended returns series.

        """
        returns_list = []
        weights = []
        for est, weight in self.weighted_estimators:
            returns = est.get_returns()
            returns_list.append(returns * weight)
            weights.append(weight)

        # Concatenate all weighted returns into a DataFrame, aligning on index
        blended_df = pd.concat(returns_list, axis=1).fillna(0)
        blended = blended_df.sum(axis=1)
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight for blended returns cannot be zero.")
        return blended / total_weight

    def get_returns(self) -> pd.Series:
        if self._returns is None:
            self._returns = self._calculate_returns()
        return self._returns
