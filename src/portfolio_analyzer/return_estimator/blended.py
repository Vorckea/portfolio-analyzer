import logging
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

    def __init__(
        self,
        weighted_estimators: Sequence[Tuple[ReturnEstimator, float]],
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        if not weighted_estimators:
            raise ValueError("At least one estimator must be provided.")
        if any(weight < 0 for _, weight in weighted_estimators):
            raise ValueError("Weights must be positive or zero.")
        if any(not isinstance(est, ReturnEstimator) for est, _ in weighted_estimators):
            raise TypeError("All estimators must be instances of ReturnEstimator.")

        total_weight = sum(weight for _, weight in weighted_estimators)
        if total_weight == 0:
            raise ValueError("Total weight for blended returns cannot be zero.")
        if abs(total_weight - 1.0) > 1e-10:
            weighted_estimators = [
                (est, weight / total_weight) for est, weight in weighted_estimators
            ]
            self.logger.warning("Total weight is not 1. Normalizing weights to sum to 1.")

        self.weighted_estimators = weighted_estimators
        self._returns = None

    def _calculate_returns(self) -> pd.Series:
        """Calculate the weighted blended returns from all estimators.

        Returns:
            pd.Series: Blended returns series.

        """
        return_list = [
            est.get_returns().mul(weight, fill_value=0) for est, weight in self.weighted_estimators
        ]
        blended = pd.concat(return_list, axis=1, join="outer").sum(axis=1)
        return blended

    def get_returns(self) -> pd.Series:
        if self._returns is None:
            self._returns = self._calculate_returns()
        return self._returns
