import pandas as pd

from .base import ReturnEstimator


class FillNaNReturn(ReturnEstimator):
    def __init__(self, returns: ReturnEstimator, replacement_returns: ReturnEstimator):
        self.returns: ReturnEstimator = returns
        self.replacement_returns: ReturnEstimator = replacement_returns
        self.filled_returns: pd.Series | None = None

    def _fill_nan_returns(self) -> pd.Series:
        """Fill both NaN and 0 values in the returns with replacement returns.

        Returns:
            pd.Series: Returns with NaN and 0 values filled.

        """
        filled_returns = self.returns.get_returns().replace(0, pd.NA)
        filled_returns = filled_returns.fillna(
            self.replacement_returns.get_returns()
        ).infer_objects(copy=False)
        return filled_returns

    def get_returns(self) -> pd.Series:
        if self.filled_returns is None:
            self.filled_returns = self._fill_nan_returns()
        return self.filled_returns
