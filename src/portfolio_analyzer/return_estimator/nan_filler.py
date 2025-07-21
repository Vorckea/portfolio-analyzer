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
        base = self.returns.get_returns()
        replacement = self.replacement_returns.get_returns()
        filled = base.mask(base == 0).combine_first(replacement)
        return filled

    def get_returns(self) -> pd.Series:
        if self.filled_returns is None:
            self.filled_returns = self._fill_nan_returns()
        return self.filled_returns
