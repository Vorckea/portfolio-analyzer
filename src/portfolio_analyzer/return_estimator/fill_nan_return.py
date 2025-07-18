import pandas as pd

from .interface import ReturnEstimator


class FillNaNReturn(ReturnEstimator):
    def __init__(self, returns: ReturnEstimator, replacement_returns: ReturnEstimator):
        self.returns: ReturnEstimator = returns
        self.replacement_returns: ReturnEstimator = replacement_returns
        self.filled_returns: pd.Series = self._fill_zero_returns()

    def _fill_zero_returns(self) -> pd.Series:
        """Fill 0 returns with replacement returns.

        Returns:
            pd.Series: Returns with 0s replaced by replacement returns.

        """
        filled_returns = self.returns.get_returns().replace(0, pd.NA)
        filled_returns = filled_returns.fillna(self.replacement_returns.get_returns())
        return filled_returns

    def get_returns(self):
        return self.filled_returns
