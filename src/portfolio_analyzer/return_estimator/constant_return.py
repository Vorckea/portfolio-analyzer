import pandas as pd

from ..config.config import AppConfig
from .return_estimator import ReturnEstimator


class ConstantReturn(ReturnEstimator):
    """A ReturnEstimator that returns a constant return value."""

    def __init__(self, constant_return: float, config: AppConfig):
        self.config = config
        self.constant_return = constant_return

    def get_returns(self) -> pd.Series:
        """Returns a Series with the constant return value."""
        return pd.Series(
            [self.constant_return] * len(self.config.tickers),
            index=self.config.tickers,
            name="Constant Return",
        )
