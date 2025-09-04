import numpy as np
import pandas as pd

from ..config.config import AppConfig
from .base import ReturnEstimator


class ConstantReturn(ReturnEstimator):
    """A ReturnEstimator that returns a constant return value."""

    def __init__(self, constant_return: float, config: AppConfig):
        self.config = config
        self.constant_return = constant_return

    def get_returns(self) -> pd.Series:
        """Returns a Series with the constant return value."""
        if self.config.tickers is None or len(self.config.tickers) == 0:
            raise ValueError("No tickers provided in the configuration.")

        return pd.Series(
            data=np.full(
                shape=len(self.config.tickers),
                fill_value=self.constant_return,
            ),
            index=self.config.tickers,
            name="Constant Return",
        )
