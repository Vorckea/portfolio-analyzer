from abc import ABC, abstractmethod

import pandas as pd


class ReturnEstimator(ABC):
    @abstractmethod
    def get_returns(self) -> pd.Series:
        """Return a pandas Series of expected returns indexed by ticker.

        Returns:
            pd.Series: Series of expected returns.

        """
        pass
