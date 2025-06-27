from abc import ABC, abstractmethod
import pandas as pd


class ReturnEstimator(ABC):
    @abstractmethod
    def get_returns(self) -> pd.Series:
        pass
