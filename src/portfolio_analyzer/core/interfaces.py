from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from .models import PortfolioResult, PortfolioSpec


class BaseReturnEstimator(ABC):
    @abstractmethod
    def fit(self, prices: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def estimate(self) -> tuple[pd.Series, pd.DataFrame]:
        raise NotImplementedError


class BaseDataSource(ABC):
    @abstractmethod
    def fetch_price_history(
        self,
        asset_list: list[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_symbol_info(self, symbol: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def fetch_cashflow(self, symbol: str, frequency: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def available_assets(self) -> list[str]:
        raise NotImplementedError


class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        spec: PortfolioSpec,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> PortfolioResult:
        raise NotImplementedError
