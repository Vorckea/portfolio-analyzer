from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from ..data.schema import PriceHistory, SymbolInfo
from .models import PortfolioResult, PortfolioSpec, ReturnsFrame


class BaseReturnEstimator(ABC):
    @abstractmethod
    def fit(self, prices: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def estimate(self) -> ReturnsFrame:
        raise NotImplementedError


class BaseDataProvider(ABC):
    @abstractmethod
    def fetch_price_history(
        self,
        asset_list: list[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> PriceHistory:
        raise NotImplementedError

    @abstractmethod
    def fetch_symbol_info(self, symbol: str) -> SymbolInfo:
        raise NotImplementedError

    @abstractmethod
    def fetch_cashflow(self, symbol: str) -> pd.Series | None:
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
