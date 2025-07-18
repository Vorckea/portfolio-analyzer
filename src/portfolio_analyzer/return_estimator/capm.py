import logging
from typing import List, Optional

import pandas as pd

from ..config.config import AppConfig
from ..data.repository import Repository
from ..utils.util import calculate_log_returns
from .base import ReturnEstimator

MARKET_TICKER = "OSEBX.OL"


class CAPM(ReturnEstimator):
    # ER_i = R_f + B_i * (ER_m - R_f)

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tickers: List[str],
        config: AppConfig,
        repository: Repository,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.repository = repository
        self.risk_free_rate = self.config.risk_free_rate
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self._returns = None

    def _calculate_capm_returns(self) -> pd.Series:
        market_ticker = MARKET_TICKER

        market_info = self.repository.fetch_ticker_info(market_ticker)
        er_market = market_info.get("expectedReturn")

        if er_market is None:
            price_data = self.repository.fetch_price_data(
                [market_ticker],
                self.start_date,
                self.end_date,
            )
            log_returns = calculate_log_returns(price_data)
            er_market = log_returns.mean().iloc[0] * self.config.trading_days_per_year

        returns = {}

        for ticker in self.tickers:
            info = self.repository.fetch_ticker_info(ticker)
            beta: Optional[float] = info.get("beta")
            if beta is None:
                returns[ticker] = 0
                continue
            returns[ticker] = self.risk_free_rate + beta * (er_market - self.risk_free_rate)
            self.logger.debug(
                f"Ticker: {ticker}, Beta: {beta:.2f}, Expected Return: {returns[ticker]:.4f}"
            )
        return pd.Series(returns)

    def get_returns(self) -> pd.Series:
        if self._returns is None:
            self._returns = self._calculate_capm_returns()
        return self._returns
