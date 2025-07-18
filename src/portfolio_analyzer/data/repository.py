import logging
from typing import Dict, List, Optional

import pandas as pd

from .data_fetcher import DataFetcher


class Repository:
    """Repository acts as a middleware between the data source (DataFetcher) and the application.

    It caches all successfully fetched data in memory for the lifetime of the process.
    Only complete and valid data is cached; failed or incomplete requests are not cached.
    """

    def __init__(self, data_fetcher: DataFetcher, logger: Optional[logging.Logger] = None):
        self.data_fetcher = data_fetcher
        self.logger = logger or logging.getLogger(__name__)
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._market_cap_cache: Dict[str, pd.Series] = {}
        self._ticker_info_cache: Dict[str, Dict] = {}
        self._cashflow_cache: Dict[str, pd.DataFrame] = {}

    def fetch_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        cache_key = f"{','.join(sorted(tickers))}:{start_date}:{end_date}"
        if cache_key in self._price_cache:
            self.logger.info(f"Cache hit for price data: {cache_key}")
            return self._price_cache[cache_key]
        self.logger.info(f"Cache miss for price data: {cache_key}. Fetching from data source.")
        data = self.data_fetcher.fetch_price_data(tickers, start_date, end_date)
        if not data.empty and not data.isnull().all().all():
            self._price_cache[cache_key] = data
            return data
        self.logger.warning(f"Fetched price data is empty or contains only NaNs for: {cache_key}")
        return data

    def fetch_market_caps(self, tickers: List[str]) -> pd.Series:
        cache_key = ",".join(sorted(tickers))
        if cache_key in self._market_cap_cache:
            self.logger.info(f"Cache hit for market caps: {cache_key}")
            return self._market_cap_cache[cache_key]
        self.logger.info(f"Cache miss for market caps: {cache_key}. Fetching from data source.")
        data = self.data_fetcher.fetch_market_caps(tickers)
        if not data.empty and not data.isnull().all():
            self._market_cap_cache[cache_key] = data
            return data
        self.logger.warning(
            f"Fetched market cap data is empty or contains only NaNs for: {cache_key}"
        )
        return data

    def fetch_ticker_info(self, ticker: str) -> Dict:
        if ticker in self._ticker_info_cache:
            self.logger.info(f"Cache hit for ticker info: {ticker}")
            return self._ticker_info_cache[ticker]
        self.logger.info(f"Cache miss for ticker info: {ticker}. Fetching from data source.")
        data = self.data_fetcher.fetch_ticker_info(ticker)
        if data:
            self._ticker_info_cache[ticker] = data
            return data
        self.logger.warning(f"Fetched ticker info is empty for: {ticker}")
        return data

    def fetch_cashflow(self, ticker: str) -> Optional[pd.DataFrame]:
        if ticker in self._cashflow_cache:
            self.logger.info(f"Cache hit for cashflow: {ticker}")
            return self._cashflow_cache[ticker]
        self.logger.info(f"Cache miss for cashflow: {ticker}. Fetching from data source.")
        data = self.data_fetcher.fetch_cashflow(ticker)
        if data is not None and not data.empty and not data.isnull().all().all():
            self._cashflow_cache[ticker] = data
            return data
        self.logger.warning(f"Fetched cashflow data is empty or contains only NaNs for: {ticker}")
        return data
