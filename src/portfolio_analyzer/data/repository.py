import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from .data_fetcher import DataFetcher


class Repository:
    """Repository acts as a middleware between the data source (DataFetcher) and the application.

    It caches all successfully fetched data in memory for the lifetime of the process.
    Only complete and valid data is cached; failed or incomplete requests are not cached.
    """

    def __init__(self, data_fetcher: DataFetcher, logger: logging.Logger | None = None):
        self.data_fetcher = data_fetcher
        self.logger = logger or logging.getLogger(__name__)
        self._price_cache: defaultdict[tuple[str, ...], pd.DataFrame] = defaultdict(pd.DataFrame)
        self._market_cap_cache: dict[tuple[str, ...], pd.Series] = {}
        self._ticker_info_cache: dict[str, dict] = {}
        self._cashflow_cache: dict[str, pd.DataFrame] = {}

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self._validate_tickers(tickers)
        cache_key = (tuple(sorted(tickers)), start_date, end_date)
        if not self._price_cache[cache_key].empty:
            self.logger.debug(f"Cache hit for price data: {cache_key}")
            return self._price_cache[cache_key].copy()
        self.logger.debug(f"Cache miss for price data: {cache_key}. Fetching from data source.")
        data = self.data_fetcher.fetch_price_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        if not data.empty and data.notna().any().any():
            self._price_cache[cache_key] = data.copy()
            return data
        self.logger.warning(f"Fetched price data is empty or contains only NaNs for: {cache_key}")
        return data

    def fetch_market_caps(self, tickers: list[str]) -> pd.Series:
        self._validate_tickers(tickers)
        cache_key = tuple(sorted(tickers))
        if cache_key in self._market_cap_cache:
            self.logger.debug(f"Cache hit for market caps: {cache_key}")
            return self._market_cap_cache[cache_key].copy()
        self.logger.debug(f"Cache miss for market caps: {cache_key}. Fetching from data source.")
        data = self.data_fetcher.fetch_market_caps(tickers)
        if not data.empty and not data.isnull().all():
            self._market_cap_cache[cache_key] = data.copy()
            return data.copy()
        self.logger.warning(
            f"Fetched market cap data is empty or contains only NaNs for: {cache_key}"
        )
        return data

    def fetch_ticker_info(self, ticker: str) -> dict[str, Any]:
        self._validate_ticker(ticker)
        if ticker in self._ticker_info_cache:
            self.logger.debug(f"Cache hit for ticker info: {ticker}")
            return self._ticker_info_cache[ticker].copy()
        self.logger.debug(f"Cache miss for ticker info: {ticker}. Fetching from data source.")
        data = self.data_fetcher.fetch_ticker_info(ticker)
        if data:
            self._ticker_info_cache[ticker] = data.copy()
            return data.copy()
        self.logger.warning(f"Fetched ticker info is empty for: {ticker}")
        return data

    def fetch_cashflow(self, ticker: str) -> pd.DataFrame | None:
        self._validate_ticker(ticker)
        if ticker in self._cashflow_cache:
            self.logger.debug(f"Cache hit for cashflow: {ticker}")
            return self._cashflow_cache[ticker].copy()
        self.logger.debug(f"Cache miss for cashflow: {ticker}. Fetching from data source.")
        data = self.data_fetcher.fetch_cashflow(ticker)
        if data is not None and not data.empty and not data.isnull().all().all():
            self._cashflow_cache[ticker] = data.copy()
            return data.copy()
        self.logger.warning(f"Fetched cashflow data is empty or contains only NaNs for: {ticker}")
        return data

    def _validate_tickers(self, tickers: list[str]) -> None:
        """Validate that tickers are provided and not empty."""
        if not tickers or not all(tickers):
            self.logger.error("No valid tickers provided.")
            raise ValueError("No valid tickers provided.")

    def _validate_ticker(self, ticker: str) -> None:
        """Validate that a single ticker is provided and not empty."""
        if not ticker:
            self.logger.error("No valid ticker provided.")
            raise ValueError("No valid ticker provided.")
