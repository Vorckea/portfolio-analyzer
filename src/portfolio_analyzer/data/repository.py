import logging
from collections import defaultdict
from collections.abc import Callable
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
        self._price_cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._market_cap_cache: dict[tuple[str, ...], pd.Series] = {}
        self._ticker_info_cache: dict[str, dict] = {}
        self._cashflow_cache: dict[str, pd.DataFrame] = {}

    def _make_ticker_key(self, tickers: list[str]) -> tuple[str, ...]:
        return tuple(sorted(tickers))

    def _is_valid_data(self, data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return (
                not data.empty and not data.isnull().all().all()
                if isinstance(data, pd.DataFrame)
                else not data.isnull().all()
            )
        return False

    def _fetch_with_cahce(
        self,
        cache: dict,
        cache_key: Any,
        fetch_fn: Callable[..., Any],
        cache_name: str,
    ) -> Any:
        if cache_key in cache:
            self.logger.debug(f"Cache hit for {cache_name}: {cache_key}")
            val = cache[cache_key]
            try:
                return val.copy()
            except Exception:
                return val
        self.logger.debug(f"Cache miss for {cache_name}: {cache_key}. Fetching from data source.")
        data = fetch_fn()
        if self._is_valid_data(data):
            try:
                cache[cache_key] = data.copy()
            except Exception:
                cache[cache_key] = data
            return (
                cache[cache_key].copy() if hasattr(cache[cache_key], "copy") else cache[cache_key]
            )
        self.logger.warning(f"Fetched {cache_name} is empty or contains only NaNs for: {cache_key}")
        return data

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self._validate_tickers(tickers)
        cache_key = (self._make_ticker_key(tickers), start_date, end_date)
        return self._fetch_with_cahce(
            cache=self._price_cache,
            cache_key=cache_key,
            fetch_fn=lambda: self.data_fetcher.fetch_price_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
            ),
            cache_name="price data",
        )

    def fetch_market_caps(self, tickers: list[str]) -> pd.Series:
        self._validate_tickers(tickers)
        cache_key = self._make_ticker_key(tickers)
        return self._fetch_with_cahce(
            cache=self._market_cap_cache,
            cache_key=cache_key,
            fetch_fn=lambda: self.data_fetcher.fetch_market_caps(tickers),
            cache_name="market caps",
        )

    def fetch_ticker_info(self, ticker: str) -> dict[str, Any]:
        self._validate_ticker(ticker)
        if ticker in self._ticker_info_cache:
            self.logger.debug(f"Cache hit for ticker info: {ticker}")
            return self._ticker_info_cache[ticker].copy()
        self.logger.debug(f"Cache miss for ticker info: {ticker}. Fetching from data source.")
        info = self.data_fetcher.fetch_ticker_info(ticker)
        if info:
            # store shallow copy (dict)
            self._ticker_info_cache[ticker] = info.copy()
            return info.copy()
        self.logger.warning(f"Fetched ticker info is empty for: {ticker}")
        return info

    def fetch_cashflow(self, ticker: str) -> pd.DataFrame | None:
        self._validate_ticker(ticker)
        if (ticker in self._cashflow_cache) and self._is_valid_data(self._cashflow_cache[ticker]):
            self.logger.debug(f"Cache hit for cashflow: {ticker}")
            return self._cashflow_cache[ticker].copy()
        self.logger.debug(f"Cache miss for cashflow: {ticker}. Fetching from data source.")
        data = self.data_fetcher.fetch_cashflow(ticker)
        if self._is_valid_data(data):
            try:
                self._cashflow_cache[ticker] = data.copy()
            except Exception:
                self._cashflow_cache[ticker] = data
            return (
                self._cashflow_cache[ticker].copy()
                if hasattr(self._cashflow_cache[ticker], "copy")
                else self._cashflow_cache[ticker]
            )
        self.logger.warning(f"Fetched cashflow is empty for: {ticker}")
        return data

    def _validate_tickers(self, tickers: list[str]) -> None:
        """Validate that tickers are provided and not empty."""
        if not tickers or not all(tickers):
            self.logger.error("No valid tickers provided.")
            raise ValueError("No valid tickers provided.")

    def _validate_ticker(self, ticker: str) -> None:
        """Validate that a single ticker is provided and not empty."""
        self._validate_tickers([ticker])
