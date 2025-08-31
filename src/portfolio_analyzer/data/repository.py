import copy
import logging
import threading
from collections.abc import Callable
from typing import Any, Hashable, TypeVar

import pandas as pd

from .data_fetcher import DataFetcher

T = TypeVar("T")


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
        self._locks = dict[Hashable, threading.Lock] = {}

    def _make_ticker_key(self, tickers: list[str]) -> tuple[str, ...]:
        return tuple(sorted(tickers))

    def _is_valid_data(self, data: Any) -> bool:
        """Return True if data is a non-empty DataFrame/Series and not all NaNs."""
        if data is None:
            return False

        if isinstance(data, pd.DataFrame):
            if data.empty:
                return False
            return not data.isnull().all().all()

        if isinstance(data, pd.Series):
            if data.empty:
                return False
            return not data.isnull().all()

        try:
            return bool(len(data))
        except Exception:
            return True

    def _safe_copy(self, obj: Any) -> Any:
        """Return a shallow copy if possible, otherwise return the original object."""
        try:
            return copy.copy(obj)
        except Exception:
            return obj

    def _get_cached(
        self, cache: dict, cache_key: Hashable, fetch_fn: Callable[..., T], cache_name: str
    ) -> T:
        # cache hit
        if cache_key in cache:
            self.logger.debug("Cache hit for %s (key=%s)", cache_name, cache_key)
            return self._safe_copy(cache[cache_key])

        # cache miss -> fetch
        self.logger.info("Cache miss for %s. Fetching (key=%s)...", cache_name, cache_key)

        lock = self._locks.setdefault(cache_key, threading.Lock())

        with lock:
            if cache_key in cache:
                self.logger.debug(
                    "Cache filled while waiting for lock for %s (key=%s)", cache_name, cache_key
                )
                return self._safe_copy(cache[cache_key])
            try:
                data = fetch_fn()
            except Exception as exc:
                self.logger.exception(
                    "Error fetching %s (key=%s): %s", cache_name, cache_key, exc, exc_info=True
                )
                raise

            if not self._is_valid_data(data):
                msg = f"Fetched {cache_name!r} is empty or invalid for key={cache_key!r}"
                self.logger.error(msg)
                raise RuntimeError(msg)

            cached = self._safe_copy(data)
            cache[cache_key] = cached
            self.logger.debug("Cached %s (key=%s)", cache_name, cache_key)
            return cached

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self._validate_tickers(tickers)
        cache_key = (self._make_ticker_key(tickers), start_date, end_date)
        return self._get_cached(
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
        return self._get_cached(
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
