import copy
import logging
import threading
from collections.abc import Callable
from typing import Any, Hashable, TypeVar

import pandas as pd

from .cache import CacheStore
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
        # in-memory caches
        self._ticker_info_cache: dict[str, dict] = {}

        # instantiate CacheStore helpers
        self._price_store = CacheStore(
            name="price data",
            logger=self.logger,
            validator=self._is_valid_data,
            copier=self._safe_copy,
        )
        self._market_cap_store = CacheStore(
            name="market caps",
            logger=self.logger,
            validator=self._is_valid_data,
            copier=self._safe_copy,
        )
        self._cashflow_store = CacheStore(
            name="cashflow",
            logger=self.logger,
            copier=copy.deepcopy,
        )

    def _make_ticker_key(self, tickers: list[str]) -> tuple[str, ...]:
        return tuple(sorted(tickers))

    def _is_valid_data(self, data: Any) -> bool:
        """Return True if data is a non-empty DataFrame/Series and not all NaNs."""
        if data is None:
            return False
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.empty:
                return False
            # not all NaN
            return (
                not data.isna().all().all()
                if isinstance(data, pd.DataFrame)
                else not data.isna().all()
            )
        # generic non-empty container check
        try:
            return bool(len(data))
        except Exception:
            return True

    def _safe_copy(self, obj: Any) -> Any:
        """Return a shallow copy if possible, otherwise return the original object."""
        try:
            return copy.copy(obj)
        except Exception:
            # fallback: return as-is if copy fails
            return obj

    def _get_cached(
        self, cache: dict, cache_key: Hashable, fetch_fn: Callable[..., T], cache_name: str
    ) -> T:
        """Deprecated compatibility shim kept for now; prefer using CacheStore.get_or_fetch."""
        # delegate to the appropriate CacheStore if possible
        if cache_name == "price data":
            return self._price_store.get_or_fetch(cache_key, fetch_fn)
        if cache_name == "market caps":
            return self._market_cap_store.get_or_fetch(cache_key, fetch_fn)

        # fallback to original behavior (kept for backward compatibility)
        if cache_key in cache:
            self.logger.debug("Cache hit for %s (key=%s)", cache_name, cache_key)
            return self._safe_copy(cache[cache_key])

        self.logger.info("Cache miss for %s. Fetching (key=%s)...", cache_name, cache_key)
        lock = threading.Lock()
        with lock:
            if cache_key in cache:
                return self._safe_copy(cache[cache_key])
            data = fetch_fn()
            if not self._is_valid_data(data):
                raise RuntimeError("Invalid data")
            cache[cache_key] = self._safe_copy(data)
            return self._safe_copy(cache[cache_key])

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self._validate_tickers(tickers)
        cache_key = (self._make_ticker_key(tickers), start_date, end_date)
        return self._price_store.get_or_fetch(
            cache_key,
            lambda: self.data_fetcher.fetch_price_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
            ),
        )

    def fetch_market_caps(self, tickers: list[str]) -> pd.Series:
        self._validate_tickers(tickers)
        cache_key = self._make_ticker_key(tickers)
        return self._market_cap_store.get_or_fetch(
            cache_key,
            lambda: self.data_fetcher.fetch_market_caps(tickers),
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
        cache_key = ticker
        return self._cashflow_store.get_or_fetch(
            cache_key,
            lambda: self.data_fetcher.fetch_cashflow(ticker),
        )

    def _validate_tickers(self, tickers: list[str]) -> None:
        """Validate that tickers are provided and not empty."""
        if not tickers or not all(tickers):
            self.logger.error("No valid tickers provided.")
            raise ValueError("No valid tickers provided.")

    def _validate_ticker(self, ticker: str) -> None:
        """Validate that a single ticker is provided and not empty."""
        self._validate_tickers([ticker])
