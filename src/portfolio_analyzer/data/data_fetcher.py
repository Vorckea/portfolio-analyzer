import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def download(self, tickers: list[str], start: str, end: str, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def Ticker(self, ticker: str) -> Any:
        pass


class DataFetcher:
    """DataFetcher class to handle fetching historical price data, market capitalizations, and ticker information."""  # noqa: E501

    def __init__(self, provider: MarketDataProvider, logger: logging.Logger | None = None):
        """Initialize the DataFetcher with a market data provider and an optional logger.

        Args:
            provider (MarketDataProvider): An instance of a market data provider that implements the
                MarketDataProvider interface. This provider is used to fetch historical price data,
                market capitalizations, and other ticker-related information.
            logger (Optional[logging.Logger], optional): An optional logger instance for logging
                information, warnings, and errors during data fetching operations. If not provided,
                a default logger will be created using `logging.getLogger(__name__)`.
                Defaults to None.

        """
        self.provider: MarketDataProvider = provider
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for the given tickers within the specified date range.

        Args:
            tickers (list[str]): List of ticker symbols to fetch data for.
            start_date (str): Start date for fetching historical data in 'YYYY-MM-DD' format.
            end_date (str): End date for fetching historical data in 'YYYY-MM-DD' format.

        Raises:
            ValueError: If no price data is fetched for any of the tickers.

        Returns:
            pd.DataFrame: DataFrame containing the historical price data with tickers as columns and
            dates as index.

        """
        if not tickers:
            self.logger.error("No tickers provided for fetching price data.")
            raise ValueError("No tickers provided for fetching price data.")

        self.logger.info("Fetching historical price data for %d tickers...", len(tickers))
        try:
            raw = self.provider.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            self.logger.exception(
                "Provider.download failed for tickers=%s start=%s end=%s",
                tickers,
                start_date,
                end_date,
            )
            raise

        if raw is None or raw.empty:
            self.logger.error(
                "No price data fetched for any tickers. Provider returned an empty DataFrame."
            )
            raise ValueError("No price data fetched for any tickers.")

        try:
            close = self._extract_close_prices(raw)
        except Exception:
            self.logger.exception("Failed to extract close prices from raw data.")
            raise

        # drop columns with all-NaN and warn about missing tickers
        close = close.dropna(axis=1, how="all")
        self._warn_missing_tickers(tickers, close.columns)

        # forward-fill to handle occasional missing values
        return close.ffill()

    def fetch_market_caps(self, tickers: list[str]) -> pd.Series:
        """Fetch market capitalization data for the given tickers.

        Args:
            tickers (list[str]): List of ticker symbols to fetch market cap for.

        Returns:
            pd.Series: Series containing market cap values indexed by ticker symbols.

        """
        if not tickers:
            self.logger.error("No tickers provided for fetching market caps.")
            raise ValueError("No tickers provided for fetching market caps.")

        self.logger.info("Fetching market cap data for %d tickers...", len(tickers))

        def _get_cap(ticker: str) -> float:
            try:
                t = self.provider.Ticker(ticker)
                info = getattr(t, "info", {}) or {}
                cap = info.get("marketCap")
                if cap is None:
                    self.logger.warning("Market cap not available for %s. Defaulting to 0.", ticker)
                    return 0.0
                return float(cap)
            except Exception:
                self.logger.exception("Failed to fetch market cap for %s. Defaulting to 0.", ticker)
                return 0.0

        caps = [_get_cap(ticker) for ticker in tickers]
        return pd.Series(caps, index=tickers, dtype=float)

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch detailed information for a specific ticker symbol.

        Args:
            ticker (str): Ticker symbol to fetch information for.

        Returns:
            Dict: Dictionary containing detailed information about the ticker, such as market cap,
            sector, industry, and other relevant data.

        """
        if not ticker:
            self.logger.error("fetch_ticker_info called with empty ticker.")
            raise ValueError("Ticker must be a non-empty string.")

        try:
            info = getattr(self.provider.Ticker(ticker), "info", {}) or {}
            return dict(info)
        except Exception:
            self.logger.exception("Failed to fetch info for %s", ticker)
            return {}

    def fetch_cashflow(self, ticker: str):
        """Fetch the cash flow statement for a specific ticker symbol.

        Args:
            ticker (str): Ticker symbol to fetch cash flow data for.

        Returns:
            pd.DataFrame or None: Cash flow statement as a DataFrame or None if not available.

        """
        if not ticker:
            self.logger.error("fetch_cashflow called with empty ticker.")
            raise ValueError("Ticker must be a non-empty string.")

        try:
            return getattr(self.provider.Ticker(ticker), "cashflow", None)
        except Exception:
            self.logger.exception("Failed to fetch cashflow for %s", ticker)
            return None

    def _extract_close_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                try:
                    return data["Close"]
                except KeyError:
                    # defensive: try cross-section
                    return data.xs("Close", axis=1, level=0)
            raise ValueError("No 'Close' level found in MultiIndex price data.")
        # Single-level
        if "Close" in data.columns:
            # return a DataFrame of closes (keeps consistent shape)
            return data[["Close"]]
        raise ValueError("No 'Close' column found in price data.")

    def _warn_missing_tickers(self, requested: Sequence[str], received) -> None:
        """Log tickers that were requested but not returned by the provider."""
        try:
            received_set = set(received)
        except Exception:
            # defensive fallback
            return
        missing = set(requested) - received_set
        if missing:
            self.logger.warning("Failed to fetch price data for: %s", ", ".join(sorted(missing)))
