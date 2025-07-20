import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def download(self, tickers: List[str], start: str, end: str, **kwargs) -> pd.DataFrame:
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

    def fetch_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for the given tickers within the specified date range.

        Args:
            tickers (List[str]): List of ticker symbols to fetch data for.
            start_date (str): Start date for fetching historical data in 'YYYY-MM-DD' format.
            end_date (str): End date for fetching historical data in 'YYYY-MM-DD' format.

        Raises:
            ValueError: If no price data is fetched for any of the tickers.

        Returns:
            pd.DataFrame: DataFrame containing the historical price data with tickers as columns and
            dates as index.

        """
        if tickers is None or not tickers:
            self._log_error("No tickers provided for fetching price data.")
            raise ValueError("No tickers provided for fetching price data.")

        self._log_info(f"Fetching historical price data for {len(tickers)} tickers...")
        data: pd.DataFrame = self.provider.download(
            tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
        )
        if data.empty:
            self._log_error(
                "No price data fetched for any tickers. Provider returned an empty DataFrame."
            )
            raise ValueError("No price data fetched for any tickers.")

        close_df = self._extract_close_prices(data).dropna(axis=1, how="all")
        self._warn_missing_tickers(tickers, close_df.columns)
        return close_df.ffill()

    def fetch_market_caps(self, tickers: List[str]) -> pd.Series:
        """Fetch market capitalization data for the given tickers.

        Args:
            tickers (List[str]): List of ticker symbols to fetch market cap for.

        Returns:
            pd.Series: Series containing market cap values indexed by ticker symbols.

        """
        self._log_info(f"Fetching market cap data for {len(tickers)} tickers...")
        market_caps = {}
        for ticker in tickers:
            try:
                info = getattr(self.provider.Ticker(ticker), "info", {})
                m_cap = info.get("marketCap")
                market_caps[ticker] = m_cap if m_cap is not None else 0
                if m_cap is None:
                    self._log_warning(f"Market cap not available for {ticker}. Defaulting to 0.")
            except Exception as e:
                self._log_error(
                    f"Failed to fetch market cap for {ticker} due to an error: {e}. Defaulting to 0."
                )
                market_caps[ticker] = 0
        return pd.Series(market_caps)

    def fetch_ticker_info(self, ticker: str) -> Dict:
        """Fetch detailed information for a specific ticker symbol.

        Args:
            ticker (str): Ticker symbol to fetch information for.

        Returns:
            Dict: Dictionary containing detailed information about the ticker, such as market cap,
            sector, industry, and other relevant data.

        """
        try:
            return getattr(self.provider.Ticker(ticker), "info", {})
        except Exception as e:
            self._log_error(f"Failed to fetch info for {ticker}: {e}")
            return {}

    def fetch_cashflow(self, ticker: str):
        """Fetch the cash flow statement for a specific ticker symbol.

        Args:
            ticker (str): Ticker symbol to fetch cash flow data for.

        Returns:
            pd.DataFrame or None: Cash flow statement as a DataFrame or None if not available.

        """
        try:
            return getattr(self.provider.Ticker(ticker), "cashflow", None)
        except Exception as e:
            self._log_error(f"Failed to fetch cashflow for {ticker}: {e}")
            return None

    def _extract_close_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"]
        elif "Close" in data.columns:
            return data[["Close"]]
        raise ValueError("No 'Close' column found in price data.")

    def _warn_missing_tickers(self, requested: List[str], received) -> None:
        missing = set(requested) - set(received)
        if missing:
            self._log_warning(f"Failed to fetch price data for: {', '.join(missing)}")

    def _log_info(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _log_warning(self, msg: str) -> None:
        if self.logger:
            self.logger.warning(msg)

    def _log_error(self, msg: str) -> None:
        if self.logger:
            self.logger.error(msg)
