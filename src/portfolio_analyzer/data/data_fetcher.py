import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def download(self, tickers: List[str], start: str, end: str, **kwargs) -> pd.DataFrame: ...
    @abstractmethod
    def Ticker(self, ticker: str): ...


class DataFetcher:
    """DataFetcher class to handle fetching historical price data, market capitalizations, and ticker information."""  # noqa: E501

    def __init__(self, provider: MarketDataProvider, logger: Optional[logging.Logger] = None):
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
        self.logger.info("Fetching historical price date for %d tickers...", len(tickers))
        data: pd.DataFrame = self.provider.download(
            tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
        )
        if data.empty:
            self.logger.error(
                "No price data fetched for any tickers. Provider returned an empty DataFrame."
            )
            raise ValueError("No price data fetched for any tickers.")

        close_df = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        close_df = close_df.dropna(axis=1, how="all")
        if len(close_df.columns) < len(tickers):
            failed_tickers = set(tickers) - set(close_df.columns)
            self.logger.warning("Failed to fetch price data for: %s", ", ".join(failed_tickers))
        return close_df.ffill()

    def fetch_market_caps(self, tickers: List[str]) -> pd.Series:
        """Fetch market capitalization data for the given tickers.

        Args:
            tickers (List[str]): List of ticker symbols to fetch market cap for.

        Returns:
            pd.Series: Series containing market cap values indexed by ticker symbols.

        """
        self.logger.info("Fetching market cap data for %d tickers...", len(tickers))
        market_caps = {}
        for ticker_symbol in tickers:
            try:
                ticker_obj = self.provider.Ticker(ticker_symbol)
                m_cap = getattr(ticker_obj, "info", {}).get("marketCap")
                if m_cap is not None:
                    market_caps[ticker_symbol] = m_cap
                else:
                    self.logger.warning(
                        "Market cap not available for %s. Defaulting to 0.", ticker_symbol
                    )
                    market_caps[ticker_symbol] = 0
            except Exception as e:
                self.logger.error(
                    "Failed to fetch market cap for %s due to an error: %s. Defaulting to 0.",
                    ticker_symbol,
                    e,
                )
                market_caps[ticker_symbol] = 0
        return pd.Series(market_caps)

    def fetch_ticker_info(self, ticker_symbol: str) -> Dict:
        """Fetch detailed information for a specific ticker symbol.

        Args:
            ticker_symbol (str): Ticker symbol to fetch information for.

        Returns:
            Dict: Dictionary containing detailed information about the ticker, such as market cap,
            sector, industry, and other relevant data.

        """
        ticker_obj = self.provider.Ticker(ticker_symbol)
        return getattr(ticker_obj, "info", {})

    def fetch_cashflow(self, ticker_symbol: str):
        """Fetch the cash flow statement for a specific ticker symbol.

        Args:
            ticker_symbol (str): Ticker symbol to fetch cash flow data for.

        Returns:
            _type_: Cash flow statement as a DataFrame or None if not available.

        """
        ticker_obj = self.provider.Ticker(ticker_symbol)
        return getattr(ticker_obj, "cashflow", None)
