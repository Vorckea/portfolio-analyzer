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
    def __init__(self, provider: MarketDataProvider, logger: Optional[logging.Logger] = None):
        self.provider: MarketDataProvider = provider
        self.logger = logger or logging.getLogger(__name__)

    def fetch_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
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
        ticker_obj = self.provider.Ticker(ticker_symbol)
        return getattr(ticker_obj, "info", {})

    def fetch_cashflow(self, ticker_symbol: str):
        ticker_obj = self.provider.Ticker(ticker_symbol)
        return getattr(ticker_obj, "cashflow", None)
