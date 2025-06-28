import logging
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from portfolio_analyzer.config.config import AppConfig


class DataFetcher:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        yf_module=yf,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.yf = yf_module

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.logger.info("Fetching historical price data for %d tickers...", len(tickers))
        data = self.yf.download(
            tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
        )
        if data.empty:
            self.logger.error(
                "No price data fetched for any tickers. yfinance returned an empty DataFrame."
            )
            raise ValueError("No price data fetched for any tickers.")
        close_df = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        close_df = close_df.dropna(axis=1, how="all")
        if len(close_df.columns) < len(tickers):
            failed_tickers = set(tickers) - set(close_df.columns)
            self.logger.warning("Failed to fetch price data for: %s", ", ".join(failed_tickers))
        return close_df.ffill()

    def fetch_market_caps(self, tickers: list[str]) -> pd.Series:
        self.logger.info("Fetching market cap data for %d tickers...", len(tickers))
        market_caps = {}
        for ticker_symbol in tickers:
            try:
                ticker_obj = self.yf.Ticker(ticker_symbol)
                m_cap = ticker_obj.info.get("marketCap")
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

    def calculate_dcf_views(self, config: AppConfig) -> Dict[str, float]:
        from portfolio_analyzer.data.dcf_calculator import DCFCalculator  # <-- Import here, locally

        self.logger.info("Calculating DCF-based views for %d tickers...", len(config.tickers))
        views = {}
        for ticker in config.tickers:
            calculator = DCFCalculator(
                ticker_symbol=ticker,
                config=config.dcf,
                risk_free_rate=config.risk_free_rate,
                data_fetcher=self,
            )
            expected_return = calculator.calculate_expected_return()
            if expected_return is not None:
                views[ticker] = expected_return
                self.logger.debug("  - View for %s: %.2f%%", ticker, expected_return * 100)
        if not views:
            self.logger.warning("No valid DCF views could be generated.")
        else:
            self.logger.info("Successfully generated %d DCF views.", len(views))
        return views

    def fetch_ticker_info(self, ticker_symbol: str) -> Dict:
        ticker_obj = self.yf.Ticker(ticker_symbol)
        return ticker_obj.info

    def fetch_cashflow(self, ticker_symbol: str):
        ticker_obj = self.yf.Ticker(ticker_symbol)
        return ticker_obj.cashflow
