import logging

import pandas as pd
import yfinance as yf

from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.data.dcf_calculator import DCFCalculator

logger = logging.getLogger(__name__)


def fetch_price_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical adjusted close prices from yfinance, skipping any that fail."""
    logger.info("Fetching historical price data for %d tickers...", len(tickers))
    # Use yfinance's built-in grouping for efficiency
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if data.empty:
        logger.error("No price data fetched for any tickers. yfinance returned an empty DataFrame.")
        raise ValueError("No price data fetched for any tickers.")

    # Handle multi-level columns if multiple tickers are downloaded
    close_df = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]

    # Drop tickers that failed to download (all NaN columns)
    close_df = close_df.dropna(axis=1, how="all")

    if len(close_df.columns) < len(tickers):
        failed_tickers = set(tickers) - set(close_df.columns)
        logger.warning("Failed to fetch price data for: %s", ", ".join(failed_tickers))

    # Forward-fill to handle intermittent missing values
    return close_df.ffill()


def calculate_dcf_views(config: AppConfig) -> dict[str, float]:
    """Calculates DCF-based views for a list of tickers."""
    logger.info("Calculating DCF-based views for %d tickers...", len(config.tickers))
    views = {}
    for ticker in config.tickers:
        # The DCFCalculator now handles its own internal errors and logging.
        # This loop just needs to catch the result.
        calculator = DCFCalculator(
            ticker_symbol=ticker,
            config=config.dcf,
            risk_free_rate=config.risk_free_rate,
        )
        expected_return = calculator.calculate_expected_return()

        if expected_return is not None:
            views[ticker] = expected_return
            logger.debug("  - View for %s: %.2f%%", ticker, expected_return * 100)

    if not views:
        logger.warning("No valid DCF views could be generated.")
    else:
        logger.info("Successfully generated %d DCF views.", len(views))

    return views


def fetch_market_caps(tickers: list[str]) -> pd.Series:
    """Fetches market capitalization data from yfinance."""
    logger.info("Fetching market cap data for %d tickers...", len(tickers))
    market_caps = {}
    for ticker_symbol in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            m_cap = ticker_obj.info.get("marketCap")
            if m_cap is not None:
                market_caps[ticker_symbol] = m_cap
            else:
                logger.warning("Market cap not available for %s. Defaulting to 0.", ticker_symbol)
                market_caps[ticker_symbol] = 0

        except Exception as e:
            logger.error(
                "Failed to fetch market cap for %s due to an error: %s. Defaulting to 0.",
                ticker_symbol,
                e,
            )
            market_caps[ticker_symbol] = 0
    return pd.Series(market_caps)
