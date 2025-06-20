import pandas as pd
import yfinance as yf

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.dcf_calculator import DCFCalculator


def fetch_price_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical adjusted close prices from yfinance, skipping any that fail."""
    print("Fetching historical price data from yfinance...")
    # Use yfinance's built-in grouping for efficiency
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError("No price data fetched for any tickers.")

    # Handle multi-level columns if multiple tickers are downloaded
    close_df = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]

    # Drop tickers that failed to download (all NaN columns)
    close_df = close_df.dropna(axis=1, how="all")

    if len(close_df.columns) < len(tickers):
        failed_tickers = set(tickers) - set(close_df.columns)
        print(f"\nWarning: Failed to fetch price data for: {', '.join(failed_tickers)}")

    # Forward-fill to handle intermittent missing values
    return close_df.ffill()


def calculate_dcf_views(config: AppConfig) -> dict[str, float]:
    """Calculates DCF-based views for a list of tickers."""
    print("\nCalculating DCF-based views...")
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
            print(f"  - View for {ticker}: {expected_return:.2%}")

    if not views:
        print("No valid DCF views could be generated.")

    return views


def fetch_market_caps(tickers: list[str]) -> pd.Series:
    """Fetches market capitalization data from yfinance."""
    print("Fetching market cap data from yfinance...")
    market_caps = {}
    for ticker_symbol in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            m_cap = ticker_obj.info.get("marketCap", None)
            market_caps[ticker_symbol] = m_cap if m_cap is not None else 0
        except Exception:
            # yfinance can be flaky, so we default to 0 on any error
            market_caps[ticker_symbol] = 0
    return pd.Series(market_caps)
