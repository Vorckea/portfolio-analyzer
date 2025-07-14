import logging
from typing import Optional

import pandas as pd

from portfolio_analyzer.return_estimator.return_estimator import ReturnEstimator

from ..config.config import AppConfig
from ..data.data_fetcher import DataFetcher
from ..utils.util import calculate_log_returns


class CAPM(ReturnEstimator):
    # ER_i = R_f + B_i * (ER_m - R_f)

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tickers: str,
        config: AppConfig,
        data_fetcher: DataFetcher,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        self.risk_free_rate = self.config.risk_free_rate
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self._returns = self._calculate_capm_returns()

    def _calculate_capm_returns(self) -> pd.Series:
        market_ticker = "OSEBX.OL"

        market_info = self.data_fetcher.fetch_ticker_info(market_ticker)
        er_market = market_info.get("expectedReturn")

        if er_market is None:
            price_data = self.data_fetcher.fetch_price_data(
                [market_ticker],
                self.start_date,
                self.end_date,
            )
            log_returns = calculate_log_returns(price_data)
            er_market = log_returns.mean().iloc[0] * self.config.trading_days_per_year

        returns = {}

        for ticker in self.tickers:
            info = self.data_fetcher.fetch_ticker_info(ticker)
            beta = info.get("beta")
            if beta is None:
                returns[ticker] = 0
                continue
            returns[ticker] = self.risk_free_rate + beta * (er_market - self.risk_free_rate)
            self.logger.debug(
                f"Ticker: {ticker}, Beta: {beta:.2f}, Expected Return: {returns[ticker]:.4f}"
            )
        return pd.Series(returns)

    def get_returns(self) -> pd.Series:
        return self._returns
