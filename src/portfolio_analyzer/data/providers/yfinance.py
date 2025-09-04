from datetime import datetime

import pandas as pd
import yfinance as yf

from ...core.interfaces import BaseDataProvider
from ...core.models import PriceHistory


class YahooFinanceDataProvider(BaseDataProvider):
    def fetch_price_history(
        self,
        asset_list: list[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> PriceHistory:
        df = yf.download(
            tickers=asset_list,
            start=start,
            end=end,
            interval=frequency,
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
            threads=True,
        )
        return PriceHistory(
            prices=df["Close"],
            volume=df["Volume"],
            start_date=df.index.min(),
            end_date=df.index.max(),
            frequency=frequency,
        )

    def fetch_symbol_info(self, symbol: str) -> dict:
        info = yf.Ticker(symbol).info
        if not info:
            return {}
        return info

    def fetch_cashflow(self, symbol: str) -> pd.Series | None:
        cashflow_df = yf.Ticker(symbol).cashflow
        if cashflow_df.empty:
            return None

        cashflow = cashflow_df.loc["Free Cash Flow"]
        if cashflow.empty:
            return None

        return cashflow

    def available_assets(self) -> list[str]:
        # yfinance does not provide a method to list all available assets
        raise NotImplementedError("Listing all available assets is not supported by yfinance.")
