from datetime import datetime

import pandas as pd
import yfinance as yf

from ...core.interfaces import BaseDataProvider
from ..schema import PriceHistory, SymbolInfo


class YahooFinanceDataProvider(BaseDataProvider):
    def fetch_price_history(
        self,
        asset_list: list[str],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> PriceHistory:
        try:
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to download price history from Yahoo Finance for assets {asset_list}: {e}"
            )

        def _extract_field(frame: pd.DataFrame, field: str):
            # direct access for flat columns
            if field in frame.columns:
                return frame[field]

            # handle MultiIndex columns (either level 0 or level 1 may be the field)
            if isinstance(frame.columns, pd.MultiIndex):
                lvl0 = list(frame.columns.get_level_values(0))
                lvl1 = list(frame.columns.get_level_values(1))
                if field in lvl0:
                    return frame.xs(field, axis=1, level=0)
                if field in lvl1:
                    return frame.xs(field, axis=1, level=1)

                # fallback: collect any columns where any level equals the field
                cols = [
                    c
                    for c in frame.columns
                    if (isinstance(c, tuple) and (c[0] == field or c[1] == field))
                ]
                if cols:
                    res = frame.loc[:, cols]
                    # normalize column names to ticker symbols where possible
                    if cols and isinstance(cols[0], tuple):
                        res.columns = [c[1] if c[1] not in (None, "") else c[0] for c in cols]
                    return res

            raise KeyError(f"Field '{field}' not found in downloaded data")

        prices = _extract_field(df, "Close")

        try:
            volume = _extract_field(df, "Volume")
        except KeyError:
            volume = None

        return PriceHistory(
            prices=prices,
            volume=volume,
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            frequency=frequency,
        )

    def fetch_symbol_info(self, symbol: str) -> SymbolInfo:
        info = yf.Ticker(symbol).info
        if not info:
            return SymbolInfo(**{})
        return SymbolInfo(**info)

    def fetch_cashflow(self, symbol: str) -> pd.Series | None:
        cashflow_df = yf.Ticker(symbol).cashflow
        if cashflow_df.empty:
            return None

        try:
            cashflow = cashflow_df.loc["Free Cash Flow"]
        except KeyError:
            return None
        if cashflow.empty:
            return None

        return cashflow

    def available_assets(self) -> list[str]:
        # yfinance does not provide a method to list all available assets
        raise NotImplementedError("Listing all available assets is not supported by yfinance.")
