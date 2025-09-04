from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PortfolioSpec(BaseModel):
    assets: list[str]
    constraints: dict[str, Any]
    objective: str


class PortfolioResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Needed for pd.Series

    weights: pd.Series
    metrics: dict[str, float]


class PriceHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Needed for pd.DataFrame

    prices: pd.DataFrame
    volume: pd.DataFrame | None
    start_date: datetime
    end_date: datetime
    frequency: str

    @property
    def duration(self) -> pd.Timedelta:
        return self.end_date - self.start_date

    @property
    def assets(self) -> list[str]:
        return self.prices.columns.tolist()


class SymbolInfo(BaseModel):
    country: str
    region: str
    long_name: str = Field(alias="longName")
    short_name: str = Field(alias="shortName")
    symbol: str
    website: str
    industry: str
    sector: str
    beta: float
    trailing_pe: float = Field(alias="trailingPE")
    forward_pe: float = Field(alias="forwardPE")
    trailing_eps: float = Field(alias="epsTrailingTwelveMonths")
    forward_eps: float = Field(alias="epsForward")
    market_cap: int = Field(alias="marketCap")
    fifty_two_week_high: float = Field(alias="fiftyTwoWeekHigh")
    fifty_two_week_low: float = Field(alias="fiftyTwoWeekLow")
    fifty_day_average: float = Field(alias="fiftyDayAverage")
    two_hundred_day_average: float = Field(alias="twoHundredDayAverage")
    currency: str
    shares_outstanding: int = Field(alias="sharesOutstanding")
    market: str
    exchange: str
    debt_to_equity: float = Field(alias="debtToEquity")
    short_ratio: float = Field(alias="shortRatio")
    price_to_book: float = Field(alias="priceToBook")
    book_value: float = Field(alias="bookValue")
