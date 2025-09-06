import re
from datetime import datetime
from typing import Annotated

import pandas as pd
import pandera.pandas as pa
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

TICKER_REGEX = re.compile(r"^[A-Za-z0-9\.-]{1,10}$")

PRICE_HISTORY_SCHEMA = pa.DataFrameSchema(
    index=pa.Index(
        pa.DateTime,
        coerce=True,
        name="Datetime",
        unique=True,
        checks=[
            pa.Check(
                lambda idx: idx.is_monotonic_increasing,
                element_wise=False,
                error="Index must be sorted ascending",
            ),
        ],
    ),
    columns=(
        {
            "": pa.Column(
                pa.Float,
                checks=[
                    pa.Check(
                        lambda s: s.notna().all(),
                        element_wise=False,
                        error="Price columns must not contain nulls",
                    ),
                    pa.Check(
                        lambda s: (s > 0).all(),
                        element_wise=False,
                        error="Price values must be positive",
                    ),
                ],
                regex=True,
            )
        }
    ),
    checks=[
        pa.Check(
            lambda df: all(isinstance(c, str) and TICKER_REGEX.match(c) for c in df.columns),
            element_wise=False,
            error="All column names must be ticker-like strings (e.g. AAPL, MSFT)",
        )
    ],
)

VOLUME_HISTORY_SCHEMA = pa.DataFrameSchema(
    index=pa.Index(
        pa.DateTime,
        coerce=True,
        name="Datetime",
        unique=True,
        checks=[
            pa.Check(
                lambda idx: idx.is_monotonic_increasing,
                element_wise=False,
                error="Index must be sorted ascending",
            ),
        ],
    ),
    columns=(
        {
            "": pa.Column(
                pa.Int,
                checks=[
                    pa.Check(
                        lambda s: s.notna().all(),
                        element_wise=False,
                        error="Volume columns must not contain nulls",
                    ),
                    pa.Check(
                        lambda s: (s >= 0).all(),
                        element_wise=False,
                        error="Volume values must be non-negative",
                    ),
                    pa.Check(
                        lambda s: (
                            lambda float_values: (float_values.round(0) == float_values).all()
                        )(s.apply(float)),
                        element_wise=False,
                        error="Volume values should be integral",
                    ),
                ],
                regex=True,
            )
        }
    ),
    checks=[
        pa.Check(
            lambda df: all(isinstance(c, str) and TICKER_REGEX.match(c) for c in df.columns),
            element_wise=False,
            error="All column names must be ticker-like strings (e.g. AAPL, MSFT)",
        )
    ],
)

VolumeDataFrame = Annotated[pd.DataFrame, AfterValidator(VOLUME_HISTORY_SCHEMA.validate)]
PriceDataFrame = Annotated[pd.DataFrame, AfterValidator(PRICE_HISTORY_SCHEMA.validate)]


class PriceHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Needed for pd.DataFrame

    prices: PriceDataFrame
    volume: VolumeDataFrame | None = None
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
