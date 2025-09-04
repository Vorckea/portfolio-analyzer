from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict


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
