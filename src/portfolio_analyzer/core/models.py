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


class ReturnsFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Needed for pd.DataFrame

    returns: pd.Series
    covariance: pd.DataFrame
    frequency: str

    @property
    def assets(self) -> list[str]:
        return self.returns.index.tolist()
