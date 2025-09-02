"""Typed Pydantic models for input data used by optimizers and estimators."""

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class AssetUniverse(BaseModel):
    """Represent the universe of asset tickers.

    Accepts any iterable of strings and normalizes to a tuple.
    """

    tickers: tuple[str, ...] = Field(..., description="Asset tickers")

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @model_validator(mode="before")
    def _ensure_tuple(cls, values: dict) -> dict:
        tickers = values.get("tickers")
        if tickers is None:
            raise ValueError("tickers must be provided")
        # allow lists/sets/tuples
        if not isinstance(tickers, tuple):
            values["tickers"] = tuple(tickers)
        if len(values["tickers"]) == 0:
            raise ValueError("tickers must not be empty")
        return values

    # pandas objects are arbitrary types; model_config enables that


class ReturnEstimates(BaseModel):
    """Return estimates and covariance used by optimizers.

    Attributes
    ----------
    expected_returns : pd.Series
        Expected or forecasted returns indexed by ticker.
    covariance : pd.DataFrame
        Covariance matrix with tickers as both index and columns.
    historical_mean : pd.Series | None
        Optional historical mean returns indexed by ticker.

    """

    expected_returns: pd.Series = Field(..., description="Expected/forecasted returns")
    covariance: pd.DataFrame = Field(..., description="Covariance matrix")
    historical_mean: pd.Series | None = Field(default=None, description="Historical mean returns")

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate_indexes(self) -> "ReturnEstimates":
        if self.expected_returns is None or self.covariance is None:
            raise ValueError("Mean returns and covariance matrix must not be empty.")
        if set(self.expected_returns.index) != set(self.covariance.index):
            raise ValueError("Tickers in mean and covariance must match.")
        if self.historical_mean is not None and set(self.historical_mean.index) != set(
            self.expected_returns.index
        ):
            raise ValueError("Tickers in historical_mean must match mean returns.")
        return self


class PriceData(BaseModel):
    """Price series and log returns dataframes.

    Both dataframes are expected to use tickers as columns.
    """

    log_returns: pd.DataFrame
    close_prices: pd.DataFrame

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class ModelInputs(BaseModel):
    """Container that groups universe, return estimates and price data.

    Ensures tickers align across the nested models.
    """

    universe: AssetUniverse
    returns: ReturnEstimates
    prices: PriceData

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _match_universe(self) -> "ModelInputs":
        tickers = set(self.universe.tickers)
        if set(self.returns.expected_returns.index) != tickers:
            raise ValueError("Tickers in returns do not match universe.")
        if set(self.prices.log_returns.columns) != tickers:
            raise ValueError("Tickers in prices do not match universe.")
        return self
