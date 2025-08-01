from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class AssetUniverse:
    tickers: tuple[str, ...] = field(metadata={"description": "Asset tickers"})

    def __post_init__(self):
        if not self.tickers:
            raise ValueError("tickers must not be empty.")


@dataclass(frozen=True, slots=True)
class ReturnEstimates:
    expected_returns: pd.Series = field(metadata={"description": "Expected/forecasted returns"})
    covariance: pd.DataFrame = field(metadata={"description": "Covariance matrix"})
    historical_mean: pd.Series | None = field(
        default=None, metadata={"description": "Historical mean returns"}
    )

    def __post_init__(self):
        if self.expected_returns.empty or self.covariance.empty:
            raise ValueError("Mean returns and covariance matrix must not be empty.")
        if set(self.expected_returns.index) != set(self.covariance.index):
            raise ValueError("Tickers in mean and covariance must match.")
        if self.historical_mean is not None and set(self.historical_mean.index) != set(
            self.expected_returns.index
        ):
            raise ValueError("Tickers in historical_mean must match mean returns.")


@dataclass(frozen=True, slots=True)
class PriceData:
    log_returns: pd.DataFrame
    close_prices: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ModelInputs:
    universe: AssetUniverse
    returns: ReturnEstimates
    prices: PriceData

    def __post_init__(self):
        tickers = set(self.universe.tickers)
        if set(self.returns.expected_returns.index) != tickers:
            raise ValueError("Tickers in returns do not match universe.")
        if set(self.prices.log_returns.columns) != tickers:
            raise ValueError("Tickers in prices do not match universe.")
