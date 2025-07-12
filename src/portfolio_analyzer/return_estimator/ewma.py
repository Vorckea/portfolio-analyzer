import pandas as pd

from portfolio_analyzer.return_estimator.return_estimator import ReturnEstimator

from ..config.config import AppConfig
from ..data.data_fetcher import DataFetcher
from ..utils.util import calculate_log_returns


class EWMA(ReturnEstimator):
    """Exponential Weighted Moving Average (EWMA) Return Estimator with optional shrinkage."""

    def __init__(
        self,
        data_fetcher: DataFetcher,
        config: AppConfig = None,
    ):
        """Exponential Weighted Moving Average (EWMA) Return Estimator with optional shrinkage.

        Args:
            log_returns (pd.DataFrame): Logarithmic returns of the assets.
            span (int): The span for the EWMA calculation.
            trading_days (int): The number of trading days in a year.
            shrinkage_factor (float, optional): Shrinkage intensity [0, 1].
            Defaults to 0 (no shrinkage).

        """
        self.config = config if config else AppConfig.get_instance()
        self.data_fetcher = data_fetcher
        self.log_returns = calculate_log_returns(
            data_fetcher.fetch_price_data(
                config.tickers, config.date_range.start, config.date_range.end
            )
        )
        self.span = config.ewma_span
        self.trading_days = config.trading_days_per_year
        self.shrinkage_factor = config.mean_shrinkage_alpha

        self.ewma_returns = self._calculate_ewma_returns()
        self.shrinked_ewma_returns = self._apply_shrinkage()

    def _calculate_ewma_returns(self) -> pd.Series:
        """Calculate annualized EWMA returns for each asset.

        Returns:
            pd.Series: Annualized EWMA returns for each asset

        """
        return self.log_returns.ewm(span=self.span).mean().iloc[-1] * self.trading_days

    def _apply_shrinkage(self) -> pd.Series:
        """Apply shrinkage to the EWMA returns if shrinkage_factor is > 0.

        Shrinks the EWMA returns towards the grand mean of the returns.

        Returns:
            pd.Series: Shrinked EWMA returns

        """
        if self.shrinkage_factor <= 0:
            return self.ewma_returns
        grand_mean = self.ewma_returns.mean()
        return (1 - self.shrinkage_factor) * self.ewma_returns + self.shrinkage_factor * grand_mean

    def get_ewma_returns(self) -> pd.Series:
        """Get the annualized EWMA returns.

        Returns:
            pd.Series: Annualized EWMA returns

        """
        return self.ewma_returns

    def get_shrinked_ewma_returns(self) -> pd.Series:
        """Get the shrinked EWMA returns.

        Returns:
            pd.Series: Shrinked EWMA returns

        """
        return self.shrinked_ewma_returns

    def get_returns(self) -> pd.Series:
        """Get the appropriate EWMA returns (shrunk if shrinkage_factor > 0).

        Returns:
            pd.Series: EWMA returns (shrinked if shrinkage_factor > 0)

        """
        if self.shrinkage_factor > 0:
            return self.get_shrinked_ewma_returns()
        return self.get_ewma_returns()
