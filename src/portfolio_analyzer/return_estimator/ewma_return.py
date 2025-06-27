import pandas as pd

from portfolio_analyzer.return_estimator.return_estimator import ReturnEstimator


class EWMAReturn(ReturnEstimator):
    def __init__(
        self, log_returns: pd.DataFrame, span: int, trading_days: int, shrinkage_factor: float = 0
    ):
        self.log_returns = log_returns
        self.span = span
        self.trading_days = trading_days
        self.shrinkage_factor = shrinkage_factor
        self.ewma_returns = self._calculate_ewma_returns()
        self.shrinked_ewma_returns = self._shrink_ewma()

    def _calculate_ewma_returns(self) -> pd.Series:
        return self.log_returns.ewm(span=self.span).mean().iloc[-1] * self.trading_days

    def _shrink_ewma(self) -> pd.Series:
        if self.shrinkage_factor <= 0:
            return self.ewma_returns

        grand_mean = self.ewma_returns.mean()
        return (1 - self.shrinkage_factor) * self.ewma_returns + self.shrinkage_factor * grand_mean

    def get_ewma_returns(self) -> pd.Series:
        return self.ewma_returns

    def get_shrinked_ewma_returns(self) -> pd.Series:
        return self.shrinked_ewma_returns

    def get_returns(self) -> pd.Series:
        if self.shrinkage_factor > 0:
            return self.get_shrinked_ewma_returns()
        return self.get_ewma_returns()
