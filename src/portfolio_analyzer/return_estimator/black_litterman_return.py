import numpy as np
import pandas as pd

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.return_estimator.return_estimator import ReturnEstimator


class BlackLittermanReturn(ReturnEstimator):
    def __init__(
        self,
        log_returns: pd.DataFrame,
        risk_free_rate: float,
        risk_aversion: float,
        market_cap_weights: pd.Series,
        tau: float,
        assets_in_view: pd.DataFrame,
        view_confidence: pd.DataFrame,
        view_vector: pd.Series,
        config: AppConfig = None,
    ):
        tickers = log_returns.columns.intersection(market_cap_weights.index).sort_values()
        self.tickers = tickers
        self.config = config

        self.log_returns = log_returns[tickers]
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.market_cap_weights = market_cap_weights.reindex(tickers)
        self.tau = tau

        assets_in_view = assets_in_view.sort_index()
        view_vector = view_vector.sort_index()
        view_confidence = view_confidence.sort_index()

        self.excess_returns_cov = self._excess_returns_covariance()
        self.implied_equilibrium_returns = self._implied_excess_equilibrium_returns()

        view_tickers = assets_in_view.columns.intersection(tickers)
        view_names = assets_in_view.index.intersection(view_vector.index).intersection(
            view_confidence.index
        )
        self.assets_in_view = assets_in_view.loc[view_names, view_tickers]
        self.view_confidence = view_confidence.loc[view_names, view_names]
        self.view_vector = view_vector.loc[view_names]

        # Print shapes for debugging
        print(f"Log returns shape: {self.log_returns.shape}")
        print(f"Market cap weights shape: {self.market_cap_weights.shape}")
        print(f"Excess returns covariance shape: {self.excess_returns_cov.shape}")
        print(f"Implied equilibrium returns shape: {self.implied_equilibrium_returns.shape}")
        print(f"Assets in view shape: {self.assets_in_view.shape}")
        print(f"View confidence shape: {self.view_confidence.shape}")
        print(f"View vector shape: {self.view_vector.shape}")

        print("assets_in_view columns:", list(self.assets_in_view.columns))
        print("market_cap_weights index:", list(self.market_cap_weights.index))
        print("excess_returns_cov columns:", list(self.excess_returns_cov.columns))
        print("assets_in_view index:", list(self.assets_in_view.index))
        print("view_confidence index:", list(self.view_confidence.index))
        print("view_confidence columns:", list(self.view_confidence.columns))
        print("view_vector index:", list(self.view_vector.index))

        assert all(self.assets_in_view.columns == self.market_cap_weights.index), (
            "Asset columns not aligned!"
        )
        assert all(self.excess_returns_cov.columns == self.market_cap_weights.index), (
            "Covariance columns not aligned!"
        )

        # Check view alignment
        assert all(self.assets_in_view.index == self.view_confidence.index), (
            "View indices not aligned!"
        )
        assert all(self.view_confidence.index == self.view_confidence.columns), (
            "View confidence matrix not square/aligned!"
        )
        assert all(self.assets_in_view.index == self.view_vector.index), (
            "View vector and assets_in_view not aligned!"
        )

        self.posterior_returns = self._posterior_returns()

    def _implied_excess_equilibrium_returns(self) -> pd.Series:
        print("delta:", self.risk_aversion)
        print("w_mkt sum:", self.market_cap_weights.sum())
        print("Covariance matrix mean:", self.excess_returns_cov.values.mean())
        print("Covariance matrix diag:", self.excess_returns_cov.values.diagonal())

        return pd.Series(
            self.risk_aversion * self.excess_returns_cov.values @ self.market_cap_weights.values,
            index=self.tickers,
        )

    def _posterior_returns(self) -> pd.Series:
        tau_cov_inv = np.linalg.inv(self.tau * self.excess_returns_cov.values)
        left_term = np.linalg.inv(
            tau_cov_inv
            + self.assets_in_view.T.values
            @ np.linalg.inv(self.view_confidence.values)
            @ self.assets_in_view.values
        )
        right_term = (
            tau_cov_inv @ self.implied_equilibrium_returns.values
            + self.assets_in_view.T.values
            @ np.linalg.inv(self.view_confidence.values)
            @ self.view_vector.values
        )
        posterior = left_term @ right_term
        return pd.Series(posterior, index=self.tickers)

    def _excess_returns_covariance(self) -> pd.DataFrame:
        log_risk_free_rate = np.log(
            (1 + self.risk_free_rate) ** (1 / self.config.trading_days_per_year)
        )
        excess_returns = self.log_returns - log_risk_free_rate
        cov = np.cov(excess_returns.values, rowvar=False)
        return pd.DataFrame(cov, index=self.tickers, columns=self.tickers)

    def get_posterior_returns(self) -> pd.Series:
        return self.posterior_returns * self.config.trading_days_per_year

    def get_implied_equilibrium_returns(self) -> pd.Series:
        return self.implied_equilibrium_returns * self.config.trading_days_per_year

    def get_returns(self) -> pd.Series:
        return self.get_posterior_returns()
