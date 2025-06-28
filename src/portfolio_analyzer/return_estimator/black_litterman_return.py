import numpy as np
import pandas as pd

from portfolio_analyzer.config.config import AppConfig
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
        config: AppConfig,
    ):
        # Align tickers and ensure consistent ordering
        tickers = log_returns.columns.intersection(market_cap_weights.index).sort_values()
        self.tickers = tickers
        self.config = config

        self.log_returns = log_returns[tickers]
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.market_cap_weights = market_cap_weights.reindex(tickers)
        self.tau = tau

        # Align view matrices/vectors
        assets_in_view = assets_in_view.sort_index()
        view_vector = view_vector.sort_index()
        view_confidence = view_confidence.sort_index()

        self.excess_returns_cov = self._excess_returns_covariance()
        self.implied_equilibrium_returns = self._implied_excess_equilibrium_returns()

        # Align view to tickers and ensure all dimensions match
        view_tickers = assets_in_view.columns.intersection(tickers)
        view_names = assets_in_view.index.intersection(view_vector.index).intersection(
            view_confidence.index
        )
        self.assets_in_view = assets_in_view.loc[view_names, view_tickers]
        self.view_confidence = view_confidence.loc[view_names, view_names]
        self.view_vector = view_vector.loc[view_names]

        self.posterior_returns = self._posterior_returns()

    def _validate_dimensions(self):
        assert all(self.assets_in_view.columns == self.market_cap_weights.index), (
            "Asset columns not aligned!"
        )
        assert all(self.excess_returns_cov.columns == self.market_cap_weights.index), (
            "Covariance columns not aligned!"
        )

        assert all(self.assets_in_view.index == self.view_confidence.index), (
            "View indices not aligned!"
        )
        assert all(self.view_confidence.index == self.view_confidence.columns), (
            "View confidence matrix not square/aligned!"
        )
        assert all(self.assets_in_view.index == self.view_vector.index), (
            "View vector and assets_in_view not aligned!"
        )

    def _implied_excess_equilibrium_returns(self) -> pd.Series:
        """Compute the implied equilibrium excess returns (Pi) using the CAPM prior.

        Returns:
            pd.Series: Implied equilibrium excess returns

        """
        pi = self.risk_aversion * self.excess_returns_cov.values @ self.market_cap_weights.values
        return pd.Series(pi, index=self.tickers)

    def _posterior_returns(self) -> pd.Series:
        """Compute the Black-Litterman posterior returns.

        Returns:
            pd.Series: Posterior returns

        """
        tau_cov = self.tau * self.excess_returns_cov.values
        tau_cov_inv = np.linalg.inv(tau_cov)
        P = self.assets_in_view.values
        Q = self.view_vector.values
        Omega_inv = np.linalg.inv(self.view_confidence.values)

        left = np.linalg.inv(tau_cov_inv + P.T @ Omega_inv @ P)
        right = tau_cov_inv @ self.implied_equilibrium_returns.values + P.T @ Omega_inv @ Q
        posterior = left @ right
        return pd.Series(posterior, index=self.tickers)

    def _excess_returns_covariance(self) -> pd.DataFrame:
        """Compute the covariance matrix of excess returns.

        Returns:
            pd.DataFrame: Covariance matrix of excess returns

        """
        log_risk_free_rate = np.log(
            (1 + self.risk_free_rate) ** (1 / self.config.trading_days_per_year)
        )
        excess_returns = self.log_returns - log_risk_free_rate
        cov = np.cov(excess_returns.values, rowvar=False)
        return pd.DataFrame(cov, index=self.tickers, columns=self.tickers)

    def get_posterior_returns(self) -> pd.Series:
        """Annualized Black-Litterman posterior returns.

        Returns:
            pd.Series: Annualized Black-Litterman posterior returns

        """
        return self.posterior_returns * self.config.trading_days_per_year

    def get_implied_equilibrium_returns(self) -> pd.Series:
        """Annualized implied equilibrium returns.

        Returns:
            pd.Series: Annualized implied equilibrium returns

        """
        return self.implied_equilibrium_returns * self.config.trading_days_per_year

    def get_returns(self) -> pd.Series:
        """Alias for get_posterior_returns.

        Returns:
            pd.Series: Annualized Black-Litterman posterior returns

        """
        return self.get_posterior_returns()
