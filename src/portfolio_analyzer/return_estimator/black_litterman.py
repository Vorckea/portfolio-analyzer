import numpy as np
import pandas as pd

from ..config.config import AppConfig
from ..data.repository import Repository
from ..utils.util import calculate_annualized_covariance, calculate_log_returns
from .base import ReturnEstimator


class BlackLitterman(ReturnEstimator):
    def __init__(
        self,
        view_vector: pd.Series,
        start_date: str,
        end_date: str,
        tickers: str,
        assets_in_view: pd.DataFrame | None = None,
        view_confidence: pd.DataFrame | None = None,
        config: AppConfig = None,
        repository: Repository = None,
    ):
        # Align tickers and ensure consistent ordering
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or AppConfig.get_instance()
        self.repository = repository
        self.log_returns = calculate_log_returns(
            close_df=repository.fetch_price_data(
                tickers=self.tickers,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        )
        market_cap_weights = self._calculate_market_cap_weights(self.log_returns)
        self.tickers = self._align_tickers(self.log_returns, market_cap_weights)
        self.log_returns = self.log_returns[self.tickers]

        self.risk_free_rate = config.risk_free_rate
        self.risk_aversion = config.black_litterman.delta
        self.market_cap_weights = market_cap_weights.reindex(self.tickers)
        self.tau = config.black_litterman.tau

        # Align view matrices/vectors
        self.view_vector = view_vector.sort_index()
        self.assets_in_view = (
            assets_in_view.sort_index()
            if assets_in_view is not None
            else self._generate_assets_in_view(self.view_vector)
        )
        self.view_confidence = (
            view_confidence.sort_index()
            if view_confidence is not None
            else self._generate_view_confidence(self.assets_in_view)
        )

        self._align_views()

        self.excess_returns_cov = self._excess_returns_covariance()
        self.implied_equilibrium_returns = self._implied_excess_equilibrium_returns()
        self.posterior_returns = self._posterior_returns()

    def _calculate_market_cap_weights(self, log_returns: pd.DataFrame) -> pd.Series:
        tickers = log_returns.columns.tolist()
        market_cap = self.repository.fetch_market_caps(tickers)
        if market_cap is None or market_cap.empty:
            raise ValueError("Market cap weights cannot be None or empty.")
        market_cap_weights = market_cap / market_cap.sum()
        if market_cap_weights is None or market_cap_weights.empty:
            raise ValueError("Market cap weights cannot be None or empty.")

        return market_cap_weights.reindex(tickers).fillna(0.0)

    @staticmethod
    def _align_tickers(log_returns: pd.DataFrame, market_cap_weights: pd.Series) -> pd.Index:
        return log_returns.columns.intersection(market_cap_weights.index).sort_values()

    def _align_views(self):
        tickers = self.tickers
        view_tickers = self.assets_in_view.columns.intersection(tickers)
        view_names = self.assets_in_view.index.intersection(self.view_vector.index).intersection(
            self.view_confidence.index
        )
        self.assets_in_view = self.assets_in_view.loc[view_names, view_tickers]
        self.view_confidence = self.view_confidence.loc[view_names, view_names]
        self.view_vector = self.view_vector.loc[view_names] / self.config.trading_days_per_year

    def _generate_assets_in_view(self, view_vector: pd.Series) -> pd.DataFrame:
        assets = view_vector.index
        columns = self.tickers
        P = pd.DataFrame(0, index=assets, columns=columns)
        for asset in assets:
            if asset in columns:
                P.loc[asset, asset] = 1.0
        return P.sort_index()

    def _generate_view_confidence(self, assets_in_view: pd.DataFrame) -> pd.DataFrame:
        cov = (
            calculate_annualized_covariance(
                log_returns=self.log_returns, trading_days=self.config.trading_days_per_year
            )
            / self.config.trading_days_per_year
        ).sort_index()
        tau = self.tau
        P = assets_in_view.values
        confidence = 0.95
        view_variances = []
        for i in range(P.shape[0]):
            Pi = P[i, :]
            variance = (1 - confidence) * (tau * Pi @ cov.values @ Pi.T)
            view_variances.append(variance)
        Omega = np.diag(view_variances)
        return pd.DataFrame(
            Omega, index=assets_in_view.index, columns=assets_in_view.index
        ).sort_index()

    def _generate_view_confidence_old(self, assets_in_view: pd.DataFrame) -> pd.DataFrame:
        daily_cov = (
            calculate_annualized_covariance(
                log_returns=self.log_returns,
                trading_days=self.config.trading_days_per_year,
            )
            / self.config.trading_days_per_year
        ).sort_index()

        tau = self.tau
        assets = assets_in_view.index
        view_variance = tau * daily_cov.loc[assets, assets].values.diagonal()
        Omega = pd.DataFrame(
            np.diag(view_variance),
            index=assets,
            columns=assets,
        ).sort_index()
        return Omega

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
