import numpy as np
import pandas as pd

from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.return_estimator.return_estimator import ReturnEstimator

from ..data.data_fetcher import DataFetcher
from ..utils.util import calculate_log_returns


class BlackLittermanReturn(ReturnEstimator):
    def __init__(
        self,
        view_vector: pd.Series,
        assets_in_view: pd.DataFrame = None,
        view_confidence: pd.DataFrame = None,
        config: AppConfig = None,
        data_fetcher: DataFetcher = None,
    ):
        # Align tickers and ensure consistent ordering
        self.config = config or AppConfig.get_instance()
        self.data_fetcher = data_fetcher
        self.log_returns = calculate_log_returns(
            data_fetcher.fetch_price_data(
                config.tickers, config.date_range.start, config.date_range.end
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
        market_cap = self.data_fetcher.fetch_market_caps(tickers)
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
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        lw.fit(self.log_returns)
        daily_cov = pd.DataFrame(
            lw.covariance_,
            index=self.tickers,
            columns=self.tickers,
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
