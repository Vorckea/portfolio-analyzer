"""Black-Litterman Model for portfolio optimization.

This module implements the Black-Litterman model, which allows for the
incorporation of subjective views on asset returns into the mean-variance
optimization framework, producing a blended posterior estimate of returns.
"""

from typing import Optional

import numpy as np
import pandas as pd


class BlackLittermanModel:
    """Implements the Black-Litterman model for portfolio optimization.

    This model combines market-implied equilibrium returns with an investor's
    subjective views to produce a posterior estimate of expected returns. This
    posterior return vector can then be used in a mean-variance optimizer.

    # TODO(user): Add attributes once the class implementation is complete.
    """

    def __init__(
        self,
        tau: float,
        delta: float,
        w_mkt: pd.Series,
        cov_matrix: pd.DataFrame,
        P: Optional[pd.DataFrame] = None,
        Q: Optional[pd.Series] = None,
        Omega: Optional[pd.DataFrame] = None,
    ):
        """Initialize the Black-Litterman model.

        Args:
            tau (float): The scalar that scales the covariance matrix.
            delta (float): The risk aversion coefficient.
            w_mkt (pd.Series): Market weights of the assets, indexed by tickers.
            cov_matrix (pd.DataFrame): Covariance matrix of asset returns, indexed by tickers.
            P (Optional[pd.DataFrame], optional): Matrix of views, where each row corresponds to a
            view and each column corresponds to an asset. Defaults to None.
            Q (Optional[pd.Series], optional): Vector of views, where each element corresponds to a
            view in P. Defaults to None.
            Omega (Optional[pd.DataFrame], optional): Diagonal covariance matrix of the views,
            where the diagonal elements correspond to the uncertainty of each view.
            Defaults to None.

        """
        self.tau = tau
        self.delta = delta

        (
            self.tickers,
            self.aligned_cov_matrix,
            self.aligned_w_mkt,
            self.aligned_P,
            self.aligned_Q,
            self.aligned_Omega,
        ) = self._align_inputs(w_mkt, cov_matrix, P, Q, Omega)

    def _align_inputs(
        self,
        w_mkt: pd.Series,
        cov_matrix: pd.DataFrame,
        P: Optional[pd.DataFrame],
        Q: Optional[pd.Series],
        Omega: Optional[pd.DataFrame],
    ) -> tuple[
        list[str],
        pd.DataFrame,
        pd.Series,
        Optional[pd.DataFrame],
        Optional[pd.Series],
        Optional[pd.DataFrame],
    ]:
        """Aligns all input matrices and series to a common asset universe.

        Ensures that all pandas objects (market weights, covariance, and view
        matrices) share the same asset order and removes assets that are not
        present across all inputs.

        Args:
            w_mkt: A series of market-cap weights for each asset.
            cov_matrix: The covariance matrix of asset returns.
            P: The view matrix, linking assets to views.
            Q: The view vector, containing the expected returns for each view.
            Omega: The covariance matrix of view uncertainties.

        Returns:
            A tuple containing the aligned inputs:
            - A list of common asset tickers.
            - The aligned covariance matrix.
            - The aligned and re-normalized market weights.
            - The aligned P, Q, and Omega matrices, or None if no views
              were provided.

        """
        asset_universe = cov_matrix.index.intersection(w_mkt.index).sort_values()
        if asset_universe.empty:
            raise ValueError("No common tickers between market weights and covariance matrix.")

        S = cov_matrix.loc[asset_universe, asset_universe]
        w = w_mkt.loc[asset_universe]
        w = w / w.sum()

        if P is None or Q is None or Omega is None or P.empty:
            return asset_universe.tolist(), S, w, None, None, None

        P_aligned_cols = P.reindex(columns=asset_universe, fill_value=0.0)
        common_views = P_aligned_cols.index.intersection(Q.index).intersection(Omega.index)

        P_final = P_aligned_cols.loc[common_views]
        Q_final = Q.loc[common_views]
        Omega_final = Omega.loc[common_views, common_views]

        if P_final.empty:
            return asset_universe.tolist(), S, w, None, None, None

        return asset_universe.tolist(), S, w, P_final, Q_final, Omega_final

    def get_implied_equilibrium_returns(self) -> pd.Series:
        """Calculate the implied equilibrium returns based on the market weights and covariance matrix.

        This method uses the formula:
        pi = delta * S * w_mkt
        where:
            - pi is the implied equilibrium returns.
            - delta is the risk aversion coefficient.
            - S is the covariance matrix of asset returns.
            - w_mkt is the market weights of the assets.


        Returns:
            pd.Series: Implied equilibrium returns, indexed by tickers.

        """
        pi = self.delta * self.aligned_cov_matrix.dot(self.aligned_w_mkt)
        return pi

    def get_posterior_returns(self) -> Optional[pd.Series]:
        """Calculate the posterior returns using the Black-Litterman formula.

        This method computes the posterior returns based on the Black-Litterman model,
        which combines the implied equilibrium returns with the views provided.
        The formula used is:
        E[R] = M_inv * ( (tau*S)^-1 * Pi + P' * Omega^-1 * Q )

        where:
            - M_inv is the inverse of the posterior covariance matrix.
            - tau is a scalar that scales the covariance matrix.
            - S is the covariance matrix of asset returns.
            - Pi is the implied equilibrium returns.
            - P is the matrix of views.
            - Omega is the diagonal covariance matrix of the views.
            - Q is the vector of views.
        If no views are provided (i.e., P, Q, or Omega is None), this method returns None.
        If the views are provided, it calculates the posterior returns using the Black-Litterman
        formula.
        If the Omega matrix is singular or the posterior covariance matrix M is singular, it will
        print a warning and return None.


        Returns:
            Optional[pd.Series]: Posterior returns, indexed by tickers, or None if views are not
            provided.

        """
        pi_series = self.get_implied_equilibrium_returns()

        if self.aligned_P is None or self.aligned_Q is None or self.aligned_Omega is None:
            return None

        # --- Convert all pandas objects to numpy arrays for calculation ---
        P_vals = self.aligned_P.values
        Q_vals = self.aligned_Q.values
        S_vals = self.aligned_cov_matrix.values
        Omega_vals = self.aligned_Omega.values
        pi_vals = pi_series.values

        # --- Core Black-Litterman formulas using only numpy arrays ---
        # M_inv = ( (tau*S)^-1 + P' * Omega^-1 * P )^-1
        tau_S_inv = np.linalg.inv(self.tau * S_vals)

        try:
            Omega_inv = np.linalg.inv(Omega_vals)
        except np.linalg.LinAlgError:
            print("Warning: Omega matrix is singular. Skipping views.")
            return None

        # This is the term P' * Omega^-1 * P
        P_T_Omega_inv_P = P_vals.T @ Omega_inv @ P_vals

        # This is the full term to invert for M
        M = tau_S_inv + P_T_Omega_inv_P
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("Warning: Posterior covariance matrix M is singular. Skipping views.")
            return None

        # E[R] = M_inv * ( (tau*S)^-1 * Pi + P' * Omega^-1 * Q )
        # This is the term P' * Omega^-1 * Q
        P_T_Omega_inv_Q = P_vals.T @ Omega_inv @ Q_vals

        # This is the full term to be multiplied by M_inv
        rhs_term = (tau_S_inv @ pi_vals) + P_T_Omega_inv_Q

        posterior_pi_vals = M_inv @ rhs_term

        return pd.Series(posterior_pi_vals, index=self.tickers)
