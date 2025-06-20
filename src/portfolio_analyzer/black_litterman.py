from typing import Optional

import numpy as np
import pandas as pd


class BlackLittermanModel:
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
    ) -> tuple:
        """Robustly aligns all inputs to a common set of tickers and views."""
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
        """Calculates the implied equilibrium returns (pi)."""
        pi = self.delta * self.aligned_cov_matrix.dot(self.aligned_w_mkt)
        return pi

    def get_posterior_returns(self) -> Optional[pd.Series]:
        """Calculates the posterior returns using a numerically stable approach.
        Converts all pandas objects to numpy arrays before calculation to avoid
        implicit alignment errors.
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
