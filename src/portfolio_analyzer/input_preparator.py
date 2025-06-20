from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_analyzer.black_litterman import BlackLittermanModel
from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.data_fetcher import (
    calculate_dcf_views,
    fetch_market_caps,
    fetch_price_data,
)


@dataclass
class ModelInputs:
    """Holds all the data required for the optimization and analysis steps."""

    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    log_returns: pd.DataFrame
    close_df: pd.DataFrame
    final_tickers: list[str]
    w_mkt: pd.Series
    hist_mean_returns: Optional[pd.Series] = None
    implied_equilibrium_returns: Optional[pd.Series] = None


def prepare_model_inputs(config: AppConfig) -> ModelInputs:
    """Orchestrates the entire data preparation pipeline with robust filtering."""
    print("--- Starting Data Pipeline ---")

    # 1. Fetch all raw data first
    market_cap_series = fetch_market_caps(config.tickers)
    try:
        close_df = fetch_price_data(
            tickers=config.tickers,
            start_date=config.date_range.start.strftime("%Y-%m-%d"),
            end_date=config.date_range.end.strftime("%Y-%m-%d"),
        )
    except ValueError as e:
        print(f"Error during data fetching: {e}")
        raise

    # 2. **CRITICAL**: Define the final list of tickers based on successful price data.
    final_tickers_list = close_df.columns.tolist()
    print(
        f"\nProceeding with {len(final_tickers_list)} tickers that have valid price data."
    )

    # 3. Filter and re-normalize market cap weights to match the final tickers
    w_mkt = pd.Series(dtype=float)
    if not market_cap_series.empty:
        w_mkt_filtered = market_cap_series.reindex(final_tickers_list).dropna()
        if not w_mkt_filtered.empty:
            w_mkt = w_mkt_filtered / w_mkt_filtered.sum()
            print("Successfully filtered and normalized market cap weights.")

    # 4. Calculate DCF views and log returns
    dcf_views = calculate_dcf_views(config) if config.use_dcf_views else {}
    log_returns = _calculate_log_returns(close_df)

    # 5. Prepare final model inputs, passing the now-consistent data
    (
        mean_returns,
        cov_matrix,
        hist_returns,
        implied_returns,
    ) = _prepare_optimization_inputs(
        log_returns=log_returns, config=config, w_mkt=w_mkt, dcf_views=dcf_views
    )

    # 6. Final packaging of the model inputs
    final_tickers = mean_returns.index.tolist()
    print("\n--- Data Pipeline Finished ---")
    return ModelInputs(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        log_returns=log_returns.reindex(columns=final_tickers),
        close_df=close_df.reindex(columns=final_tickers),
        final_tickers=final_tickers,
        w_mkt=w_mkt.reindex(final_tickers).fillna(0),
        hist_mean_returns=hist_returns,
        implied_equilibrium_returns=implied_returns,
    )


def _calculate_log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily log returns from price data."""
    return np.log(close_df / close_df.shift(1)).dropna()


def _ewma_shrunk_returns(
    log_returns: pd.DataFrame, span: int, alpha: float, trading_days: int
) -> pd.Series:
    """Calculates annualized EWMA returns and applies shrinkage."""
    ewma = log_returns.ewm(span=span).mean().iloc[-1] * trading_days
    return _shrink_mean(ewma, alpha)


def _shrink_mean(returns: pd.Series, alpha: float) -> pd.Series:
    """Shrinks returns towards the grand mean."""
    grand_mean = returns.mean()
    return (1 - alpha) * returns + alpha * grand_mean


def _prepare_optimization_inputs(
    log_returns: pd.DataFrame,
    config: AppConfig,
    w_mkt: Optional[pd.Series] = None,
    dcf_views: Optional[dict] = None,
) -> tuple[pd.Series, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Internal function to create final mean vector and covariance matrix."""
    hist_mean_returns = _ewma_shrunk_returns(
        log_returns,
        config.ewma_span,
        config.mean_shrinkage_alpha,
        config.trading_days_per_year,
    )

    lw = LedoitWolf()
    lw.fit(log_returns)
    cov_matrix_annualized = pd.DataFrame(
        lw.covariance_ * config.trading_days_per_year,
        index=log_returns.columns,
        columns=log_returns.columns,
    )

    final_mean_returns = hist_mean_returns.copy()
    implied_equilibrium_returns = None
    P_views, Q_views, Omega_df = None, None, None

    if dcf_views:
        print("\nConstructing Black-Litterman views from DCF estimates...")
        view_tickers = list(dcf_views.keys())
        aligned_view_tickers = [t for t in view_tickers if t in log_returns.columns]
        if aligned_view_tickers:
            Q_views = pd.Series({t: dcf_views[t] for t in aligned_view_tickers})
            P_views = pd.DataFrame(
                0.0, index=Q_views.index, columns=log_returns.columns
            )
            for ticker in Q_views.index:
                P_views.loc[ticker, ticker] = 1.0
            view_cov = P_views @ cov_matrix_annualized @ P_views.T
            omega_diag_views = np.diag(view_cov) * config.black_litterman.tau
            # **FIX**: Create Omega as a labeled DataFrame
            Omega_df = pd.DataFrame(
                np.diag(omega_diag_views), index=Q_views.index, columns=Q_views.index
            )
            print(f"Successfully constructed {len(Q_views)} views.")

    if w_mkt is not None and not w_mkt.empty:
        print("\nAttempting to run Black-Litterman model...")
        try:
            bl_model = BlackLittermanModel(
                tau=config.black_litterman.tau,
                delta=config.black_litterman.delta,
                w_mkt=w_mkt,
                cov_matrix=cov_matrix_annualized,
                P=P_views,
                Q=Q_views,
                Omega=Omega_df,
            )
            posterior_returns = bl_model.get_posterior_returns()
            implied_equilibrium_returns = bl_model.get_implied_equilibrium_returns()

            if posterior_returns is not None:
                print("Successfully applied Black-Litterman model with views.")
                final_mean_returns = posterior_returns
            else:
                print(
                    "No valid views; blending historical and implied equilibrium returns."
                )
                w_blend = config.black_litterman.equilibrium_blend_weight
                common_idx = hist_mean_returns.index.intersection(
                    implied_equilibrium_returns.index
                )
                final_mean_returns = (
                    w_blend * implied_equilibrium_returns.loc[common_idx]
                    + (1 - w_blend) * hist_mean_returns.loc[common_idx]
                )

            assets_no_mkt_cap = w_mkt[w_mkt <= 1e-9].index
            common_no_cap = assets_no_mkt_cap.intersection(final_mean_returns.index)
            if not common_no_cap.empty:
                final_mean_returns.update(
                    implied_equilibrium_returns.loc[common_no_cap]
                )
        except Exception as e:
            print(f"Black-Litterman model failed: {e}. Using historical returns.")
            final_mean_returns = hist_mean_returns
    else:
        final_mean_returns = hist_mean_returns

    # **IMPROVEMENT**: Blend the final BL/Equilibrium returns with historical momentum.
    if (
        config.black_litterman.momentum_blend_weight > 0
        and implied_equilibrium_returns is not None
    ):
        print(
            f"\nBlending final returns with {config.black_litterman.momentum_blend_weight:.0%} momentum."
        )
        # Ensure both series are aligned before blending
        common_idx = final_mean_returns.index.intersection(hist_mean_returns.index)
        final_mean_returns_aligned = final_mean_returns.loc[common_idx]
        hist_mean_returns_aligned = hist_mean_returns.loc[common_idx]

        final_mean_returns = (
            (1 - config.black_litterman.momentum_blend_weight)
            * final_mean_returns_aligned
            + config.black_litterman.momentum_blend_weight * hist_mean_returns_aligned
        )

    # Final alignment
    common_tickers = sorted(
        list(final_mean_returns.index.intersection(cov_matrix_annualized.index))
    )
    final_mean_returns = final_mean_returns.loc[common_tickers]
    final_cov_matrix = cov_matrix_annualized.loc[common_tickers, common_tickers]
    final_hist_returns = hist_mean_returns.loc[common_tickers]
    final_implied_returns = (
        implied_equilibrium_returns.loc[common_tickers]
        if implied_equilibrium_returns is not None
        else None
    )

    print(f"\nInput preparation complete. Using {len(common_tickers)} tickers.")
    return (
        final_mean_returns,
        final_cov_matrix,
        final_hist_returns,
        final_implied_returns,
    )
