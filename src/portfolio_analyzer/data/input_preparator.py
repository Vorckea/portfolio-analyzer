"""Prepares model inputs for portfolio optimization and analysis."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.core.black_litterman import BlackLittermanModel
from portfolio_analyzer.data.data_fetcher import (
    calculate_dcf_views,
    fetch_market_caps,
    fetch_price_data,
)

logger = logging.getLogger(__name__)


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
    logger.info(
        "--- Starting Data Pipeline for %d tickers from %s to %s ---",
        len(config.tickers),
        config.date_range.start.strftime("%Y-%m-%d"),
        config.date_range.end.strftime("%Y-%m-%d"),
    )
    logger.info("DCF Views Enabled: %s", config.use_dcf_views)

    # 1. Fetch all raw data first
    market_cap_series = fetch_market_caps(config.tickers)
    try:
        close_df = fetch_price_data(
            tickers=config.tickers,
            start_date=config.date_range.start.strftime("%Y-%m-%d"),
            end_date=config.date_range.end.strftime("%Y-%m-%d"),
        )
        logger.debug("Fetched price data shape: %s", close_df.shape)
    except ValueError as e:
        logger.error("Data fetching failed with a ValueError, cannot proceed.", exc_info=True)
        raise

    # 2. **CRITICAL**: Define the final list of tickers based on successful price data.
    final_tickers_list = close_df.columns.tolist()
    logger.info("Proceeding with %d tickers that have valid price data.", len(final_tickers_list))

    # 3. Filter and re-normalize market cap weights to match the final tickers
    w_mkt = pd.Series(dtype=float)
    if not market_cap_series.empty:
        original_mkt_cap_tickers = set(market_cap_series.index)
        w_mkt_filtered = market_cap_series.reindex(final_tickers_list).dropna()
        if not w_mkt_filtered.empty:
            w_mkt = w_mkt_filtered / w_mkt_filtered.sum()
            logger.info(
                "Successfully filtered and normalized market cap weights for %d tickers.",
                len(w_mkt),
            )
            dropped_mkt_cap_tickers = original_mkt_cap_tickers - set(w_mkt.index)
            if dropped_mkt_cap_tickers:
                logger.debug(
                    "Tickers dropped due to missing market cap: %s", dropped_mkt_cap_tickers
                )
        else:
            logger.warning("No market cap data available for the filtered tickers.")

    # 4. Calculate DCF views and log returns
    dcf_views = calculate_dcf_views(config) if config.use_dcf_views else {}
    log_returns = _calculate_log_returns(close_df)
    logger.debug("Calculated log returns shape: %s", log_returns.shape)

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
    logger.info("--- Data Pipeline Finished ---")
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

    logger.info("Calculating annualized covariance matrix using Ledoit-Wolf shrinkage...")
    lw = LedoitWolf()
    lw.fit(log_returns)
    cov_matrix_annualized = pd.DataFrame(
        lw.covariance_ * config.trading_days_per_year,
        index=log_returns.columns,
        columns=log_returns.columns,
    )
    logger.debug("Calculated annualized covariance matrix shape: %s", cov_matrix_annualized.shape)

    final_mean_returns = hist_mean_returns.copy()
    implied_equilibrium_returns = None
    P_views, Q_views, Omega_df = None, None, None

    if dcf_views:
        logger.info("Constructing Black-Litterman views from DCF estimates...")
        view_tickers = list(dcf_views.keys())
        aligned_view_tickers = [t for t in view_tickers if t in log_returns.columns]
        if aligned_view_tickers:
            Q_views = pd.Series({t: dcf_views[t] for t in aligned_view_tickers})
            P_views = pd.DataFrame(0.0, index=Q_views.index, columns=log_returns.columns)
            for ticker in Q_views.index:
                P_views.loc[ticker, ticker] = 1.0
            view_cov = P_views @ cov_matrix_annualized @ P_views.T
            omega_diag_views = np.diag(view_cov) * config.black_litterman.tau
            # **FIX**: Create Omega as a labeled DataFrame
            Omega_df = pd.DataFrame(
                np.diag(omega_diag_views), index=Q_views.index, columns=Q_views.index
            )
            logger.info("Successfully constructed %d views.", len(Q_views))

    if w_mkt is not None and not w_mkt.empty:
        logger.info("Attempting to run Black-Litterman model...")
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
                logger.info("Successfully applied Black-Litterman model with views.")
                final_mean_returns = posterior_returns
            else:
                logger.warning(
                    "No valid views; blending historical and implied equilibrium returns."
                )
                w_blend = config.black_litterman.equilibrium_blend_weight
                common_idx = hist_mean_returns.index.intersection(implied_equilibrium_returns.index)
                final_mean_returns = (
                    w_blend * implied_equilibrium_returns.loc[common_idx]
                    + (1 - w_blend) * hist_mean_returns.loc[common_idx]
                )

            assets_no_mkt_cap = w_mkt[w_mkt <= 1e-9].index
            common_no_cap = assets_no_mkt_cap.intersection(final_mean_returns.index)
            if not common_no_cap.empty:
                logger.info(
                    "Updating returns for %d assets without market cap to use implied equilibrium "
                    "returns.",
                    len(common_no_cap),
                )
                final_mean_returns.update(implied_equilibrium_returns.loc[common_no_cap])
        except Exception as e:
            logger.exception("Black-Litterman model failed. Reverting to historical returns.")
            final_mean_returns = hist_mean_returns
    else:
        logger.info(
            "No market cap weights provided; using historical returns as the final estimate."
        )
        final_mean_returns = hist_mean_returns

    # **IMPROVEMENT**: Blend the final BL/Equilibrium returns with historical momentum.
    if config.black_litterman.momentum_blend_weight > 0 and implied_equilibrium_returns is not None:
        logger.info(
            f"Blending final returns with {config.black_litterman.momentum_blend_weight:.0%} "
            f"momentum."
        )
        # Ensure both series are aligned before blending
        common_idx = final_mean_returns.index.intersection(hist_mean_returns.index)
        final_mean_returns_aligned = final_mean_returns.loc[common_idx]
        hist_mean_returns_aligned = hist_mean_returns.loc[common_idx]

        final_mean_returns = (
            (1 - config.black_litterman.momentum_blend_weight) * final_mean_returns_aligned
            + config.black_litterman.momentum_blend_weight * hist_mean_returns_aligned
        )

    # Final alignment
    logger.debug(
        "Pre-alignment tickers: mean_returns=%d, cov_matrix=%d",
        len(final_mean_returns),
        len(cov_matrix_annualized),
    )
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

    logger.info("Input preparation complete. Using %d tickers.", len(common_tickers))
    return (
        final_mean_returns,
        final_cov_matrix,
        final_hist_returns,
        final_implied_returns,
    )
