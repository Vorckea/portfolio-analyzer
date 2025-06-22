"""Prepares model inputs for portfolio optimization and analysis.

This module contains the core data pipeline functions. It fetches raw price
and financial data, processes it into log returns, calculates covariance
matrices, and optionally applies the Black-Litterman model to generate the
final inputs (mean returns, covariance) for the optimizer.
"""

import logging
from typing import Dict, List, Optional, Tuple

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
from portfolio_analyzer.data.models import ModelInputs
from portfolio_analyzer.utils.exceptions import DataFetchingError

logger = logging.getLogger(__name__)


def _calculate_annualized_covariance(
    log_returns: pd.DataFrame, trading_days: int = 252
) -> pd.DataFrame:
    """Calculate the annualized covariance matrix using Ledoit-Wolf shrinkage."""
    lw = LedoitWolf()
    lw.fit(log_returns)
    cov_matrix = pd.DataFrame(
        lw.covariance_ * trading_days,
        index=log_returns.columns,
        columns=log_returns.columns,
    )
    logger.debug("Calculated annualized covariance matrix shape: %s", cov_matrix.shape)
    return cov_matrix


def _apply_black_litterman_model(
    hist_mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    market_cap_weights: pd.Series,
    config: AppConfig,
    dcf_views: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """Apply the Black-Litterman model to blend historical and view-based returns."""
    if market_cap_weights is not None and not market_cap_weights.empty:
        logger.info("Attempting to run Black-Litterman model...")
        try:
            P_views, Q_views, Omega_df = None, None, None
            if dcf_views:
                logger.info("Constructing Black-Litterman views from DCF estimates...")
                view_tickers = list(dcf_views.keys())
                aligned_view_tickers = [t for t in view_tickers if t in hist_mean_returns.index]
                if aligned_view_tickers:
                    Q_views = pd.Series({t: dcf_views[t] for t in aligned_view_tickers})
                    P_views = pd.DataFrame(
                        0.0, index=Q_views.index, columns=hist_mean_returns.index
                    )
                    for ticker in Q_views.index:
                        P_views.loc[ticker, ticker] = 1.0
                    view_cov = P_views @ cov_matrix @ P_views.T
                    omega_diag_views = np.diag(view_cov) * config.black_litterman.tau
                    Omega_df = pd.DataFrame(
                        np.diag(omega_diag_views), index=Q_views.index, columns=Q_views.index
                    )
                    logger.info("Successfully constructed %d views.", len(Q_views))

            bl_model = BlackLittermanModel(
                tau=config.black_litterman.tau,
                delta=config.black_litterman.delta,
                w_mkt=market_cap_weights,
                cov_matrix=cov_matrix,
                P=P_views,
                Q=Q_views,
                Omega=Omega_df,
            )
            posterior_returns = bl_model.get_posterior_returns()
            implied_equilibrium_returns = bl_model.get_implied_equilibrium_returns()

            if posterior_returns is not None:
                logger.info("Successfully applied Black-Litterman model.")
                final_mean_returns = posterior_returns
                return final_mean_returns, implied_equilibrium_returns
            elif implied_equilibrium_returns is not None:
                logger.warning(
                    "BL model did not produce posterior returns. Using implied equilibrium returns."
                )
                final_mean_returns = implied_equilibrium_returns
                return final_mean_returns, implied_equilibrium_returns

        except Exception:
            logger.exception("Black-Litterman model failed. Reverting to historical returns.")
            return hist_mean_returns.copy(), None


def build_model_inputs(
    log_returns: pd.DataFrame,
    config: AppConfig,
    market_cap_weights: Optional[pd.Series] = None,
    dcf_views: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
    """Build the final model inputs from processed log returns and views.

    This is a core component of the data pipeline that takes log returns and
    optional market cap weights and DCF views to produce the final mean
    return vector and covariance matrix for optimization.

    Args:
        log_returns (pd.DataFrame): DataFrame of daily log returns for assets.
        config (AppConfig): The application configuration object.
        market_cap_weights (Optional[pd.Series]): Market cap weights for assets.
            Required for Black-Litterman.
        dcf_views (Optional[Dict[str, float]]): Views on asset returns, typically
            from a DCF model. Required for Black-Litterman.

    Returns:
        Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
            A tuple containing the final mean returns, covariance matrix,
            historical mean returns, implied equilibrium returns, and a list of
            final tickers.

    """
    # 1. Calculate historical mean returns with EWMA and shrinkage
    hist_mean_returns = _ewma_shrunk_returns(
        log_returns,
        span=config.ewma_span,
        alpha=config.mean_shrinkage_alpha,
        trading_days=config.trading_days_per_year,
    )
    logger.debug("Calculated historical mean returns shape: %s", hist_mean_returns.shape)

    # 2. Calculate annualized covariance matrix
    cov_matrix_annualized = _calculate_annualized_covariance(
        log_returns, config.trading_days_per_year
    )

    # 3. Initialize returns and run Black-Litterman model if applicable
    final_mean_returns, implied_equilibrium_returns = _apply_black_litterman_model(
        hist_mean_returns,
        cov_matrix_annualized,
        market_cap_weights,
        config,
        dcf_views,
    )

    # 4. Blend final returns with historical momentum
    if config.black_litterman.momentum_blend_weight > 0:
        logger.info(
            "Blending final returns with %.0f%% momentum.",
            config.black_litterman.momentum_blend_weight * 100,
        )
        common_idx = final_mean_returns.index.intersection(hist_mean_returns.index)
        final_mean_returns_aligned = final_mean_returns.loc[common_idx]
        hist_mean_returns_aligned = hist_mean_returns.loc[common_idx]

        final_mean_returns = (
            (1 - config.black_litterman.momentum_blend_weight) * final_mean_returns_aligned
            + config.black_litterman.momentum_blend_weight * hist_mean_returns_aligned
        )

    # 5. Final alignment and packaging
    common_tickers = sorted(
        list(final_mean_returns.index.intersection(cov_matrix_annualized.index))
    )
    final_mean_returns = final_mean_returns.reindex(common_tickers)
    final_cov_matrix = cov_matrix_annualized.reindex(index=common_tickers, columns=common_tickers)

    logger.info("Core calculations complete. Finalized with %d tickers.", len(common_tickers))

    return (
        final_mean_returns,
        final_cov_matrix,
        hist_mean_returns.reindex(common_tickers),
        implied_equilibrium_returns.reindex(common_tickers)
        if implied_equilibrium_returns is not None
        else None,
        common_tickers,
    )


def prepare_model_inputs(config: AppConfig) -> ModelInputs:
    """Orchestrates the entire data preparation pipeline.

    Fetches prices, calculates returns, and builds all necessary inputs for
    the optimization models, returning them in a structured dataclass.

    Args:
        config (AppConfig): The application configuration object.

    Returns:
        ModelInputs: A dataclass holding all the prepared data.

    """
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
    except ValueError as e:
        logger.error("Data fetching failed with a ValueError, cannot proceed.", exc_info=True)
        raise DataFetchingError("Failed to fetch price data.") from e

    # 2. Define the final list of tickers based on successful price data.
    final_tickers_list = close_df.columns.tolist()
    logger.info("Proceeding with %d tickers that have valid price data.", len(final_tickers_list))

    # 3. Filter and re-normalize market cap weights to match the final tickers
    w_mkt = pd.Series(dtype=float)
    if not market_cap_series.empty:
        w_mkt_filtered = market_cap_series.reindex(final_tickers_list).dropna()
        if not w_mkt_filtered.empty:
            w_mkt = w_mkt_filtered / w_mkt_filtered.sum()
            logger.info("Successfully normalized market cap weights for %d tickers.", len(w_mkt))
        else:
            logger.warning("No market cap data available for the filtered tickers.")

    # 4. Calculate DCF views and log returns
    dcf_views = calculate_dcf_views(config) if config.use_dcf_views else {}
    log_returns = _calculate_log_returns(close_df)

    # 5. Get calculated inputs from the core builder function
    (
        mean_returns,
        cov_matrix,
        hist_mean_returns,
        implied_equilibrium_returns,
        final_tickers,
    ) = build_model_inputs(
        log_returns=log_returns,
        config=config,
        market_cap_weights=w_mkt,
        dcf_views=dcf_views,
    )

    # 6. Assemble the final ModelInputs object with all required data
    model_inputs = ModelInputs(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        log_returns=log_returns.reindex(columns=final_tickers),
        close_df=close_df.reindex(columns=final_tickers),
        w_mkt=w_mkt.reindex(final_tickers).fillna(0),
        final_tickers=final_tickers,
        hist_mean_returns=hist_mean_returns,
        implied_equilibrium_returns=implied_equilibrium_returns,
    )

    logger.info("--- Data Pipeline Finished ---")
    return model_inputs


def _calculate_log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily logarithmic returns from a DataFrame of closing prices.

    Internal helper function.
    """
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
