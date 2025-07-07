"""Prepares model inputs for portfolio optimization and analysis.

This module contains the core data pipeline functions. It fetches raw price
and financial data, processes it into log returns, calculates covariance
matrices, and optionally applies the Black-Litterman model to generate the
final inputs (mean returns, covariance) for the optimizer.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.covariance import LedoitWolf

from ..config.config import AppConfig
from ..data.data_fetcher import DataFetcher
from ..data.models import ModelInputs
from ..return_estimator.black_litterman_return import BlackLittermanReturn
from ..return_estimator.blended_return import BlendedReturn
from ..return_estimator.capm_return_estimator import CAPMReturnEstimator
from ..return_estimator.ewma_return import EWMAReturn
from ..utils.exceptions import DataFetchingError
from ..utils.util import calculate_log_returns

logger = logging.getLogger(__name__)


def _calculate_annualized_covariance(
    log_returns: pd.DataFrame, trading_days: int = 252
) -> pd.DataFrame:
    """Calculate the annualized covariance matrix using Ledoit-Wolf shrinkage."""
    lw = LedoitWolf()
    lw.fit(log_returns)
    return pd.DataFrame(
        data=lw.covariance_ * trading_days,
        index=log_returns.columns,
        columns=log_returns.columns,
    )


def _apply_black_litterman_model(
    log_returns: pd.DataFrame,
    risk_free_rate: float,
    risk_aversion: float,
    covariance_matrix: pd.DataFrame,
    market_cap_weights: pd.Series,
    tau: float,
    dcf_views: pd.Series = None,
    config: AppConfig | None = None,
) -> Optional[BlackLittermanReturn]:
    if market_cap_weights is None or market_cap_weights.empty:
        return None
    if dcf_views is None or dcf_views.empty:
        return None

    log_returns = log_returns.sort_index(axis=1)
    covariance_matrix = covariance_matrix.loc[log_returns.columns, log_returns.columns]
    market_cap_weights = market_cap_weights.reindex(log_returns.columns).fillna(0.0)

    view_vector = dcf_views.copy()

    return BlackLittermanReturn(
        log_returns=log_returns,
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
        market_cap_weights=market_cap_weights,
        tau=tau,
        view_vector=view_vector,
        config=config,
    )


def build_model_inputs(
    log_returns: pd.DataFrame,
    config: AppConfig,
    market_cap_weights: Optional[pd.Series] = None,
    dcf_views: Optional[pd.Series] = None,
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
        dcf_views (Optional[pd.Series]): Views on asset returns, typically
            from a DCF model. Required for Black-Litterman.

    Returns:
        Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
            A tuple containing the final mean returns, covariance matrix,
            historical mean returns, implied equilibrium returns, and a list of
            final tickers.

    """
    ewma = EWMAReturn(
        log_returns=log_returns,
        span=config.ewma_span,
        trading_days=config.trading_days_per_year,
        shrinkage_factor=config.mean_shrinkage_alpha,
    )
    hist_mean_returns = ewma.get_shrinked_ewma_returns()
    cov_matrix_annualized = _calculate_annualized_covariance(
        log_returns, config.trading_days_per_year
    )
    bl_model = _apply_black_litterman_model(
        log_returns=log_returns,
        risk_free_rate=config.risk_free_rate,
        risk_aversion=config.black_litterman.delta,
        covariance_matrix=cov_matrix_annualized,
        market_cap_weights=market_cap_weights,
        tau=config.black_litterman.tau,
        dcf_views=dcf_views,
        config=config,
    )
    implied_equilibrium_returns = bl_model.get_implied_equilibrium_returns() if bl_model else None

    blended_returns = BlendedReturn(
        [
            (bl_model, 1 - config.black_litterman.momentum_blend_weight) if bl_model else None,
            (ewma, config.black_litterman.momentum_blend_weight),
        ],
    )
    final_mean_returns = blended_returns.get_returns()
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


def prepare_model_inputs(config: AppConfig, data_fetcher: DataFetcher) -> ModelInputs:
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

    # Fetch data
    market_cap_series = data_fetcher.fetch_market_caps(config.tickers)
    try:
        close_df = data_fetcher.fetch_price_data(
            tickers=config.tickers,
            start_date=config.date_range.start.strftime("%Y-%m-%d"),
            end_date=config.date_range.end.strftime("%Y-%m-%d"),
        )
    except ValueError as e:
        logger.error("Data fetching failed with a ValueError, cannot proceed.", exc_info=True)
        raise DataFetchingError("Failed to fetch price data.") from e
    final_tickers_list = close_df.columns.tolist()
    logger.info("Proceeding with %d tickers that have valid price data.", len(final_tickers_list))

    # Market cap weights
    w_mkt = pd.Series(dtype=float)
    if not market_cap_series.empty:
        w_mkt_filtered = market_cap_series.reindex(final_tickers_list).dropna()
        if not w_mkt_filtered.empty:
            w_mkt = w_mkt_filtered / w_mkt_filtered.sum()
            logger.info("Successfully normalized market cap weights for %d tickers.", len(w_mkt))
        else:
            logger.warning("No market cap data available for the filtered tickers.")

    # DCF views (Via CAPM estimator)
    capm_return_estimator = CAPMReturnEstimator(
        config=config,
        data_fetcher=data_fetcher,
    )
    dcf_views = capm_return_estimator.get_returns()
    log_returns = calculate_log_returns(close_df)

    # Build model inputs
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

    # Assemble ModelInputs dataclass
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
