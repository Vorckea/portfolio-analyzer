"""Prepares model inputs for portfolio optimization and analysis.

This module contains the core data pipeline functions. It fetches raw price
and financial data, processes it into log returns, calculates covariance
matrices, and optionally applies the Black-Litterman model to generate the
final inputs (mean returns, covariance) for the optimizer.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

from ..config.config import AppConfig
from ..data.data_fetcher import DataFetcher
from ..data.models import ModelInputs
from ..return_estimator.black_litterman_return import BlackLittermanReturn
from ..return_estimator.blended_return import BlendedReturn
from ..return_estimator.capm_return_estimator import CAPMReturnEstimator
from ..return_estimator.ewma_return import EWMAReturn
from ..utils.exceptions import DataFetchingError
from ..utils.util import calculate_annualized_covariance, calculate_log_returns

logger = logging.getLogger(__name__)


def build_model_inputs(
    log_returns: pd.DataFrame,
    config: AppConfig,
    dcf_views: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
    """Build the final model inputs from processed log returns and views.

    This is a core component of the data pipeline that takes log returns and DCF views to produce the final mean
    return vector and covariance matrix for optimization.

    Args:
        log_returns (pd.DataFrame): DataFrame of daily log returns for assets.
        config (AppConfig): The application configuration object.
        dcf_views (Optional[pd.Series]): Views on asset returns, typically
            from a DCF model. Required for Black-Litterman.

    Returns:
        Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
            A tuple containing the final mean returns, covariance matrix,
            historical mean returns, implied equilibrium returns, and a list of
            final tickers.

    """
    ewma = EWMAReturn(
        data_fetcher=DataFetcher(yf),
        config=config,
    )

    bl_model = BlackLittermanReturn(
        view_vector=dcf_views,
        config=config,
        data_fetcher=DataFetcher(yf),
    )

    blended_returns = BlendedReturn(
        [
            (bl_model, 1 - config.black_litterman.momentum_blend_weight) if bl_model else None,
            (ewma, config.black_litterman.momentum_blend_weight),
        ],
    )

    hist_mean_returns = ewma.get_returns()
    cov_matrix_annualized = calculate_annualized_covariance(
        log_returns, config.trading_days_per_year
    )
    implied_equilibrium_returns = bl_model.get_implied_equilibrium_returns() if bl_model else None
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
        data_fetcher (DataFetcher): An instance of DataFetcher to retrieve data.

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
        dcf_views=dcf_views,
    )

    # Assemble ModelInputs dataclass
    model_inputs = ModelInputs(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        log_returns=log_returns.reindex(columns=final_tickers),
        close_df=close_df.reindex(columns=final_tickers),
        final_tickers=final_tickers,
        hist_mean_returns=hist_mean_returns,
        implied_equilibrium_returns=implied_equilibrium_returns,
    )
    logger.info("--- Data Pipeline Finished ---")
    return model_inputs
