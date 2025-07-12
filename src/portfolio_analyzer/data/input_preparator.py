"""Prepares model inputs for portfolio optimization and analysis.

This module contains the core data pipeline functions. It fetches raw price
and financial data, processes it into log returns, calculates covariance
matrices, and optionally applies the Black-Litterman model to generate the
final inputs (mean returns, covariance) for the optimizer.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

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
    data_fetcher: DataFetcher,
    dcf_views: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
    """Assamble the final model inputs from processed log returns and views.

    Args:
        log_returns (pd.DataFrame): DataFrame containing log returns for each ticker.
        config (AppConfig): Application configuration object containing settings.
        data_fetcher (DataFetcher): An instance of DataFetcher to retrieve price data.
        dcf_views (Optional[pd.Series], optional): Optional DCF views to blend with the returns.
            Defaults to None.

    Returns:
        Tuple[pd.Series, pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]: Tuple of
            (mean_returns, cov_matrix, hist_mean_returns, implied_equilibrium_returns,
            final_tickers).

    """
    ewma = EWMAReturn(data_fetcher=data_fetcher, config=config)

    bl_ewma = BlackLittermanReturn(
        view_vector=ewma.get_returns(), config=config, data_fetcher=data_fetcher
    )

    bl_model = BlackLittermanReturn(
        view_vector=dcf_views,
        config=config,
        data_fetcher=data_fetcher,
    )

    blended_returns = BlendedReturn(
        [
            (bl_model, 1 - config.black_litterman.momentum_blend_weight) if bl_model else None,
            (bl_ewma, config.black_litterman.momentum_blend_weight),
        ]
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
    """Orchestrates the data preparation pipeline.

    This function serves as the entry point for preparing model inputs for portfolio optimization.

    Args:
        config (AppConfig): The application configuration object containing settings.
        data_fetcher (DataFetcher): An instance of DataFetcher to retrieve price data.

    Raises:
        DataFetchingError: If data fetching fails, this error is raised to indicate
            that the pipeline cannot proceed.

    Returns:
        ModelInputs: A dataclass containing all necessary inputs for the portfolio optimization model,
        including mean returns, covariance matrix, log returns, and final tickers.

    """
    logger.info(
        "--- Starting Data Pipeline for %d tickers from %s to %s ---",
        len(config.tickers),
        config.date_range.start.strftime("%Y-%m-%d"),
        config.date_range.end.strftime("%Y-%m-%d"),
    )
    logger.info("DCF Views Enabled: %s", config.use_dcf_views)

    try:
        close_df = data_fetcher.fetch_price_data(
            tickers=config.tickers,
            start_date=config.date_range.start.strftime("%Y-%m-%d"),
            end_date=config.date_range.end.strftime("%Y-%m-%d"),
        )
    except ValueError as e:
        logger.error("Data fetching failed with a ValueError, cannot proceed.", exc_info=True)
        raise DataFetchingError("Failed to fetch price data.") from e

    final_tickers = close_df.columns.tolist()
    logger.info("Proceeding with %d tickers that have valid price data.", len(final_tickers))

    capm_return_estimator = CAPMReturnEstimator(
        config=config,
        data_fetcher=data_fetcher,
    )
    views = capm_return_estimator.get_returns()
    log_returns = calculate_log_returns(close_df)

    (
        mean_returns,
        cov_matrix,
        hist_mean_returns,
        implied_equilibrium_returns,
        final_tickers,
    ) = build_model_inputs(
        log_returns=log_returns,
        config=config,
        data_fetcher=data_fetcher,
        dcf_views=views,
    )

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
