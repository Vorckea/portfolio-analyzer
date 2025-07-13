"""Prepares model inputs for portfolio optimization and analysis.

This module contains the core data pipeline functions. It fetches raw price
and financial data, processes it into log returns, calculates covariance
matrices, and optionally applies the Black-Litterman model to generate the
final inputs (mean returns, covariance) for the optimizer.
"""

import logging

from ..config.config import AppConfig
from ..data.data_fetcher import DataFetcher
from ..data.models import ModelInputs
from ..return_estimator.return_estimator import ReturnEstimator
from ..utils.util import calculate_annualized_covariance, calculate_log_returns

logger = logging.getLogger(__name__)


def prepare_model_inputs(
    config: AppConfig, returns: ReturnEstimator, data_fetcher: DataFetcher
) -> ModelInputs:
    logger.info(
        "--- Starting Data Pipeline for %d tickers from %s to %s ---",
        len(config.tickers),
        config.date_range.start.strftime("%Y-%m-%d"),
        config.date_range.end.strftime("%Y-%m-%d"),
    )

    mean_returns = returns.get_returns()
    price_df = data_fetcher.fetch_price_data(
        tickers=config.tickers,
        start_date=config.date_range.start,
        end_date=config.date_range.end,
    )
    log_returns = calculate_log_returns(price_df)
    cov_matrix = calculate_annualized_covariance(
        log_returns=log_returns,
        trading_days=config.trading_days_per_year,
    )
    final_tickers = sorted(list(mean_returns.index.intersection(cov_matrix.index)))

    model_inputs = ModelInputs(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        log_returns=log_returns.reindex(columns=final_tickers),
        close_df=price_df.reindex(columns=final_tickers),
        final_tickers=final_tickers,
    )
    logger.info("--- Data Pipeline Finished ---")
    return model_inputs
