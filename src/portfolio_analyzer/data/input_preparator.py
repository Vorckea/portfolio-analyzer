"""Prepares model inputs for portfolio optimization and analysis.

This module contains the core data pipeline functions. It fetches raw price
and financial data, processes it into log returns, calculates covariance
matrices, and optionally applies the Black-Litterman model to generate the
final inputs (mean returns, covariance) for the optimizer.
"""

import logging

from ..config.config import AppConfig
from ..return_estimator.base import ReturnEstimator
from ..utils.util import calculate_annualized_covariance, calculate_log_returns
from .new_models import AssetUniverse, ModelInputs, PriceData, ReturnEstimates
from .repository import Repository

logger = logging.getLogger(__name__)


def prepare_model_inputs(
    config: AppConfig,
    returns: ReturnEstimator,
    repository: Repository,
    start_date: str,
    end_date: str,
    tickers: list[str],
) -> ModelInputs:
    if tickers is None or not tickers:
        logger.error("No tickers provided for preparing model inputs.")
        raise ValueError("No tickers provided for preparing model inputs.")

    logger.info(
        "--- Starting Data Pipeline for %d tickers from %s to %s ---",
        len(tickers),
        start_date,
        end_date,
    )

    mean_returns = returns.get_returns()
    price_df = repository.fetch_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    log_returns = calculate_log_returns(price_df)
    cov_matrix = calculate_annualized_covariance(
        log_returns=log_returns,
        trading_days=config.trading_days_per_year,
    )

    final_tickers = mean_returns.index.intersection(cov_matrix.index).sort_values()
    """model_inputs = ModelInputs(
        mean_returns=mean_returns.loc[final_tickers],
        cov_matrix=cov_matrix.loc[final_tickers, final_tickers],
        log_returns=log_returns[final_tickers],
        close_df=price_df[final_tickers],
        final_tickers=final_tickers.tolist(),
    )"""

    universe = AssetUniverse(tickers=tuple(final_tickers))
    returns = ReturnEstimates(
        expected_returns=mean_returns.loc[final_tickers],
        covariance=cov_matrix.loc[final_tickers, final_tickers],
    )
    prices = PriceData(
        log_returns=log_returns[final_tickers],
        close_prices=price_df[final_tickers],
    )
    model_inputs = ModelInputs(
        universe=universe,
        returns=returns,
        prices=prices,
    )

    logger.info("--- Data Pipeline Finished ---")
    return model_inputs
