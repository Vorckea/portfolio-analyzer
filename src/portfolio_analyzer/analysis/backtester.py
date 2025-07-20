"""Backtester class for running historical backtests of portfolio optimization strategies.

This class uses the provided configuration to fetch historical price data,
prepare model inputs, and optimize the portfolio based on the specified strategy.
It supports rebalancing at specified frequencies and can compare the strategy's performance
against a benchmark ticker if provided.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from portfolio_analyzer.analysis.metrics import (
    calculate_performance_summary,
    calculate_relative_metrics,
)
from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.core.optimizer import PortfolioOptimizer
from portfolio_analyzer.data.repository import Repository

from ..data import new_input_preparator as newip
from ..return_estimator import EWMA
from ..utils.util import calculate_log_returns

logger = logging.getLogger(__name__)


class Backtester:
    """Runs historical backtests of portfolio optimization strategies."""

    def __init__(
        self,
        config: AppConfig,
        optimizer_cls: PortfolioOptimizer,
        repository: Repository,
    ):
        """Initialize the Backtester."""
        self.config: AppConfig = config
        self.strategy_name = "Mean-Variance Optimization"
        self.optimizer_cls: PortfolioOptimizer = optimizer_cls
        self.repository: Repository = repository

    def _prepare_inputs_for_date(
        self, log_returns_slice: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Prepare model inputs using a slice of historical log returns.

        This is a simplified version of the logic in `prepare_model_inputs`
        to demonstrate the efficient, in-memory pattern.
        """
        if log_returns_slice.empty:
            return pd.Series(dtype=float), pd.DataFrame()

        start_date = log_returns_slice.index.min().strftime("%Y-%m-%d")
        end_date = log_returns_slice.index.max().strftime("%Y-%m-%d")
        tickers = log_returns_slice.columns.tolist()

        ewma_returns = EWMA(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            repository=self.repository,
            config=self.config,
        )

        model_inputs = newip.prepare_model_inputs(
            config=self.config,
            returns=ewma_returns,
            repository=self.repository,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
        )

        return model_inputs.mean_returns, model_inputs.cov_matrix

    def run(self, benchmark_ticker: str | None = None) -> Tuple[pd.DataFrame, dict]:
        """Run the backtest from the configured start to end date.

        At each rebalancing period, it fetches historical data, prepares model
        inputs, and runs the portfolio optimization. The resulting portfolio
        weights are used to calculate returns for the subsequent period.

        Args:
            benchmark_ticker (Optional[str]): The ticker symbol for a benchmark
                asset (e.g., an index) to compare performance against.

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing:
                - A DataFrame with the portfolio value over time, alongside the
                  benchmark value if provided.
                - A dictionary containing key performance metrics for both the
                  strategy and the benchmark.

        """
        if not isinstance(self.config, AppConfig):
            raise TypeError("config must be an instance of AppConfig")

        logger.info("--- Running Optimized Backtest for '%s' ---", self.strategy_name)

        # 1. Determine the full date range needed, including the lookback period
        lookback_offset = pd.DateOffset(days=self.config.backtesting.lookback_period_days)
        full_start_date = self.config.date_range.start - lookback_offset
        full_end_date = self.config.date_range.end

        all_tickers = self.config.tickers.copy()
        if benchmark_ticker and benchmark_ticker not in all_tickers:
            all_tickers.append(benchmark_ticker)

        # 2. Fetch all price data for the entire period ONCE
        full_price_data = self.repository.fetch_price_data(
            all_tickers,
            full_start_date.strftime("%Y-%m-%d"),
            full_end_date.strftime("%Y-%m-%d"),
        )
        if full_price_data.empty:
            logger.error("Could not fetch any price data for the backtest period.")
            return pd.DataFrame(), {}

        # Filter to tickers with data for the whole period to avoid errors
        full_price_data = full_price_data.dropna(axis=1, how="any")
        valid_tickers = [t for t in self.config.tickers if t in full_price_data.columns]
        if not valid_tickers:
            logger.error("No tickers have complete data for the specified backtest range.")
            return pd.DataFrame(), {}
        logger.info("Running backtest with %d valid tickers.", len(valid_tickers))

        full_log_returns = calculate_log_returns(full_price_data)

        # 3. Set up rebalancing dates and portfolio tracking
        rebalance_dates = pd.date_range(
            start=self.config.date_range.start,
            end=self.config.date_range.end,
            freq=self.config.backtesting.rebalance_frequency,
        )

        portfolio_values = []
        last_weights = None
        is_first_run = True

        # 4. Main rebalancing loop
        for date in tqdm(rebalance_dates, desc="Backtesting"):
            try:
                # Define the lookback window for this date
                lookback_start_date = date - lookback_offset
                log_returns_slice = full_log_returns.loc[lookback_start_date:date]

                if log_returns_slice.empty or len(log_returns_slice) < 2:
                    logger.warning(
                        "Not enough data for lookback period ending on %s. Skipping.", date
                    )
                    continue

                # Prepare inputs from the sliced data (NO API calls here)
                mean_returns, cov_matrix = self._prepare_inputs_for_date(
                    log_returns_slice[valid_tickers]
                )
                if mean_returns.empty:
                    continue

                # Optimize portfolio
                optimizer: PortfolioOptimizer = self.optimizer_cls(
                    mean_returns, cov_matrix, self.config
                )
                result = optimizer.optimize(lambda_reg=self.config.optimization.lambda_reg)

                if not result or not result.success:
                    if last_weights is None:
                        logger.warning(
                            "Optimization failed on first run for %s. Cannot proceed.", date
                        )
                        continue
                    weights = last_weights
                else:
                    weights = result.opt_weights

                # Calculate returns for the *next* period
                rebalance_freq_offset = pd.tseries.frequencies.to_offset(
                    self.config.backtesting.rebalance_frequency
                )
                period_end_date = date + rebalance_freq_offset
                actual_period_returns = full_log_returns.loc[date:period_end_date]

                if actual_period_returns.empty:
                    if portfolio_values:
                        portfolio_values.append((date, portfolio_values[-1][1]))
                    continue

                total_period_returns = actual_period_returns.sum()
                period_portfolio_return = (
                    weights * total_period_returns.reindex(weights.index).fillna(0.0)
                ).sum()

                if is_first_run:
                    current_value = self.config.backtesting.initial_capital * np.exp(
                        period_portfolio_return
                    )
                    is_first_run = False
                else:
                    if not portfolio_values:
                        continue
                    current_value = portfolio_values[-1][1] * np.exp(period_portfolio_return)

                portfolio_values.append((date, current_value))
                last_weights = weights

            except (ValueError, RuntimeError, KeyError) as e:
                logger.exception(
                    "Error during backtest period for %s. Holding previous state. Error: %s",
                    date,
                    e,
                )
                if portfolio_values:
                    portfolio_values.append((date, portfolio_values[-1][1]))
                continue

        if not portfolio_values:
            logger.warning("Backtest did not produce any results.")
            return pd.DataFrame(), {}

        result_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index(
            "Date"
        )

        # 5. Calculate Benchmark Performance
        if benchmark_ticker and benchmark_ticker in full_price_data.columns:
            benchmark_prices = full_price_data[benchmark_ticker].loc[result_df.index.min() :]
            if not benchmark_prices.empty:
                benchmark_performance = (
                    benchmark_prices / benchmark_prices.iloc[0]
                ) * self.config.backtesting.initial_capital
                result_df["Benchmark Value"] = benchmark_performance.reindex(
                    result_df.index, method="ffill"
                )

        logger.info("Backtest complete. Calculating performance metrics...")
        performance_metrics = self._calculate_performance_metrics(result_df)

        return result_df, performance_metrics

    def _get_periods_per_year(self) -> int:
        """Determines the number of rebalancing periods per year from the config."""
        freq = self.config.backtesting.rebalance_frequency.upper()
        # A simple mapping for common pandas frequency strings
        if "A" in freq or "Y" in freq:
            return 1
        if "Q" in freq:
            return 4
        if "M" in freq:
            return 12
        if "W" in freq:
            return 52
        if "B" in freq or "D" in freq:
            return self.config.trading_days_per_year

        logger.warning(
            "Unrecognized rebalance frequency '%s'. Defaulting annualization to %d periods.",
            self.config.backtesting.rebalance_frequency,
            self.config.trading_days_per_year,
        )
        return self.config.trading_days_per_year

    def _calculate_performance_metrics(self, result_df: pd.DataFrame) -> dict:
        """Calculates key performance metrics for the strategy and benchmark."""
        logger.debug("Calculating performance metrics from backtest results.")
        metrics = {"strategy": {}, "benchmark": {}}
        periods_per_year = self._get_periods_per_year()
        logger.info("Annualizing metrics using %d periods per year.", periods_per_year)

        # --- Strategy Metrics ---
        strategy_series = result_df["Portfolio Value"]
        metrics["strategy"] = calculate_performance_summary(
            strategy_series, self.config.risk_free_rate, periods_per_year
        )

        # --- Benchmark and Relative Metrics ---
        if "Benchmark Value" in result_df.columns:
            benchmark_series = result_df["Benchmark Value"].dropna()
            if not benchmark_series.empty:
                metrics["benchmark"] = calculate_performance_summary(
                    benchmark_series, self.config.risk_free_rate, periods_per_year
                )

                # Calculate Alpha and Beta using the new utility function
                relative_metrics = calculate_relative_metrics(
                    strategy_returns=strategy_series.pct_change(),
                    benchmark_returns=benchmark_series.pct_change(),
                    risk_free_rate=self.config.risk_free_rate,
                    strategy_annualized_return=metrics["strategy"].get("Annualized Return", 0.0),
                    benchmark_annualized_return=metrics["benchmark"].get("Annualized Return", 0.0),
                )
                metrics["strategy"].update(relative_metrics)

        return metrics
