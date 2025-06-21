"""Backtester class for running historical backtests of portfolio optimization strategies.

This class uses the provided configuration to fetch historical price data,
prepare model inputs, and optimize the portfolio based on the specified strategy.
It supports rebalancing at specified frequencies and can compare the strategy's performance
against a benchmark ticker if provided.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.data_fetcher import fetch_price_data
from portfolio_analyzer.input_preparator import prepare_model_inputs
from portfolio_analyzer.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


class Backtester:
    """Backtester class for executing portfolio optimization strategies over historical data."""

    def __init__(self, config: AppConfig):
        """Initialize the Backtester with the application configuration.

        Args:
            config (AppConfig): The application configuration containing all necessary parameters
            for backtesting, including tickers, date range, and optimization settings.

        """
        self.config = config
        self.strategy_name = "Mean-Variance Optimization"

    def run(self, benchmark_ticker: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
        """Execute the backtest over the specified historical period.

        Args:
            benchmark_ticker (Optional[str]): Ticker for benchmark comparison.

        Returns:
            A tuple containing:
                - A DataFrame with the portfolio value over time.
                - A dictionary with performance metrics.

        """
        logger.info("--- Running Backtest for '%s' ---", self.strategy_name)

        all_tickers = self.config.tickers.copy()
        if benchmark_ticker and benchmark_ticker not in all_tickers:
            all_tickers.append(benchmark_ticker)

        full_price_data = fetch_price_data(
            all_tickers,
            self.config.date_range.start.strftime("%Y-%m-%d"),
            self.config.date_range.end.strftime("%Y-%m-%d"),
        )
        if full_price_data.empty:
            logger.error("Could not fetch any price data for the backtest period.")
            return pd.DataFrame(), {}

        full_log_returns = np.log(full_price_data / full_price_data.shift(1))

        rebalance_dates = pd.date_range(
            start=self.config.date_range.start,
            end=self.config.date_range.end,
            freq=self.config.backtesting.rebalance_frequency,
        )

        portfolio_values = []
        last_weights = None
        is_first_run = True

        for date in tqdm(rebalance_dates, desc="Backtesting"):
            # Create a temporary config for the lookback window
            temp_config = self.config.model_copy(deep=True)
            temp_config.date_range.end = date
            temp_config.date_range.start = date - pd.DateOffset(
                days=self.config.backtesting.lookback_period_days
            )

            try:
                # Prepare data for this lookback period
                model_inputs = prepare_model_inputs(temp_config)
                if model_inputs.mean_returns.empty:
                    continue

                # Optimize portfolio
                optimizer = PortfolioOptimizer(
                    model_inputs.mean_returns, model_inputs.cov_matrix, self.config
                )
                result = optimizer.optimize(lambda_reg=self.config.optimization.lambda_reg)

                if not result or not result.success:
                    if last_weights is None:
                        continue  # Cannot proceed without initial weights
                    # If optimization fails, hold previous weights
                    weights = last_weights
                else:
                    weights = result.opt_weights

                # Calculate returns for the *next* period using the full data
                rebalance_freq_offset = pd.tseries.frequencies.to_offset(
                    self.config.backtesting.rebalance_frequency
                )
                period_end_date = date + rebalance_freq_offset

                # Slice the full return data for the forward-looking period
                actual_period_returns = full_log_returns.loc[date:period_end_date]

                if actual_period_returns.empty:
                    if portfolio_values:  # If no returns, hold value
                        portfolio_values.append((date, portfolio_values[-1][1]))
                    continue

                # Sum returns over the period and align with portfolio weights
                total_period_returns = actual_period_returns.sum()

                # Calculate portfolio return, filling missing asset returns with 0
                period_portfolio_return = (
                    weights * total_period_returns.reindex(weights.index).fillna(0.0)
                ).sum()

                # Convert log return to arithmetic return for value calculation
                if is_first_run:
                    current_value = self.config.backtesting.initial_capital * np.exp(
                        period_portfolio_return
                    )
                    is_first_run = False
                else:
                    # Ensure we have a previous value to compound
                    if not portfolio_values:
                        continue
                    current_value = portfolio_values[-1][1] * np.exp(period_portfolio_return)

                portfolio_values.append((date, current_value))
                last_weights = weights

            except Exception:
                logger.exception("Error during backtest on %s. Holding previous state.", date)
                if portfolio_values:
                    portfolio_values.append((date, portfolio_values[-1][1]))
                continue

        if not portfolio_values:
            logger.warning("Backtest did not produce any results.")
            return pd.DataFrame(), {}

        result_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"])
        result_df = result_df.set_index("Date")

        # 3. Calculate Benchmark Performance
        if benchmark_ticker and benchmark_ticker in full_price_data.columns:
            benchmark_prices = full_price_data[benchmark_ticker].dropna()
            if not benchmark_prices.empty:
                # Normalize benchmark to start at the same initial capital
                benchmark_performance = (
                    benchmark_prices / benchmark_prices.iloc[0]
                ) * self.config.backtesting.initial_capital
                # Reindex benchmark to align with strategy's rebalance dates, filling gaps
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
        strategy_metrics = metrics["strategy"]
        benchmark_metrics = metrics["benchmark"]

        periods_per_year = self._get_periods_per_year()
        logger.info("Annualizing metrics using %d periods per year.", periods_per_year)

        # --- Strategy Metrics ---
        portfolio_series = result_df["Portfolio Value"]
        strategy_metrics["Final Value"] = portfolio_series.iloc[-1]
        strategy_metrics["Total Return"] = (
            portfolio_series.iloc[-1] / portfolio_series.iloc[0]
        ) - 1

        portfolio_returns = portfolio_series.pct_change().dropna()
        annualized_return = portfolio_returns.mean() * periods_per_year
        strategy_metrics["Annualized Return"] = annualized_return

        volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        strategy_metrics["Annualized Volatility"] = volatility

        sharpe_ratio = (
            (annualized_return - self.config.risk_free_rate) / volatility
            if volatility != 0
            else 0.0
        )
        strategy_metrics["Sharpe Ratio"] = sharpe_ratio

        peak = portfolio_series.expanding(min_periods=1).max()
        drawdown = (portfolio_series - peak) / peak
        strategy_metrics["Max Drawdown"] = drawdown.min()

        # --- Benchmark Metrics ---
        if "Benchmark Value" in result_df.columns:
            benchmark_series = result_df["Benchmark Value"].dropna()
            benchmark_metrics["Final Value"] = benchmark_series.iloc[-1]
            benchmark_metrics["Total Return"] = (
                benchmark_series.iloc[-1] / benchmark_series.iloc[0]
            ) - 1

            benchmark_returns = benchmark_series.pct_change().dropna()
            bench_annualized_return = benchmark_returns.mean() * periods_per_year
            benchmark_metrics["Annualized Return"] = bench_annualized_return

            benchmark_volatility = benchmark_returns.std() * np.sqrt(periods_per_year)
            benchmark_metrics["Annualized Volatility"] = benchmark_volatility

            bench_sharpe_ratio = (
                (bench_annualized_return - self.config.risk_free_rate) / benchmark_volatility
                if benchmark_volatility != 0
                else 0.0
            )
            benchmark_metrics["Sharpe Ratio"] = bench_sharpe_ratio

            bench_peak = benchmark_series.expanding(min_periods=1).max()
            bench_drawdown = (benchmark_series - bench_peak) / bench_peak
            benchmark_metrics["Max Drawdown"] = bench_drawdown.min()

            # --- Relative Metrics (Alpha, Beta) ---
            aligned_df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
            aligned_df.columns = ["Portfolio", "Benchmark"]

            if len(aligned_df) > 1 and aligned_df["Benchmark"].var() > 0:
                cov_matrix = aligned_df.cov()
                beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
                strategy_metrics["Beta"] = beta

                alpha = annualized_return - (
                    self.config.risk_free_rate
                    + beta * (bench_annualized_return - self.config.risk_free_rate)
                )
                strategy_metrics["Alpha"] = alpha
            else:
                strategy_metrics["Beta"] = np.nan
                strategy_metrics["Alpha"] = np.nan

        return metrics
