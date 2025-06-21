import numpy as np
import pandas as pd
from tqdm import tqdm

from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.data_fetcher import fetch_price_data
from portfolio_analyzer.input_preparator import prepare_model_inputs
from portfolio_analyzer.portfolio_optimizer import PortfolioOptimizer


class Backtester:
    """Runs a historical backtest of portfolio optimization strategy."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.strategy_name = "Mean-Variance Optimization"

    def run(self, benchmark_ticker: str = None) -> pd.DataFrame:
        """Execute the backtest over the specified historical period.

        Returns:
            A DataFrame containing the portfolio value over time.

        """
        print(f"\n--- Running Backtest for '{self.strategy_name}' ---")

        all_tickers = self.config.tickers
        if benchmark_ticker and benchmark_ticker not in all_tickers:
            all_tickers = all_tickers + [benchmark_ticker]

        full_price_data = fetch_price_data(
            all_tickers, self.config.date_range.start, self.config.date_range.end
        )
        if full_price_data.empty:
            print("Could not fetch any price data for the backtest period.")
            return pd.DataFrame()

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

            except Exception as e:
                print(f"Error during backtest on {date}: {e}. Holding previous state.")
                if portfolio_values:
                    portfolio_values.append((date, portfolio_values[-1][1]))
                continue

        if not portfolio_values:
            return pd.DataFrame()

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

        return result_df
