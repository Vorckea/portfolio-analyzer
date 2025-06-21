from typing import Optional

from IPython.display import display
from matplotlib import pyplot as plt

from portfolio_analyzer.analysis.monte_carlo_simulator import MonteCarloSimulator
from portfolio_analyzer.config import AppConfig
from portfolio_analyzer.core.portfolio_optimizer import PortfolioOptimizer, PortfolioResult
from portfolio_analyzer.data import input_preparator as ip
from portfolio_analyzer.reporting.plotting import (
    plot_optimal_weights,
    plot_simulation_distribution,
    plot_simulation_paths,
)
from portfolio_analyzer.reporting.reporting import (
    display_optimization_summary_html,
    display_simulation_summary_html,
)


class PortfolioAnalysisSession:
    def __init__(self, config: AppConfig, model_inputs: ip.ModelInputs):
        self.config = config
        self.model_inputs = model_inputs
        self.latest_result: Optional[PortfolioResult] = None

        if not self.model_inputs.mean_returns.empty:
            self.optimizer = PortfolioOptimizer(
                mean_returns=self.model_inputs.mean_returns,
                cov_matrix=self.model_inputs.cov_matrix,
                config=self.config,
            )
            self.mc_simulator = MonteCarloSimulator(self.config)
        else:
            self.optimizer = None
            self.mc_simulator = None

    def run_interactive_optimization(self, lambda_reg: float):
        """Run optimization and displays the summary and weights plot."""
        if not self.optimizer:
            print("Optimizer not initialized due to data pipeline failure.")
            return

        # Store the result within the session
        self.latest_result = self.optimizer.optimize(lambda_reg=lambda_reg)

        if self.latest_result and self.latest_result.success:
            display(display_optimization_summary_html(self.latest_result))
            plot_optimal_weights(
                self.latest_result, self.config.optimization.max_weight_per_asset, lambda_reg
            )
        else:
            print("Optimization failed. Could not generate a valid portfolio.")

    def run_interactive_monte_carlo(
        self, num_sim_interactive: int, time_horizon_interactive: float, df_t_interactive: int
    ):
        """Run Monte Carlo simulation on the latest optimized portfolio."""
        if not self.latest_result or not self.latest_result.success:
            print("Optimization result not available. Please run the optimization widget first.")
            return

        try:
            simulation_result = self.mc_simulator.run(
                portfolio_result=self.latest_result,
                num_simulations=int(num_sim_interactive),
                time_horizon_years=time_horizon_interactive,
                df_t_distribution=int(df_t_interactive),
            )
            display(display_simulation_summary_html(simulation_result))

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            plot_simulation_distribution(simulation_result, ax=axes[0])
            plot_simulation_paths(simulation_result, ax=axes[1])
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
