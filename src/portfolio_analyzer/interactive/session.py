from typing import Optional

from IPython.display import display
from matplotlib import pyplot as plt

from portfolio_analyzer.analysis.monte_carlo_simulator import MonteCarloSimulator
from portfolio_analyzer.config.config import AppConfig
from portfolio_analyzer.core.portfolio_optimizer import PortfolioOptimizer
from portfolio_analyzer.data import models as ip
from portfolio_analyzer.data.models import PortfolioResult
from portfolio_analyzer.reporting.display import (
    display_optimization_summary_html,
    display_simulation_summary_html,
)
from portfolio_analyzer.reporting.plotting import (
    plot_optimal_weights,
    plot_simulation_distribution,
    plot_simulation_paths,
)


class PortfolioAnalysisSession:
    """Manages the state and logic for an interactive analysis session.

    This class holds the optimizer and simulator instances and provides methods
    that can be linked to Jupyter widgets for interactive exploration of
    portfolio optimization and simulation parameters.

    Attributes:
        config (AppConfig): The application configuration object.
        model_inputs (ip.ModelInputs): The prepared data inputs for the models.
        latest_result (Optional[PortfolioResult]): The result of the last optimization.
        optimizer (Optional[PortfolioOptimizer]): The portfolio optimizer instance.
        mc_simulator (Optional[MonteCarloSimulator]): The Monte Carlo simulator instance.

    """

    def __init__(
        self,
        config: AppConfig,
        model_inputs: ip.ModelInputs,
        optimizer: Optional[PortfolioOptimizer] = None,
        mc_simulator: Optional[MonteCarloSimulator] = None,
    ):
        """Initialize the PortfolioAnalysisSession.

        Args:
            config (AppConfig): The application configuration object.
            model_inputs (ip.ModelInputs): The prepared data inputs for the models.
            optimizer (Optional[PortfolioOptimizer]): The portfolio optimizer instance.
            mc_simulator (Optional[MonteCarloSimulator]): The Monte Carlo simulator instance.

        """
        self.config = config
        self.model_inputs = model_inputs
        self.latest_result: Optional[PortfolioResult] = None
        self.optimizer = optimizer
        self.mc_simulator = mc_simulator

    def run_interactive_optimization(self, lambda_reg: float):
        """Run optimization and displays results for interactive use.

        This method is designed to be called from an `ipywidgets.interact`
        slider to allow real-time updates of the L2 regularization parameter.

        Args:
            lambda_reg (float): The L2 regularization coefficient.

        """
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
        """Run Monte Carlo simulation and displays results for interactive use.

        This method is designed to be called from `ipywidgets.interact` sliders
        to allow real-time updates of simulation parameters.

        Args:
            num_sim_interactive (int): The number of simulation paths.
            time_horizon_interactive (float): The simulation time horizon in years.
            df_t_interactive (int): The degrees of freedom for the Student's t-distribution.

        """
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
