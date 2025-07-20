import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from portfolio_analyzer.data.models import PortfolioResult, SimulationResult


def calculate_correlation_matrix(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate the correlation matrix from a covariance matrix."""
    if not isinstance(cov_matrix, pd.DataFrame) or cov_matrix.empty:
        return pd.DataFrame()

    std_devs = np.sqrt(np.diag(cov_matrix))
    # Avoid division by zero for assets with no variance
    std_devs[std_devs == 0] = 1.0
    inv_std_dev_matrix = np.diag(1 / std_devs)

    # Correlation = D^-1 * Cov * D^-1
    corr_matrix = inv_std_dev_matrix @ cov_matrix.values @ inv_std_dev_matrix

    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    return pd.DataFrame(corr_matrix, index=cov_matrix.index, columns=cov_matrix.columns)


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    if correlation_matrix.empty or len(correlation_matrix) < 2:
        print("Correlation matrix has fewer than 2 assets. Skipping heatmap.")
        return

    print("\nDisplaying Clustered Correlation Heatmap:")
    figsize = (10, 8) if len(correlation_matrix) <= 15 else (12, 10)
    annot_size = 7 if len(correlation_matrix) <= 10 else 5

    try:
        g = sns.clustermap(
            correlation_matrix,
            annot=len(correlation_matrix) <= 15,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            figsize=figsize,
            annot_kws={"size": annot_size},
        )
        g.ax_heatmap.set_title("Clustered Asset Correlation Matrix", fontsize=16, fontweight="bold")
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
        plt.show()
        plt.close(g.fig)
    except Exception as e:
        print(f"Could not generate clustermap: {e}")


def plot_correlation_network(
    correlation_matrix: pd.DataFrame, threshold: float = 0.3, ax: plt.Axes = None
) -> None:
    """Display a network graph of asset correlations.

    Nodes are always displayed. Edges are only shown for correlations
    with an absolute value greater than the threshold.
    """
    if correlation_matrix.empty or len(correlation_matrix) < 2:
        if ax:
            ax.text(0.5, 0.5, "Not enough data for network.", ha="center", va="center")
        else:
            print("Correlation matrix has fewer than 2 assets. Skipping network graph.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    G = nx.from_pandas_adjacency(correlation_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    # --- Filter Edges based on threshold ---
    edges_to_remove = [
        (u, v) for u, v, data in G.edges(data=True) if abs(data.get("weight", 0)) < threshold
    ]
    G.remove_edges_from(edges_to_remove)

    print("\nDisplaying Correlation Network Graph:")

    # --- Setup for plotting ---
    pos = nx.spring_layout(G, k=0.8 / np.sqrt(G.number_of_nodes()), iterations=50)

    # --- Create Plot ---
    # Always draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=800,
        node_color="skyblue",
        alpha=0.9,
        linewidths=1,
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    # Only draw edges and legend if there are any edges left
    if G.edges():
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        edge_colors = ["g" if w > 0 else "r" for w in edge_weights]
        abs_weights = [abs(w) for w in edge_weights]
        min_w, max_w = min(abs_weights), max(abs_weights)
        edge_widths = (
            [1 + 4 * (w - min_w) / (max_w - min_w) for w in abs_weights]
            if max_w > min_w
            else [2.5] * len(edge_weights)
        )

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax)

        legend_elements = [
            plt.Line2D([0], [0], color="g", lw=2, label=f"Positive Correlation (> {threshold})"),
            plt.Line2D(
                [0],
                [0],
                color="r",
                lw=2,
                label=f"Negative Correlation (< -{threshold})",
            ),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=10)
    else:
        ax.text(0.5, 0.5, f"No correlations > {threshold}", ha="center", va="center")

    ax.set_title(
        f"Asset Correlation Network (Threshold: {threshold:.2f})",
        fontsize=16,
        fontweight="bold",
    )
    ax.axis("off")

    if "fig" in locals():
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def display_optimization_summary(result: PortfolioResult) -> None:
    """Prints a formatted summary of the optimization results."""
    if not result or not result.success:
        print("Optimization failed or no results to display.")
        return

    print("\n--- Optimization Results ---\n")
    print("Optimal Weights:")
    if result.opt_weights is not None and not result.opt_weights.empty:
        for ticker_item, weight_item in result.opt_weights.sort_values(ascending=False).items():
            print(f"\t{ticker_item}: {weight_item:.2%}")
        print(f"\tTotal Sum: {result.opt_weights.sum():.4f}")
    else:
        print("\tNo assets in the final portfolio.")

    print("\nPortfolio Performance Metrics:")
    print(f"\tExpected Annualized Return (Arithmetic): {result.arithmetic_return:.2%}")
    print(f"\tAnnualized Volatility: {result.std_dev:.2%}")
    print(f"\tSharpe Ratio (Arithmetic Return based): {result.display_sharpe:.2f}")


def plot_optimal_weights(
    result: PortfolioResult,
    max_weight_per_asset: float,
    ax: plt.Axes = None,
) -> None:
    """Plot a bar chart of the optimal portfolio weights."""
    if not result or not result.success or result.opt_weights is None:
        # If we have an axes, we can display a message on it
        if ax:
            ax.text(0.5, 0.5, "No data to plot.", ha="center", va="center")
            ax.set_title("Optimal Weights")
        return

    if result.opt_weights.empty:
        if ax:
            ax.text(
                0.5,
                0.5,
                "Optimization resulted in an empty portfolio.",
                ha="center",
                va="center",
            )
            ax.set_title("Optimal Weights")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    weights_to_plot = result.opt_weights.sort_values(ascending=False)

    sns.barplot(
        x=weights_to_plot.index,
        y=weights_to_plot.values,
        palette="viridis",
        hue=weights_to_plot.index,
        legend=False,
        ax=ax,
    )
    ax.set_title(
        "Optimal Portfolio Weights",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Stocks", fontsize=15)
    ax.set_ylabel("Weight", fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.axhline(
        y=max_weight_per_asset,
        color="r",
        linestyle="--",
        label=f"Max Weight Limit ({max_weight_per_asset:.0%})",
    )
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    if "fig" in locals():
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def display_simulation_summary(result: SimulationResult) -> None:
    """Print a formatted text summary of simulation statistics to the console.

    Args:
        result (SimulationResult): The Monte Carlo simulation result object.

    """
    print("\n--- Simulation Results Summary ---")
    print(f"Final Portfolio Value (Median): {result.stats['median']:,.2f}")
    print(f"Final Portfolio Value (Mean):   {result.stats['mean']:,.2f}")
    print(f"Standard Deviation:             {result.stats['std_dev']:,.2f}")
    print("-" * 35)
    print(f"95% VaR (Value at Risk):        {result.stats['var_95']:,.2f}")
    print(f"95% CVaR (Conditional VaR):     {result.stats['cvar_95']:,.2f}")
    print(f"Probability of Profit:          {result.stats['prob_breakeven']:.2%}")
    print("-" * 35)


def plot_simulation_distribution(result: SimulationResult, ax: plt.Axes = None):
    """Plot the distribution of final portfolio values from a simulation.

    Args:
        result (SimulationResult): The Monte Carlo simulation result object.
        ax (plt.Axes, optional): A matplotlib axes object to plot on. If None,
            a new figure and axes are created.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        result.final_values,
        kde=True,
        bins=100,
        color="skyblue",
        line_kws={"linewidth": 2.5, "color": "navy"},
        ax=ax,
    )

    ax.axvline(
        result.stats["median"],
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {result.stats['median']:,.0f}",
    )
    ax.axvline(
        result.stats["var_95"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"95% VaR: {result.stats['var_95']:,.0f}",
    )

    title = (
        f"Distribution of Final Portfolio Value after {result.time_horizon_years}"
        f"Year(s)\n"
        f"({result.num_simulations:,} Simulations using {result.dist_model_name} Model)"
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Final Portfolio Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.legend(fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    if "fig" in locals():
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_simulation_paths(result: SimulationResult, ax: plt.Axes = None):
    """Plot a sample of simulated portfolio value paths over time.

    Args:
        result (SimulationResult): The Monte Carlo simulation result object.
        ax (plt.Axes, optional): A matplotlib axes object to plot on. If None,
            a new figure and axes are created.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    num_to_plot = min(500, result.num_simulations)
    paths_to_plot = result.simulation_paths[:, :: result.num_simulations // num_to_plot]

    ax.plot(paths_to_plot, color="lightblue", alpha=0.2, linewidth=1)
    ax.plot(
        result.simulation_paths.mean(axis=1),
        color="blue",
        linewidth=2,
        label="Mean Path",
    )
    ax.plot(
        np.median(result.simulation_paths, axis=1),
        color="orange",
        linewidth=2,
        label="Median Path",
    )

    ax.set_title("Sample of Simulated Portfolio Paths", fontsize=16, fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:,.0f}"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend()
    if "fig" in locals():
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    max_sharpe_result: PortfolioResult,
    min_vol_result: PortfolioResult,
    current_opt_result: PortfolioResult | None = None,
    ax: plt.Axes = None,
) -> None:
    """Plot the efficient frontier with key portfolios highlighted.

    Args:
        frontier_df (pd.DataFrame): DataFrame of return and volatility points.
        max_sharpe_result (PortfolioResult): The max Sharpe ratio portfolio result.
        min_vol_result (PortfolioResult): The minimum volatility portfolio result.
        current_opt_result (Optional[PortfolioResult]): The currently selected
            optimized portfolio to highlight.
        ax (plt.Axes, optional): A matplotlib axes object to plot on. If None,
            a new figure and axes are created.

    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the frontier (Y-axis is log return)
    ax.plot(
        frontier_df["Volatility"],
        frontier_df["Return"],
        linestyle="--",
        color="blue",
        linewidth=2,
        label="Efficient Frontier",
    )

    # --- Highlight Key Portfolios using LOG returns for consistency ---

    # Max Sharpe Ratio Portfolio
    if max_sharpe_result and max_sharpe_result.success:
        ax.scatter(
            max_sharpe_result.std_dev,
            max_sharpe_result.log_return,  # Use log_return
            marker="*",
            color="red",
            s=250,
            label=f"Max Sharpe (SR: {max_sharpe_result.display_sharpe:.2f})",
            zorder=5,
        )

    # Minimum Volatility Portfolio
    if min_vol_result and min_vol_result.success:
        ax.scatter(
            min_vol_result.std_dev,
            min_vol_result.log_return,  # Use log_return
            marker="X",
            color="green",
            s=200,
            label=f"Min Volatility (Vol: {min_vol_result.std_dev:.2%})",
            zorder=5,
        )

    # Current Interactive/Optimized Portfolio
    if current_opt_result and current_opt_result.success:
        is_not_max_sharpe = not np.isclose(
            current_opt_result.std_dev, max_sharpe_result.std_dev
        ) or not np.isclose(current_opt_result.log_return, max_sharpe_result.log_return)

        if is_not_max_sharpe:
            ax.scatter(
                current_opt_result.std_dev,
                current_opt_result.log_return,  # Use log_return
                marker="o",
                edgecolor="black",
                color="orange",
                s=150,
                label=f"Current Opt (SR: {current_opt_result.display_sharpe:.2f})",
                zorder=4,
            )

    # Formatting
    ax.set_title("Efficient Frontier Analysis", fontsize=18, fontweight="bold")
    ax.set_xlabel("Volatility (Annualized Std. Dev)", fontsize=12)
    ax.set_ylabel("Expected Log Return (Annualized)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.2%}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.2%}"))
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    if fig:
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_backtest_results(
    backtest_results: pd.DataFrame,
    benchmark_ticker: str | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Plot the portfolio value over time from a backtest against a benchmark.

    Args:
        backtest_results (pd.DataFrame): DataFrame containing the portfolio
            value series and optionally a benchmark series.
        benchmark_ticker (Optional[str]): The ticker of the benchmark used.
        ax (Optional[plt.Axes]): A matplotlib axes object to plot on. If None,
            a new figure and axes are created.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    backtest_results["Portfolio Value"].plot(ax=ax, linewidth=2, label="Strategy")

    if "Benchmark Value" in backtest_results.columns and benchmark_ticker:
        backtest_results["Benchmark Value"].plot(
            ax=ax,
            linewidth=2,
            label=f"Benchmark ({benchmark_ticker})",
            linestyle="--",
        )

    ax.set_title("Backtest: Strategy vs. Benchmark", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)

    if fig:
        plt.tight_layout()
        plt.show()
        plt.close(fig)
