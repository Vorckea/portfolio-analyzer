import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from finance_project.monte_carlo_simulator import SimulationResult
from finance_project.portfolio_optimizer import PortfolioResult


def calculate_correlation_matrix(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculates the correlation matrix from a covariance matrix."""
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
        g.ax_heatmap.set_title(
            "Clustered Asset Correlation Matrix", fontsize=16, fontweight="bold"
        )
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
        plt.show()
    except Exception as e:
        print(f"Could not generate clustermap: {e}")


def plot_correlation_network(
    correlation_matrix: pd.DataFrame, threshold: float = 0.3, ax: plt.Axes = None
) -> None:
    """Displays a network graph of asset correlations.

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
        (u, v)
        for u, v, data in G.edges(data=True)
        if abs(data.get("weight", 0)) < threshold
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

        nx.draw_networkx_edges(
            G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax
        )

        legend_elements = [
            plt.Line2D(
                [0], [0], color="g", lw=2, label=f"Positive Correlation (> {threshold})"
            ),
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


def display_optimization_summary(result: PortfolioResult) -> None:
    """Prints a formatted summary of the optimization results."""
    if not result or not result.success:
        print("Optimization failed or no results to display.")
        return

    print("\n--- Optimization Results ---\n")
    print("Optimal Weights:")
    if result.opt_weights is not None and not result.opt_weights.empty:
        for ticker_item, weight_item in result.opt_weights.sort_values(
            ascending=False
        ).items():
            print(f"\t{ticker_item}: {weight_item:.2%}")
        print(f"\tTotal Sum: {result.opt_weights.sum():.4f}")
    else:
        print("\tNo assets in the final portfolio.")

    print("\nPortfolio Performance Metrics:")
    print(f"\tExpected Annualized Return (Arithmetic): {result.arithmetic_return:.2%}")
    print(f"\tAnnualized Volatility: {result.std_dev:.2%}")
    print(f"\tSharpe Ratio (Arithmetic Return based): {result.display_sharpe:.2f}")


def display_optimization_summary_html(result: PortfolioResult) -> None:
    """Displays a styled HTML summary of the optimization results with a two-column
    layout for weights."""
    if not result or not result.success:
        html = """
        <div style="display: flex; justify-content: flex-start;">
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #c0392b; border: 1px solid #e74c3c; background-color: #fbe9e7; border-radius: 10px; padding: 20px; width: 550px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <strong>Optimization Failed:</strong> Could not generate a valid portfolio.
            </div>
        </div>
        """  # noqa: E501
        display(HTML(html))
        return

    style = """
    <style>
        .summary-container { display: flex; justify-content: flex-start; }
        .summary-card { background-color: #fdfdfd; border: 1px solid #e8e8e8; border-radius: 10px; padding: 25px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; width: 550px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #333; }
        .summary-card h3 { margin-top: 0; margin-bottom: 20px; font-size: 1.3em; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #f0f0f0; padding-bottom: 15px; }
        .summary-card h4 { margin-top: 25px; margin-bottom: 10px; font-size: 1.1em; font-weight: 600; color: #444; }
        .summary-card .metrics-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .summary-card .metric { display: flex; flex-direction: column; text-align: center; background-color: #f9f9f9; padding: 12px; border-radius: 8px; }
        .summary-card .metric-label { font-size: 0.85em; color: #666; margin-bottom: 5px; }
        .summary-card .metric-value { font-size: 1.25em; font-weight: 600; color: #005a9e; }
        /* --- Use a two-column grid for the weights list --- */
        .summary-card .weights-list {
            list-style-type: none;
            padding-left: 0;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0 25px; /* Row and column gap */
        }
        .summary-card .weights-list li { display: flex; justify-content: space-between; padding: 9px 5px; border-bottom: 1px solid #f5f5f5; }
        .summary-card .weights-list li:nth-last-child(-n+2) {
             border-bottom: none; /* Removes border from last items in both columns if item count is even */
        }
        .summary-card .ticker-name { font-weight: 500; }
        .summary-card .ticker-weight { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-weight: 600; }
    </style>
    """  # noqa: E501

    weights_html = ""
    if result.opt_weights is not None and not result.opt_weights.empty:
        sorted_weights = result.opt_weights.sort_values(ascending=False)
        for ticker, weight in sorted_weights.items():
            weights_html += f'<li><span class="ticker-name">{ticker}</span><span class="ticker-weight">{weight:.2%}</span></li>'  # noqa: E501

        # If the number of items is odd, the last item's border is not correctly handled
        # by nth-last-child.
        # This ensures the very last item never has a bottom border.
        if len(sorted_weights) % 2 != 0:
            weights_html = weights_html.replace(
                "<li><span", '<li style="border-bottom: none;"><span', 1
            )

    else:
        weights_html = "<li>No assets in the final portfolio.</li>"

    html = f"""
    {style}
    <div class="summary-container">
        <div class="summary-card">
            <h3>Optimal Portfolio Summary</h3>
            <div class="metrics-grid">
                <div class="metric"><span class="metric-label">Expected Return</span><span class="metric-value">{result.arithmetic_return:.2%}</span></div>
                <div class="metric"><span class="metric-label">Volatility</span><span class="metric-value">{result.std_dev:.2%}</span></div>
                <div class="metric"><span class="metric-label">Sharpe Ratio</span><span class="metric-value">{result.display_sharpe:.2f}</span></div>
            </div>
            <h4>Asset Allocation</h4>
            <ul class="weights-list">{weights_html}</ul>
        </div>
    </div>
    """  # noqa: E501
    display(HTML(html))


def plot_optimal_weights(
    result: PortfolioResult,
    max_weight_per_asset: float,
    lambda_reg: float,
    ax: plt.Axes = None,
) -> None:
    """Plots a bar chart of the optimal portfolio weights."""
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
        f"Optimal Portfolio Weights (Lambda = {lambda_reg:.2f})",
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


def display_simulation_summary(result: SimulationResult) -> None:
    """Prints a formatted summary of simulation statistics."""
    print("\n--- Simulation Results Summary ---")
    print(f"Final Portfolio Value (Median): {result.stats['median']:,.2f}")
    print(f"Final Portfolio Value (Mean):   {result.stats['mean']:,.2f}")
    print(f"Standard Deviation:             {result.stats['std_dev']:,.2f}")
    print("-" * 35)
    print(f"95% VaR (Value at Risk):        {result.stats['var_95']:,.2f}")
    print(f"95% CVaR (Conditional VaR):     {result.stats['cvar_95']:,.2f}")
    print(f"Probability of Profit:          {result.stats['prob_breakeven']:.2%}")
    print("-" * 35)


def display_simulation_summary_html(result: SimulationResult) -> None:
    """Displays a cleaner, left-aligned HTML summary of the simulation results."""
    stats = result.stats
    dist_name = result.dist_model_name
    title = f"Monte Carlo Simulation ({dist_name})"

    # CSS styles for a cleaner, modern look
    style = """
    <style>
        .summary-container {
            display: flex;
            justify-content: flex-start; /* Aligns the card to the left */
        }
        .summary-card {
            background-color: #fdfdfd;
            border: 1px solid #e8e8e8;
            border-radius: 10px;
            padding: 25px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            width: 500px; /* Set a fixed width for the card */
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            color: #333;
        }
        .summary-card h3 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 1.3em;
            font-weight: 600;
            color: #1a1a1a;
        }
        .summary-card .subtitle {
            font-size: 0.95em;
            color: #777;
            margin-bottom: 20px;
            border-bottom: 1px solid #f0f0f0;
            padding-bottom: 15px;
        }
        .summary-card .grid {
            display: grid;
            grid-template-columns: 1fr 1fr; /* Two-column layout */
            gap: 15px 25px; /* Row and column gap */
        }
        .summary-card .metric {
            display: flex;
            flex-direction: column;
        }
        .summary-card .metric-label {
            font-size: 0.9em;
            color: #666;
        }
        .summary-card .metric-value {
            font-size: 1.15em;
            font-weight: 500;
        }
        .summary-card .metric-value.positive {
            color: #27ae60;
        }
        .summary-card .metric-value.negative {
            color: #c0392b;
        }
    </style>
    """  # noqa: E501

    # HTML structure for the card
    html = f"""
    {style}
    <div class="summary-container">
        <div class="summary-card">
            <h3>{title}</h3>
            <div class="subtitle">
                Ran <strong>{result.num_simulations:,}</strong> simulations over <strong>{result.time_horizon_years}</strong> year(s).
            </div>
            <div class="grid">
                <div class="metric">
                    <span class="metric-label">Median Final Value</span>
                    <span class="metric-value">{stats["median"]:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mean Final Value</span>
                    <span class="metric-value">{stats["mean"]:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">95% Value at Risk (VaR)</span>
                    <span class="metric-value negative">{stats["var_95"]:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">95% Conditional VaR</span>
                    <span class="metric-value negative">{stats["cvar_95"]:,.2f}</span>
                </div>
                 <div class="metric">
                    <span class="metric-label">Volatility</span>
                    <span class="metric-value">{stats["std_dev"]:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Probability of Profit</span>
                    <span class="metric-value positive">{stats["prob_breakeven"]:.2%}</span>
                </div>
            </div>
        </div>
    </div>
    """  # noqa: E501
    display(HTML(html))


def plot_simulation_distribution(result: SimulationResult, ax: plt.Axes = None):
    """Plots the distribution of final portfolio values."""
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


def plot_simulation_paths(result: SimulationResult, ax: plt.Axes = None):
    """Plots a sample of the simulated portfolio value paths over time."""
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
