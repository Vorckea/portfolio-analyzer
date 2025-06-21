from IPython.display import HTML
import pandas as pd

from portfolio_analyzer.analysis.monte_carlo_simulator import SimulationResult
from portfolio_analyzer.core.portfolio_optimizer import PortfolioResult


def display_optimization_summary_html(result: PortfolioResult) -> HTML:
    """Return a styled HTML summary of the optimization results."""
    if not result or not result.success:
        html = """
        <div style="display: flex; justify-content: flex-start;">
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #c0392b; border: 1px solid #e74c3c; background-color: #fbe9e7; border-radius: 10px; padding: 20px; width: 550px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <strong>Optimization Failed:</strong> Could not generate a valid portfolio.
            </div>
        </div>
        """  # noqa: E501
        return HTML(html)

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
    return HTML(html)


def display_simulation_summary_html(result: SimulationResult) -> HTML:
    """Return a cleaner, left-aligned HTML summary of the simulation results."""
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
    return HTML(style + html)


def display_backtest_summary_html(metrics: dict) -> HTML:
    """Return a styled HTML summary of the backtest performance metrics."""
    style = """
    <style>
        .summary-container { display: flex; justify-content: space-between; gap: 20px; }
        .summary-card { flex: 1; background-color: #fdfdfd; border: 1px solid #e8e8e8; border-radius: 10px; padding: 25px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #333; }
        .summary-card h3 { margin-top: 0; margin-bottom: 20px; font-size: 1.3em; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #f0f0f0; padding-bottom: 15px; }
        .summary-table { width: 100%; border-collapse: collapse; }
        .summary-table th, .summary-table td { text-align: left; padding: 12px 15px; border-bottom: 1px solid #f5f5f5; }
        .summary-table th { font-weight: 600; color: #444; background-color: #f9f9f9; }
        .summary-table td:nth-child(1) { font-weight: 500; }
        .summary-table tr:last-child td { border-bottom: none; }
        .summary-table td:not(:first-child) { text-align: right; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; }
    </style>
    """  # noqa: E501

    strat_metrics = metrics.get("strategy", {})
    bench_metrics = metrics.get("benchmark", {})

    def get_metric(data, key, fmt):
        val = data.get(key)
        if val is None or pd.isna(val):
            return "N/A"
        if fmt == "pct":
            return f"{val:.2%}"
        if fmt == "num":
            return f"{val:,.2f}"
        if fmt == "dec":
            return f"{val:.3f}"
        return str(val)

    html = f"""
    <div class="summary-card">
        <h3>Backtest Performance</h3>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Strategy</th>
                    <th>Benchmark</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Final Value</td>
                    <td>{get_metric(strat_metrics, "Final Value", "num")}</td>
                    <td>{get_metric(bench_metrics, "Final Value", "num")}</td>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>{get_metric(strat_metrics, "Total Return", "pct")}</td>
                    <td>{get_metric(bench_metrics, "Total Return", "pct")}</td>
                </tr>
                <tr>
                    <td>Annualized Return</td>
                    <td>{get_metric(strat_metrics, "Annualized Return", "pct")}</td>
                    <td>{get_metric(bench_metrics, "Annualized Return", "pct")}</td>
                </tr>
                <tr>
                    <td>Annualized Volatility</td>
                    <td>{get_metric(strat_metrics, "Annualized Volatility", "pct")}</td>
                    <td>{get_metric(bench_metrics, "Annualized Volatility", "pct")}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{get_metric(strat_metrics, "Sharpe Ratio", "dec")}</td>
                    <td>{get_metric(bench_metrics, "Sharpe Ratio", "dec")}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{get_metric(strat_metrics, "Max Drawdown", "pct")}</td>
                    <td>{get_metric(bench_metrics, "Max Drawdown", "pct")}</td>
                </tr>
                <tr>
                    <td>Beta</td>
                    <td>{get_metric(strat_metrics, "Beta", "dec")}</td>
                    <td>{1.0:.3f}</td>
                </tr>
                <tr>
                    <td>Alpha</td>
                    <td>{get_metric(strat_metrics, "Alpha", "dec")}</td>
                    <td>{0.0:.3f}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    return HTML(style + html)
