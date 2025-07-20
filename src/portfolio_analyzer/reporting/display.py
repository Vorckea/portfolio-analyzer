import pandas as pd
from IPython.display import HTML

from portfolio_analyzer.data.models import PortfolioResult, SimulationResult
from portfolio_analyzer.utils.html_helpers import get_summary_card_html


def display_optimization_summary_html(result: PortfolioResult) -> HTML:
    """Generate a styled HTML summary of portfolio optimization results.

    Args:
        result (PortfolioResult): The portfolio optimization result object.

    Returns:
        HTML: An IPython.display.HTML object containing the formatted summary.

    """
    if not result or not result.success:
        html = """
        <div style="display: flex; justify-content: flex-start;">
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #c0392b; border: 1px solid #e74c3c; background-color: #fbe9e7; border-radius: 10px; padding: 20px; width: 550px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <strong>Optimization Failed:</strong> Could not generate a valid portfolio.
            </div>
        </div>
        """  # noqa: E501
        return HTML(html)

    if (opt_weights := result.opt_weights) is not None or opt_weights.empty:
        sorted_weights = opt_weights.sort_values(ascending=False)
        weights_html = "".join(
            f'<li><span class="ticker-name">{ticker}</span><span class="ticker-weight">{weight:.2%}</span></li>'
            for ticker, weight in sorted_weights.items()
        )
    else:
        weights_html = "<li>No assets in the final portfolio.</li>"

    body_html = f"""
    <div class="metrics-grid">
        <div class="metric"><span class="metric-label">Expected Return</span><span class="metric-value">{result.arithmetic_return:.2%}</span></div>
        <div class="metric"><span class="metric-label">Volatility</span><span class="metric-value">{result.std_dev:.2%}</span></div>
        <div class="metric"><span class="metric-label">Sharpe Ratio</span><span class="metric-value">{result.display_sharpe:.2f}</span></div>
    </div>
    <h4>Asset Allocation</h4>
    <ul class="weights-list">{weights_html}</ul>
    """  # noqa: E501
    return HTML(get_summary_card_html("Optimal Portfolio Summary", "", body_html))


def display_simulation_summary_html(result: SimulationResult) -> HTML:
    """Generate a cleaner, left-aligned HTML summary of the simulation results.

    Args:
        result (SimulationResult): The Monte Carlo simulation result object.

    Returns:
        HTML: An IPython.display.HTML object containing the formatted summary.

    """
    title = "Monte Carlo Simulation Summary"
    subtitle = f"Ran <strong>{result.num_simulations:,}</strong> simulations over <strong>{result.time_horizon_years}</strong> year(s)."  # noqa: E501
    stats = result.stats

    def get_stat(key, fmt="{:,.2f}"):
        val = stats.get(key)
        return fmt.format(val) if val is not None else "N/A"

    body_html = f"""
    <div class="metrics-grid">
        <div class="metric">
            <span class="metric-label">5th Percentile</span>
            <span class="metric-value">{get_stat("ci_5")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Median Value</span>
            <span class="metric-value">{get_stat("median")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">95th Percentile</span>
            <span class="metric-value">{get_stat("ci_95")}</span>
        </div>
    </div>
    """
    return HTML(get_summary_card_html(title, subtitle, body_html))


def display_backtest_summary_html(metrics: dict) -> HTML:
    """Generate a styled HTML summary of backtest performance metrics.

    Args:
        metrics (dict): A dictionary of performance metrics from a backtest run.

    Returns:
        HTML: An IPython.display.HTML object containing the formatted summary table.

    """
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

    body_html = f"""
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
    """
    return HTML(get_summary_card_html("Backtest Performance", "Strategy vs. Benchmark", body_html))
