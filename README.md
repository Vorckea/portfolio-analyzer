# Portfolio Analyzer

Portfolio Analyzer is a modular toolkit for advanced portfolio construction, optimization, and risk analytics. It leverages state-of-the-art quantitative finance techniques, including Black-Litterman blending, robust covariance estimation, and Monte Carlo simulation, all accessible through an interactive Jupyter notebook workflow.

## Key Features

- **Automated Data Pipeline**: Fetches and processes historical prices and market capitalizations for user-specified assets.
- **Robust Statistical Estimation**: Implements EWMA and Ledoit-Wolf shrinkage for stable mean and covariance estimates.
- **Black-Litterman Integration**: Blends equilibrium returns with forward-looking views, including DCF-based signals.
- **Flexible Portfolio Optimization**: Maximizes risk-adjusted return (Sharpe Ratio) with support for constraints and L2 regularization.
- **Comprehensive Risk Analysis**: Projects future portfolio outcomes using Monte Carlo simulation with configurable distributional assumptions.
- **Interactive Workflow**: Step-by-step analysis, optimization, and backtesting in [notebooks/portfolio_analysis.ipynb](notebooks/portfolio_analysis.ipynb).
- **Advanced Visualization**: Correlation heatmaps, network graphs, efficient frontier, and detailed backtest reporting.
- **Historical Backtesting**: Evaluates strategy performance with rolling rebalancing and benchmark comparison.

## Installation

Portfolio Analyzer uses [Poetry](https://python-poetry.org/) for dependency management.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Vorckea/portfolio-analyzer.git
    cd portfolio-analyzer
    ```

2. **Install dependencies:**
    ```bash
    poetry install
    ```

## Usage

The primary workflow is provided in [notebooks/portfolio_analysis.ipynb](notebooks/portfolio_analysis.ipynb):

1. **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

2. **Launch Jupyter Lab:**
    ```bash
    jupyter lab
    ```

3. Open and execute the cells in [notebooks/portfolio_analysis.ipynb](notebooks/portfolio_analysis.ipynb) to perform configuration, data preparation, optimization, simulation, and backtesting.

## Configuration

All key parameters (tickers, date ranges, model hyperparameters, and user-defined views) are managed centrally in [src/portfolio_analyzer/config.py](src/portfolio_analyzer/config.py).

## Documentation & Support

- For detailed module documentation, see inline docstrings and comments in the [src/portfolio_analyzer](src/portfolio_analyzer) directory.
- For troubleshooting or feature requests, please open an issue on [GitHub](https://github.com/Vorckea/portfolio-analyzer/issues).

---

*This project is intended for research and educational purposes. Please review and validate all results before using in a production or investment context.*