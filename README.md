# Portfolio Analyzer

This project provides a toolkit for advanced portfolio optimization and risk analysis using modern financial modeling techniques like Black-Litterman and Monte Carlo simulation.

## Features

- **Data Pipeline**: Fetches and processes market data for specified assets.
- **Robust Estimation**: Uses EWMA and Ledoit-Wolf shrinkage for stable covariance matrices.
- **Black-Litterman Model**: Blends market equilibrium returns with user-defined views.
- **Portfolio Optimization**: Maximizes Sharpe Ratio with constraints and L2 regularization.
- **Risk Analysis**: Simulates future portfolio performance using Monte Carlo methods.

## Setup and Installation

This project uses Poetry for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd portfolio-analyzer
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

## Usage

The main analysis workflow is in `notebooks/portfolio_analysis.ipynb`.

1.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

2.  **Launch Jupyter Lab:**
    ```bash
    jupyter lab
    ```

3.  Open and run the cells in `notebooks/portfolio_analysis.ipynb`.

## Configuration

All key parameters (tickers, date ranges, model hyperparameters) can be adjusted in `src/portfolio_analyzer/config.py`.