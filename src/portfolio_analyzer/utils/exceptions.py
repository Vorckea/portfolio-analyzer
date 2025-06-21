"""Custom exception types for the Portfolio Analyzer application."""


class PortfolioAnalyzerError(Exception):
    """Base exception class for this application."""

    pass


class InputAlignmentError(PortfolioAnalyzerError):
    """Raised when input data (e.g., returns, covariance) cannot be aligned."""

    pass


class OptimizationError(PortfolioAnalyzerError):
    """Raised when a portfolio optimization routine fails to converge."""

    pass


class DataFetchingError(PortfolioAnalyzerError):
    """Raised when fetching external data fails."""

    pass
