"""Return estimator module initialization.

This package provides a unified interface for various return estimation models used in portfolio
analysis, including Black-Litterman, CAPM, DCF, EWMA, and blended approaches. All estimators inherit
from the ReturnEstimator base class and can be used interchangeably for portfolio optimization and
risk analysis.

Available estimators:
    - BlackLitterman: Black-Litterman model with support for user views and market equilibrium.
    - BlendedReturn: Combines multiple return estimation methods.
    - CAPM: Capital Asset Pricing Model.
    - ConstantReturn: Returns a constant expected return.
    - DCF: Discounted Cash Flow-based return estimation.
    - EWMA: Exponentially Weighted Moving Average for mean estimation.
    - FillNaNReturn: Handles missing values in return series.
    - ReturnEstimator: Abstract base class for all return estimators.
"""

from .base import ReturnEstimator
from .black_litterman import BlackLitterman
from .blended import BlendedReturn
from .capm import CAPM
from .constant_return import ConstantReturn
from .dcf import DCF
from .ewma import EWMA
from .nan_filler import FillNaNReturn

__all__ = [
    "BlackLitterman",
    "BlendedReturn",
    "CAPM",
    "ConstantReturn",
    "DCF",
    "EWMA",
    "FillNaNReturn",
    "ReturnEstimator",
]
