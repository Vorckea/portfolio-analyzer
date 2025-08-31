"""Configuration module for the portfolio analyzer application."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, ClassVar

__all__ = [
    "DataRangeConfig",
    "OptimizationConfig",
    "BlackLittermanConfig",
    "DistributionModel",
    "BacktestingConfig",
    "MonteCarloConfig",
    "DCFConfig",
    "AppConfig",
]

DEFAULT_START_DATE = datetime.today() - timedelta(days=5 * 365)
DEFAULT_END_DATE = datetime.today()


@dataclass
class DataRangeConfig:
    """Date range for fetching historical data."""

    end: datetime = field(default_factory=lambda: DEFAULT_END_DATE)
    start: datetime = field(default_factory=lambda: DEFAULT_START_DATE)


DEFAULT_MAX_WEIGHT_PER_ASSET = 0.25
DEFAULT_MIN_WEIGHT_PER_ASSET = 1e-4


@dataclass
class OptimizationConfig:
    """Portfolio optimization parameters."""

    lambda_reg: float = 1.0
    max_weight_per_asset: float = DEFAULT_MAX_WEIGHT_PER_ASSET
    min_weight_per_asset: float = DEFAULT_MIN_WEIGHT_PER_ASSET


@dataclass
class BlackLittermanConfig:
    """Configuration for the Black-Litterman model parameters.

    This class defines the parameters used in the Black-Litterman model,
    including the delta and tau values, equilibrium blend weight, and
    momentum blend weight. These parameters control how the model combines
    market-implied returns with user-defined views and historical momentum.
    The delta parameter represents the risk aversion coefficient, while tau
    is a scaling factor for the covariance matrix. The equilibrium blend
    weight determines how much to blend the market-implied returns when no
    views are available, and the momentum blend weight controls the balance
    between historical momentum and Black-Litterman valuation.
    The momentum blend weight allows for a flexible approach to portfolio
    construction, enabling users to adjust the influence of historical
    momentum versus the Black-Litterman valuation in the final portfolio
    construction.

    Attributes:
        delta (float): Risk aversion coefficient.
        tau (float): Scaling factor for the covariance matrix.
        equilibrium_blend_weight (float): Weight for market-implied returns when no views are
        available.
        momentum_blend_weight (float): Weight for historical momentum versus Black-Litterman
        valuation.

    """

    delta: float = 2.5
    tau: float = 0.05
    # **NEW**: How much weight to give to historical momentum vs. BL valuation.
    # 0.0 = Pure Black-Litterman, 1.0 = Pure Historical Momentum.
    momentum_blend_weight: float = 0.3


class DistributionModel(Enum):
    """Distribution models for Monte Carlo simulations."""

    NORMAL = "Normal"
    STUDENT_T = "Student's T"


@dataclass
class BacktestingConfig:
    """Backtesting parameters."""

    initial_capital: float = 100000.0
    rebalance_frequency: str = "3M"  # e.g., '1M', '3M', '1Y'
    lookback_period_days: int = 3 * 365


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation parameters.

    This class defines the parameters used in Monte Carlo simulations,
    including the initial value of the portfolio, the number of simulations,
    the time horizon in years, the type of distribution to use for returns,
    and the degrees of freedom for the Student's T distribution if selected.
    """

    initial_value: float = 1_000_000
    num_simulations: int = 100_000
    time_horizon_years: int = 1
    distribution_type: DistributionModel = DistributionModel.NORMAL
    df_t_distribution: int = 5


@dataclass
class DCFConfig:
    """Configuration for Discounted Cash Flow (DCF) model parameters.

    This class defines the parameters used in the DCF model, including
    the high growth phase duration, fallback growth rate, and perpetual
    growth rate. It also includes assumptions for the Weighted Average Cost
    of Capital (WACC), such as the market risk premium, effective tax rate,
    cost of debt, and the minimum required spread between WACC and perpetual
    growth rate. These parameters are crucial for accurately estimating the
    present value of future cash flows, which is the core of the DCF valuation
    method. The high growth phase represents the initial period of rapid
    growth for a company, while the transition phase allows for a gradual
    fade to a more stable, perpetual growth rate. The WACC assumptions are
    essential for discounting future cash flows to their present value,
    as they reflect the cost of capital used to finance the company's operations.
    The minimum required spread between WACC and perpetual growth rate is a
    safeguard to prevent unrealistic terminal values, ensuring that the
    DCF model remains grounded in reasonable financial assumptions.
    """

    # Stage 1: High Growth Phase
    high_growth_years: int = 5
    # **NEW**: Use analyst estimates when available, otherwise use this fallback.
    fallback_growth_rate: float = 0.05
    # **NEW**: Add caps to prevent extreme analyst estimates.
    min_growth_rate: float = 0.01
    max_growth_rate: float = 0.15

    # Stage 2: Transition Phase
    fade_years: int = 5  # Number of years for growth to fade to perpetual rate

    perpetual_growth_rate: float = 0.025

    # WACC Assumptions
    market_risk_premium: float = 0.055
    effective_tax_rate: float = 0.21
    cost_of_debt: float = 0.05

    # **NEW**: Minimum required spread between WACC and perpetual growth rate.
    # This prevents the terminal value from becoming unrealistically large.
    wacc_g_spread: float = 0.03

    # Validation parameters
    min_beta: float = 0.05
    max_beta: float = 3.5


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04074
DEFAULT_EWMA_SPAN = 90
DEFAULT_MEAN_SHRINKAGE_ALPHA = 0.2


@dataclass
class AppConfig:
    """Application configuration for the portfolio analyzer.

    This class holds the configuration settings for the portfolio analyzer application,
    including the list of tickers to analyze, date range for historical data,
    and various financial model parameters such as optimization settings,
    Black-Litterman model parameters, Monte Carlo simulation settings, DCF model parameters,
    and backtesting configurations.
    It provides a structured way to manage the application's settings and allows for easy
    customization of the analysis parameters without modifying the core logic of the application.
    """

    tickers: list[str] = field(
        default_factory=lambda: [
            "NORBT.OL",
            "SAAB-B.ST",
            "DNB.OL",
            "TEL.OL",
            "ORK.OL",
            "SAMPO.HE",
            "STB.OL",
            "SEA1.OL",
            "SALME.OL",
            "KOG.OL",
            "SHB-A.ST",
            "MULTI.OL",
            "NORCO.OL",
            "EPR.OL",
            "SWED-A.ST",
            "ORNBV.HE",
            "MAERSK-B.CO",
            "BOL.ST",
            "VALMT.HE",
            "0P00000MVB.IR",
            "0P0001IMY8.F",
            "0P00000B0I",
            "VAR.OL",
        ]
    )
    use_dcf_views: bool = True
    date_range: DataRangeConfig = field(default_factory=DataRangeConfig)
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    risk_free_rate: float = RISK_FREE_RATE
    ewma_span: int = DEFAULT_EWMA_SPAN
    mean_shrinkage_alpha: float = DEFAULT_MEAN_SHRINKAGE_ALPHA
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    black_litterman: BlackLittermanConfig = field(default_factory=BlackLittermanConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    dcf: DCFConfig = field(default_factory=DCFConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)

    _instance: ClassVar["AppConfig | None"] = None

    @classmethod
    def get_instance(cls) -> "AppConfig":
        """Get the singleton instance of the AppConfig.

        Returns:
            AppConfig: The application configuration.

        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, config: "AppConfig"):
        """Explicitly set the singleton instance (useful for testing or custom configs)."""
        cls._instance = config

    @staticmethod
    def default() -> "AppConfig":
        config = AppConfig.get_instance()
        return config.model_copy(deep=False)

    def model_copy(self, deep: bool = True) -> "AppConfig":
        """Create a deep copy of the AppConfig instance.

        This method allows for creating a copy of the AppConfig instance,
        which can be useful for creating temporary configurations for
        backtesting or simulations without modifying the original configuration.
        This is particularly useful in scenarios where you want to run
        multiple simulations or backtests with different parameters while
        keeping the original configuration intact.

        Args:
            deep (bool, optional): Whether to create a deep copy of the configuration.
            If True, a new instance is created with the same parameters. Defaults to True.

        Returns:
            AppConfig: A new instance of AppConfig with the same parameters.

        """
        from copy import deepcopy

        return deepcopy(self) if deep else self.__class__(**self.__dict__)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
