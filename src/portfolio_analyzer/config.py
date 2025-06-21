from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List


@dataclass
class DataRangeConfig:
    end: datetime = field(default_factory=lambda: datetime.today())
    start: datetime = field(default_factory=lambda: datetime.today() - timedelta(days=5 * 365))


@dataclass
class OptimizationConfig:
    lambda_reg: float = 1.0
    max_weight_per_asset: float = 0.25
    min_weight_per_asset: float = 1e-4


@dataclass
class BlackLittermanConfig:
    """Parameters for the Black-Litterman model."""

    delta: float = 2.5
    tau: float = 0.05
    # How much to blend the market-implied returns when no views are available.
    equilibrium_blend_weight: float = 0.5
    # **NEW**: How much weight to give to historical momentum vs. BL valuation.
    # 0.0 = Pure Black-Litterman, 1.0 = Pure Historical Momentum.
    momentum_blend_weight: float = 0.2


class DistributionModel(Enum):
    NORMAL = "Normal"
    STUDENT_T = "Student's T"


@dataclass
class BacktestingConfig:
    """Parameters for the historical backtest."""

    initial_capital: float = 100000.0
    rebalance_frequency: str = "3M"  # e.g., '1M', '3M', '1Y'
    lookback_period_days: int = 3 * 365  # How much history to use for each optimization


@dataclass
class MonteCarloConfig:
    initial_value: float = 1_000_000
    num_simulations: int = 100_000
    time_horizon_years: int = 1
    distribution_type: DistributionModel = DistributionModel.NORMAL
    df_t_distribution: int = 5


@dataclass
class DCFConfig:
    """Assumptions for the more advanced 3-stage DCF model."""

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


@dataclass
class AppConfig:
    tickers: List[str] = field(
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
            "KNEBV.HE",
            "SHB-A.ST",
            "MULTI.OL",
            "NORCO.OL",
            "EPR.OL",
            "0P00000MVB.IR",
            "0P0001IMY8.F",
            "0P00000B0I",
        ]
    )
    use_dcf_views: bool = True

    date_range: DataRangeConfig = field(default_factory=DataRangeConfig)
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.04074
    ewma_span: int = 90
    mean_shrinkage_alpha: float = 0.2
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    black_litterman: BlackLittermanConfig = field(default_factory=BlackLittermanConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    dcf: DCFConfig = field(default_factory=DCFConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)

    def model_copy(self, deep: bool = True) -> "AppConfig":
        """Create a deep copy of the configuration model."""
        from copy import deepcopy

        return deepcopy(self) if deep else self.__class__(**self.__dict__)
