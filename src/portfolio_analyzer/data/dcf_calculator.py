import logging

import numpy as np
import yfinance as yf

from portfolio_analyzer.config.config import DCFConfig

logger = logging.getLogger(__name__)


class DCFCalculator:
    """An advanced, adaptive 3-stage DCF model with improved validation and logging.

    This class fetches financial data for a given stock ticker, validates it
    against configured thresholds, and calculates an intrinsic value using a
    Discounted Cash Flow (DCF) model. The result is used to generate a view
    for the Black-Litterman model.
    """

    def __init__(self, ticker_symbol: str, config: DCFConfig, risk_free_rate: float):
        """Initialize the DCFCalculator."""
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.config = config
        self.risk_free_rate = risk_free_rate
        self._data = {}

    def _fetch_data(self) -> bool:
        """Fetch and validate required financial data for the DCF calculation.

        It checks for essential data points, validates sector, beta, and FCF,
        and logs the specific reason for skipping a ticker if validation fails.

        Returns:
            bool: True if data was fetched and validated successfully, False otherwise.

        """
        try:
            info = self.ticker.info

            # Helper for clear logging
            def skip(reason):
                logger.info("Skipping %s: %s.", self.ticker_symbol, reason)
                return False

            # 1. Sector validation
            if info.get("sector") == "Financial Services":
                return skip("Financial Sector company")

            # 2. Fetch all data points
            self._data["beta"] = info.get("beta")
            self._data["mkt_cap"] = info.get("marketCap")
            self._data["total_debt"] = info.get("totalDebt", 0)
            self._data["shares_outstanding"] = info.get("sharesOutstanding")
            self._data["current_price"] = info.get("currentPrice")
            self._data["analyst_growth_estimate"] = info.get("earningsGrowth")

            # 3. Essential data validation
            if any(
                v is None
                for v in [
                    self._data["beta"],
                    self._data["mkt_cap"],
                    self._data["shares_outstanding"],
                    self._data["current_price"],
                ]
            ):
                return skip("Missing essential data (beta, mkt_cap, etc.)")
            if self._data["mkt_cap"] <= 0:
                return skip("Invalid market cap")
            if not (self.config.min_beta < self._data["beta"] < self.config.max_beta):
                return skip(
                    f"Beta ({self._data['beta']:.2f}) is outside configured range "
                    f"({self.config.min_beta} - {self.config.max_beta})"
                )

            # 4. Free Cash Flow validation (using 3-year average)
            cashflow_df = self.ticker.cashflow
            if cashflow_df.empty or "Free Cash Flow" not in cashflow_df.index:
                return skip("Free Cash Flow data not available")

            recent_fcf = cashflow_df.loc["Free Cash Flow"].head(3).dropna()
            if recent_fcf.empty:
                return skip("Not enough FCF data points for a 3-year average")

            avg_fcf = recent_fcf.mean()
            if avg_fcf <= 0:
                return skip(f"Negative 3-year average FCF ({avg_fcf:,.0f})")

            self._data["fcf"] = avg_fcf
            return True

        except Exception:
            logger.exception(
                "Skipping %s due to an unexpected error during data fetching.", self.ticker_symbol
            )
            return False

    def _get_adaptive_growth_rate(self) -> float:
        """Determine the initial growth rate for the DCF model.

        It prioritizes the analyst's growth estimate if available and valid,
        clipping it within a configured min/max range. Otherwise, it falls
        back to a default growth rate.

        Returns:
            float: The determined growth rate for Stage 1 of the DCF model.

        """
        analyst_growth = self._data.get("analyst_growth_estimate")
        if analyst_growth and isinstance(analyst_growth, (int, float)):
            return np.clip(analyst_growth, self.config.min_growth_rate, self.config.max_growth_rate)
        return self.config.fallback_growth_rate

    def _calculate_wacc(self) -> float:
        """Calculates the Weighted Average Cost of Capital (WACC)."""
        cost_of_equity = self.risk_free_rate + self._data["beta"] * self.config.market_risk_premium
        equity_value = self._data["mkt_cap"]
        debt_value = self._data["total_debt"]
        total_value = equity_value + debt_value

        if total_value == 0:
            return cost_of_equity

        w_e = equity_value / total_value
        w_d = debt_value / total_value
        return (w_e * cost_of_equity) + (
            w_d * self.config.cost_of_debt * (1 - self.config.effective_tax_rate)
        )

    def calculate_expected_return(self) -> float | None:
        """Calculates expected return using the 3-stage DCF model."""
        if not self._fetch_data():
            return None

        wacc = self._calculate_wacc()

        # **IMPROVEMENT**: Enforce a sufficient spread between WACC and perpetual growth rate.
        if (wacc - self.config.perpetual_growth_rate) < self.config.wacc_g_spread:
            logger.info(
                "Skipping %s: WACC (%.2f%%) is too close to perpetual growth rate (%.2f%%).",
                self.ticker_symbol,
                wacc * 100,
                self.config.perpetual_growth_rate * 100,
            )
            return None

        pv_fcf_list = []
        last_fcf = self._data["fcf"]
        current_year = 0
        stage1_growth_rate = self._get_adaptive_growth_rate()

        # Stage 1
        for i in range(1, self.config.high_growth_years + 1):
            current_year += 1
            last_fcf *= 1 + stage1_growth_rate
            pv_fcf_list.append(last_fcf / ((1 + wacc) ** current_year))

        # Stage 2
        growth_decline = (
            stage1_growth_rate - self.config.perpetual_growth_rate
        ) / self.config.fade_years
        current_growth_rate = stage1_growth_rate
        for i in range(1, self.config.fade_years + 1):
            current_year += 1
            current_growth_rate -= growth_decline
            last_fcf *= 1 + current_growth_rate
            pv_fcf_list.append(last_fcf / ((1 + wacc) ** current_year))

        # Stage 3
        terminal_fcf = last_fcf * (1 + self.config.perpetual_growth_rate)
        terminal_value = terminal_fcf / (wacc - self.config.perpetual_growth_rate)
        pv_terminal_value = terminal_value / ((1 + wacc) ** current_year)

        enterprise_value = sum(pv_fcf_list) + pv_terminal_value
        equity_value = enterprise_value - self._data["total_debt"]

        if equity_value <= 0 or self._data["shares_outstanding"] <= 0:
            return None

        intrinsic_value_per_share = equity_value / self._data["shares_outstanding"]
        current_price = self._data["current_price"]

        expected_return = (intrinsic_value_per_share / current_price) - 1
        return np.clip(expected_return, -0.5, 1.0)
