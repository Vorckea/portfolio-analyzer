import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config.config import AppConfig
from ..data.repository import Repository
from .base import ReturnEstimator


class DCF(ReturnEstimator):
    def __init__(
        self,
        tickers: List[str],
        risk_free_rate: float,
        repository: Repository,  # Injected
        config: AppConfig,  # Injected
        logger: Optional[logging.Logger] = None,  # Injected
    ):
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.repository = repository
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.dcf_returns = None

    def _fetch_data(self, ticker):
        try:
            info = self.repository.fetch_ticker_info(ticker)
            cashflow_df = self.repository.fetch_cashflow(ticker)
            data = {}

            def skip(reason):
                self.logger.info("Skipping %s: %s.", ticker, reason)
                return None

            if info.get("sector") == "Financial Services":
                return skip("Financial Sector company")

            data["beta"] = info.get("beta")
            data["mkt_cap"] = info.get("marketCap")
            data["total_debt"] = info.get("totalDebt", 0)
            data["shares_outstanding"] = info.get("sharesOutstanding")
            data["current_price"] = info.get("currentPrice")
            data["analyst_growth_estimate"] = info.get("earningsGrowth")

            if any(
                v is None
                for v in [
                    data["beta"],
                    data["mkt_cap"],
                    data["shares_outstanding"],
                    data["current_price"],
                ]
            ):
                return skip("Missing essential data (beta, mkt_cap, etc.)")
            if data["mkt_cap"] <= 0:
                return skip("Invalid market cap")
            if not (self.config.dcf.min_beta < data["beta"] < self.config.dcf.max_beta):
                return skip(
                    f"Beta ({data['beta']:.2f}) is outside configured range "
                    f"({self.config.dcf.min_beta} - {self.config.dcf.max_beta})"
                )

            if cashflow_df.empty or "Free Cash Flow" not in cashflow_df.index:
                return skip("Free Cash Flow data not available")

            recent_fcf = cashflow_df.loc["Free Cash Flow"].head(3).dropna()
            if recent_fcf.empty:
                return skip("Not enough FCF data points for a 3-year average")

            avg_fcf = recent_fcf.mean()
            if avg_fcf <= 0:
                return skip(f"Negative 3-year average FCF ({avg_fcf:,.0f})")

            data["fcf"] = avg_fcf
            return data

        except Exception:
            self.logger.exception(
                "Skipping %s due to an unexpected error during data fetching.", ticker
            )
            return None

    def _get_adaptive_growth_rate(self, data) -> float:
        analyst_growth = data.get("analyst_growth_estimate")
        if analyst_growth and isinstance(analyst_growth, (int, float)):
            return np.clip(
                analyst_growth,
                self.config.dcf.min_growth_rate,
                self.config.dcf.max_growth_rate,
            )
        return self.config.dcf.fallback_growth_rate

    def _calculate_wacc(self, data) -> float:
        cost_of_equity = self.risk_free_rate + data["beta"] * self.config.dcf.market_risk_premium
        equity_value = data["mkt_cap"]
        debt_value = data["total_debt"]
        total_value = equity_value + debt_value

        if total_value == 0:
            return cost_of_equity

        w_e = equity_value / total_value
        w_d = debt_value / total_value
        return (w_e * cost_of_equity) + (
            w_d * self.config.dcf.cost_of_debt * (1 - self.config.dcf.effective_tax_rate)
        )

    def _calculate_expected_return(self, ticker) -> float | None:
        data = self._fetch_data(ticker)
        if data is None:
            return None

        wacc = self._calculate_wacc(data)
        pv_fcf_list = []
        last_fcf = data["fcf"]
        current_year = 0
        stage1_growth_rate = self._get_adaptive_growth_rate(data)

        for i in range(1, self.config.dcf.high_growth_years + 1):
            current_year += 1
            last_fcf *= 1 + stage1_growth_rate
            pv_fcf_list.append(last_fcf / ((1 + wacc) ** current_year))

        growth_decline = (
            stage1_growth_rate - self.config.dcf.perpetual_growth_rate
        ) / self.config.dcf.fade_years
        current_growth_rate = stage1_growth_rate
        for i in range(1, self.config.dcf.fade_years + 1):
            current_year += 1
            current_growth_rate -= growth_decline
            last_fcf *= 1 + current_growth_rate
            pv_fcf_list.append(last_fcf / ((1 + wacc) ** current_year))

        terminal_fcf = last_fcf * (1 + self.config.dcf.perpetual_growth_rate)
        terminal_value = terminal_fcf / (wacc - self.config.dcf.perpetual_growth_rate)
        pv_terminal_value = terminal_value / ((1 + wacc) ** current_year)

        enterprise_value = sum(pv_fcf_list) + pv_terminal_value
        equity_value = enterprise_value - data["total_debt"]

        if equity_value <= 0 or data["shares_outstanding"] <= 0:
            return None

        intrinsic_value_per_share = equity_value / data["shares_outstanding"]
        current_price = data["current_price"]

        expected_return = (intrinsic_value_per_share / current_price) - 1
        log_return = np.log(1 + expected_return)
        return np.clip(log_return, -0.7, 0.7)

    def get_dcf_returns(self) -> pd.Series:
        self.logger.info("Calculating DCF returns for %d tickers...", len(self.tickers))
        results = {}
        for ticker in self.tickers:
            expected_return = self._calculate_expected_return(ticker)
            if expected_return is not None:
                results[ticker] = expected_return
            else:
                self.logger.info("No valid DCF return for %s, skipping.", ticker)
                results[ticker] = np.nan
        self.logger.info("Calculated DCF returns for %d tickers.", len(results))
        return pd.Series(results, dtype=float).fillna(0.0)

    def get_returns(self) -> pd.Series:
        if self.dcf_returns is None:
            self.dcf_returns = self.get_dcf_returns()
        return self.dcf_returns
