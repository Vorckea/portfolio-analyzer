from datetime import datetime

import numpy as np
import pandas as pd
import pandera.errors as pe
import pytest
from pydantic import ValidationError

from portfolio_analyzer.data.schema import (
    PRICE_HISTORY_SCHEMA,
    VOLUME_HISTORY_SCHEMA,
    PriceHistory,
)


def _make_price_df():
    """Create a small valid price DataFrame for tests."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.5], "MSFT": [150.0, 151.2, 152.3]},
        index=idx,
    )
    df.index.name = "Datetime"
    return df


def _make_volume_df(integral=True):
    """Create a small volume DataFrame; set integral=False to include floats."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    if integral:
        arr = [[1000, 1100], [1200, 1300], [1250, 1400]]
    else:
        arr = [[1000.5, 1100.2], [1200.3, 1300.7], [1250.1, 1400.9]]
    df = pd.DataFrame(arr, index=idx, columns=["AAPL", "MSFT"])
    df.index.name = "Datetime"
    return df


def test_price_schema_accepts_valid_df():
    """Valid price DataFrame should pass the pandera schema."""
    df = _make_price_df()
    validated = PRICE_HISTORY_SCHEMA.validate(df)
    pd.testing.assert_frame_equal(validated, df)


def test_price_schema_rejects_unsorted_index():
    """An unsorted index should fail validation."""
    df = _make_price_df()
    df = df.sort_index(ascending=False)
    with pytest.raises(pe.SchemaError):
        PRICE_HISTORY_SCHEMA.validate(df)


def test_price_schema_rejects_negative_or_null_prices():
    """Negative or null price values should fail validation."""
    df = _make_price_df()
    df.loc[df.index[0], "AAPL"] = -1.0
    with pytest.raises(pe.SchemaError):
        PRICE_HISTORY_SCHEMA.validate(df)

    df = _make_price_df()
    df.loc[df.index[0], "AAPL"] = None
    with pytest.raises(pe.SchemaError):
        PRICE_HISTORY_SCHEMA.validate(df)


def test_volume_schema_accepts_integral_values():
    """Integral volume values should pass validation."""
    df = _make_volume_df(integral=True)
    validated = VOLUME_HISTORY_SCHEMA.validate(df)
    pd.testing.assert_frame_equal(validated, df)


def test_volume_schema_rejects_non_integral_values():
    """Non-integral float volume values should fail the integral check."""
    df = _make_volume_df(integral=False)
    with pytest.raises(pe.SchemaError):
        VOLUME_HISTORY_SCHEMA.validate(df)


def test_pricehistory_pydantic_model_validation():
    """PriceHistory model should accept valid data and reject invalid price frames."""
    prices = _make_price_df()
    volume = _make_volume_df(integral=True)

    ph = PriceHistory(
        prices=prices,
        volume=volume,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 3),
        frequency="D",
    )
    assert list(ph.assets) == ["AAPL", "MSFT"]

    bad_prices = prices.copy()
    bad_prices.index.name = "NotDatetime"
    # Note: If the DataFrame index name is incorrect, validation may fail at different stages.
    # - If pandera's schema validation runs first, a pandera.SchemaError is raised directly.
    # - If pydantic's Annotated validator wraps the error, a pydantic.ValidationError is raised.
    # This test accepts either error type to accommodate the current (library-dependent) validation order.
    with pytest.raises((ValidationError, pe.SchemaError)):
        PriceHistory(
            prices=bad_prices,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 3),
            frequency="D",
        )


def test_pricehistory_pydantic_model_pct_change_returns():
    """Test the pct_change_returns property of PriceHistory."""
    prices = _make_price_df()
    ph = PriceHistory(
        prices=prices,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 3),
        frequency="D",
    )
    expected_pct_changes = prices.pct_change().dropna(how="all")
    pd.testing.assert_frame_equal(ph.pct_change_returns, expected_pct_changes)


def test_pricehistory_pydantic_model_log_returns():
    """Test the log_returns property of PriceHistory."""
    prices = _make_price_df()
    ph = PriceHistory(
        prices=prices,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 3),
        frequency="D",
    )
    expected_log_returns = (ph.pct_change_returns + 1).apply(np.log)
    pd.testing.assert_frame_equal(ph.log_returns, expected_log_returns)
