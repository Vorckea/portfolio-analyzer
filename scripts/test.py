import yfinance as yf

from portfolio_analyzer.core.models import SymbolInfo


def fetch_symbol_info(symbol: str) -> SymbolInfo:
    info = yf.Ticker(symbol).info
    if not info:
        return SymbolInfo(**{})
    return SymbolInfo(**info)


print(fetch_symbol_info("AAPL"))
