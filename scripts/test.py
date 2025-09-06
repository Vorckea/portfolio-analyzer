from portfolio_analyzer.data.providers.yfinance import YahooFinanceDataProvider

provider = YahooFinanceDataProvider()
data = provider.fetch_price_history(
    ["AAPL", "MSFT", "GOOGL"], start="2025-01-01", end="2025-03-01", frequency="1h"
)
print(data.prices.head())
print(data.volume.head())
