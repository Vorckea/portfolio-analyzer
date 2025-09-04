import pandas as pd
import yfinance as yf

cashflow_df = yf.Ticker("AAPL").cashflow
print(cashflow_df.head())

free_cash_flow = cashflow_df.loc["Free Cash Flow"]
print(free_cash_flow.head())
print(type(free_cash_flow))
