from utils import actual_360
from market_data import MarketDeposit, MarketFuture, MarketSwap
from datetime import date

valuation_date = date(2025, 4, 25)

rate_deposit = 0.0217

future_prices = [0.01805, 0.01715, 0.0165]
future_maturities = [date(2025, 8, 18), date(2025, 10, 13), date(2025, 12, 15)]

swap_rates = [0.01932, 0.0187, 0.01936, 0.02095, 0.02234, 0.02394, 0.02524, 0.02384]
swap_years = [1, 2, 3, 5, 7, 10, 15, 30]

instruments = []

instruments.append(MarketDeposit(rate_deposit, valuation_date, actual_360, 3))
for price, maturity in zip(future_prices, future_maturities):
    instruments.append(MarketFuture(price, valuation_date, actual_360, 3, maturity))
for rate, year in zip(swap_rates, swap_years):
    instruments.append(MarketSwap(rate, valuation_date, actual_360, 12 * year, 12))