from utils import actual_360
from datetime import date
from instruments import Deposit, Future, Swap
import pandas as pd

valuation_date = date(2025, 4, 25)

rate_deposit = 0.0217

future_prices = [0.01805, 0.01715, 0.0165]
future_maturities = [date(2025, 8, 18), date(2025, 10, 13), date(2025, 12, 15)]

swap_rates = [0.01932, 0.0187, 0.01936, 0.02095, 0.02234, 0.02394, 0.02524, 0.02384]
swap_years = [1, 2, 3, 5, 7, 10, 15, 30]

instruments = []

instruments.append([Deposit(actual_360, 3), rate_deposit, valuation_date])
for price, maturity in zip(future_prices, future_maturities):
    instruments.append([Future(actual_360, 3), price, valuation_date, maturity])
for rate, year in zip(swap_rates, swap_years):
    instruments.append([Swap(actual_360, 12 * year, 12), rate, valuation_date])


import numpy as np
import pandas as pd

option_maturities = [1, 2, 3, 5, 10, 15, 30]  # Maturities from 1 to 5 years
swap_tenors = [5, 10, 15, 20, 25]  # Tenors from 1 to 20 years
strikes = [0.02 + 0.005 * i for i in range(1, 21)]  # Strikes from 0.005 to 0.1 in steps of 0.005

data_swaption = []

np.random.seed(42)
for T in option_maturities:
    for tenor in swap_tenors:
        for K in strikes:
            forward = 0.022 + 0.0015 * (tenor / 5) + 0.0005 * (T - 1)
            vol = 0.18 + 0.02 * abs(K - forward)
            price = 10000 + 1000 * vol  # purement fictif
            discount = 0.95 - 0.01 * (T / 5)
            data_swaption.append({
                "option_maturity": T,
                "swap_tenor": tenor,
                "strike": K,
                "price": price,
                "forward_rate": forward,
                "discount_factor": discount,
                "implied_vol": vol,
            })

df_swaption = pd.DataFrame(data_swaption)

