from bootstrapping import Deposit, Future, Swap, Curve, display_bootstrap_result, display_adjusted_curve
from bonds import Bond
from datetime import date
from utils import actual_360, linear_interpolation, log_interpolation, spline_interpolation, display_grid, nelson_siegel, nelson_siegel_svensson

def bootstrapping():
    valuation_date = date(2025, 4, 25)

    rate_deposit = 0.0217

    future_prices = [0.01805, 0.01715, 0.0165]
    future_maturities = [date(2025, 8, 18), date(2025, 10, 13), date(2025, 12, 15)]

    swap_rates = [0.01932, 0.0187, 0.01936, 0.02095, 0.02234, 0.02394, 0.02524, 0.02384]
    swap_years = [1, 2, 3, 5, 7, 10, 15, 30]

    instruments = []

    instruments.append(Deposit(rate_deposit, valuation_date, actual_360, 3))
    for price, maturity in zip(future_prices, future_maturities):
        instruments.append(Future(price, valuation_date, actual_360, 3, maturity))
    for rate, year in zip(swap_rates, swap_years):
        instruments.append(Swap(rate, valuation_date, actual_360, 12 * year, 12))

    curves = [Curve(linear_interpolation), 
              Curve(log_interpolation), 
              Curve(spline_interpolation)]
    
    for curve in curves:
        curve.bootstrap(instruments)
        curve.compute_curve()
        curve.fit(nelson_siegel)
        curve.adjust_curve(nelson_siegel)
        curve.fit(nelson_siegel_svensson)
        curve.adjust_curve(nelson_siegel_svensson)

    display_bootstrap_result(curves, ["Linear", "Log", "Spline"])
    display_adjusted_curve(curves, ["Linear", "Log", "Spline"])

def bond_example():
    bond = Bond(82.44, date(2025, 4, 25), actual_360, 7 * 12, 100, [i for i in range(1, 8)], [0.00625] * 7, [0.0217] * 7)

    bond.display_metrics()

    bond.display_price_graph()
    bond.display_durations_graph()
    bond.display_YTM_graph()

bond_example()