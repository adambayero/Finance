from bootstrapping import Deposit, Future, Swap, Curve, display_bootstrap_result, display_adjusted_curve, InflationCurve
from bonds import Bond, FixedRateBond, ZeroCouponBond, FloatingRateBond, InflationLinkedBond
from datetime import date, timedelta
from utils import actual_360, linear_interpolation, log_interpolation, spline_interpolation, display_grid, nelson_siegel, nelson_siegel_svensson

def bootstrapping_example():
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

    curves = [Curve(linear_interpolation, instruments), 
              Curve(log_interpolation, instruments), 
              Curve(spline_interpolation, instruments)]

    display_bootstrap_result(curves, ["Linear", "Log", "Spline"])
    display_adjusted_curve(curves, ["Linear", "Log", "Spline"])

def bond_example():
    valuation_date = date(2025, 4, 25)

    """ois_deposit_rate = 0.0380

    ois_swap_rates = [ 0.0382, 0.0383, 0.0384, 0.0386, 0.0388, 0.0390, 0.0392, 0.0395, 0.0400]

    ois_swap_months = [1, 2, 3, 6, 9, 12, 24, 36, 60]

    instruments = []
    instruments.append(Deposit(ois_deposit_rate, valuation_date, actual_360, 1))

    for rate, months in zip(ois_swap_rates, ois_swap_months):
        instruments.append(Swap(rate, valuation_date, actual_360, months, 12))
        
    discount_curve = Curve(linear_interpolation, instruments)"""

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

    forward_curve = Curve(log_interpolation, instruments)
    discount_curve = forward_curve

    cpi_dates = [date(2025, 1, 31) + timedelta(days=30 * i) for i in range(60)]
    cpi_values = [108.0 + 0.2 * i for i in range(len(cpi_dates))]
    inflation_curve = InflationCurve(cpi_dates, cpi_values)


    print("\n🟩 Fixed Rate Bond")
    bond_fixed = FixedRateBond(
        price=82.44,
        valuation_date=valuation_date,
        day_count=actual_360,
        months=7 * 12,
        nominal=100,
        coupon_dates=[i for i in range(1, 8)],
        coupon_rates=[0.00625] * 7,
        discount_curve=discount_curve
    )
    bond_fixed.display_metrics()

    print("\n🟦 Zero Coupon Bond")
    bond_zc = ZeroCouponBond(
        price=80.0,
        valuation_date=valuation_date,
        day_count=actual_360,
        months=5 * 12,
        nominal=100,
        maturity=5,
        discount_curve=discount_curve
    )
    bond_zc.display_metrics()

    print("\n🟨 Floating Rate Bond")
    bond_float = FloatingRateBond(
        price=101.0,
        valuation_date=valuation_date,
        day_count=actual_360,
        months=3 * 12,
        nominal=100,
        coupon_dates=[i for i in range(1, 4)],
        spreads=[0.001] * 3,
        discount_curve=discount_curve,
        forward_curve=forward_curve
    )
    bond_float.display_metrics()

    print("\n🟥 Inflation Linked Bond")
    bond_infl = InflationLinkedBond(
        price=102.0,
        valuation_date=valuation_date,
        day_count=actual_360,
        months=5 * 12,
        nominal=100,
        coupon_dates=[i for i in range(1, 6)],
        real_coupon_rates=[0.01] * 5,
        discount_curve=discount_curve,
        inflation_curve=inflation_curve,
        cpi_initial=108.0
    )
    bond_infl.display_metrics()

bond_example()