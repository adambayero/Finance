from utils import linear_interpolation, log_interpolation, spline_interpolation, actual_360
from market_data import MarketDeposit, MarketFuture, MarketSwap, MarketBond, MarketZeroCouponBond, MarketFixedRateBond, MarketFloatingRateBond, MarketInflationLinkedBond, InflationCurve
from models import display_bootstrap_result, display_adjusted_curve
from market_data import instruments
from models import create_curve
from datetime import date

def main():
    curves = [
        create_curve(linear_interpolation, instruments),
        create_curve(log_interpolation, instruments),
        create_curve(spline_interpolation, instruments)
    ]
    display_bootstrap_result(curves, ["linear", "log", "spline"])
    display_adjusted_curve(curves, ["linear", "log", "spline"])

    valuation_date = date(2025, 4, 25)

    cpi_initial = 108.5

    cpi_times = [0, 1, 2, 3, 4, 5]
    cpi_values = [108.35, 110.1, 112.0, 114.2, 116.5, 118.9]
    inflation_curve = InflationCurve(cpi_times, cpi_values)

    forward_curve = create_curve(log_interpolation, instruments)
    zero_coupon_bond = MarketZeroCouponBond( price=80.0, valuation_date=valuation_date, day_count=actual_360, months=60, nominal=100, maturity=5.0)
    fixed_bond = MarketFixedRateBond( price=98.0, valuation_date=valuation_date, day_count=actual_360, months=5 * 12, nominal=100, coupon_interval=12, coupon_rate=0.02)
    floating_bond = MarketFloatingRateBond( price=101.0, valuation_date=valuation_date, day_count=actual_360, months=3 * 12, nominal=100, coupon_interval=3, spread=0.0025, forward_curve=forward_curve)
    inflation_bond = MarketInflationLinkedBond( price=102.0, valuation_date=valuation_date, day_count=actual_360, months=5 * 12, nominal=100, coupon_interval=12, coupon_rates=[0.01] * 5, inflation_curve=inflation_curve, cpi_initial=cpi_initial, lag_months=3)

    zero_coupon_bond.display_metrics()
    fixed_bond.display_metrics()
    floating_bond.display_metrics()
    inflation_bond.display_metrics()
    return 0

main()