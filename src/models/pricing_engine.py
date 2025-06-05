from instruments import Deposit, Future, Swap, Bond, Swaption
from utils import display_tabular
from market_data import ZeroCouponCurve, InflationCurve
from scipy.stats import norm
from typing import Callable
from scipy.optimize import root
import numpy as np
from numpy.typing import NDArray
from datetime import date
from dateutil.relativedelta import relativedelta

class DepositPricer:
    def __init__(self):
        self.maturity = None
        
    def compute_maturity(self, deposit: Deposit, valuation_date: date) -> None:
        self.maturity = deposit.day_count(valuation_date, valuation_date + relativedelta(months=deposit.months))
    
    def price(self, deposit: Deposit, rate, valuation_date) -> float:
        self.compute_maturity(deposit, valuation_date)

        return 1 / (1 + rate * self.maturity)

class FuturePricer:
    def __init__(self):
        self.maturity = None
        self.start = None

    def compute_maturity(self, future: Future, valuation_date: date, maturity_date: date) -> None:
        self.maturity = future.day_count(valuation_date, maturity_date)
        self.start = future.day_count(valuation_date, maturity_date - relativedelta(months=future.months))

    def price(self, future: Future, rate: float, valuation_date: date, maturity_date: date, curve: dict, interpolation: Callable) -> float:
        self.compute_maturity(future, valuation_date, maturity_date)

        D_start = interpolation(curve, self.start)
        return D_start / (1 + (self.maturity - self.start) * rate)

class SwapPricer:
    def __init__(self):
        self.coupon_maturities = []
        self.maturity = None

    def compute_maturities(self, swap: Swap, valuation_date: date) -> None:
        self.coupon_maturities = []
        self.maturity = valuation_date + relativedelta(months=swap.months)
        current_date = valuation_date
        while current_date <= self.maturity:
            self.coupon_maturities.append(swap.day_count(valuation_date, current_date))
            current_date += relativedelta(months=swap.coupon_interval)
        self.maturity = self.coupon_maturities[-1]

    def price(self, swap: Swap, rate: float, valuation_date: date, curve: dict, interpolation: Callable) -> tuple[list[float], NDArray[np.float64]]:
        self.compute_maturities(swap, valuation_date)

        unknown_dates = [t for t in self.coupon_maturities if t > max(curve.keys())]

        def equation(x):
            local_curve = curve.copy()
            for t, d in zip(unknown_dates, x):
                local_curve[t] = d
            zero_curve = lambda t: interpolation(local_curve, t)
            left_curve = curve.copy()
            right_curve = {unknown_dates[i]: x[i] for i in range(1, len(x))}
            res = []

            pv_fixed = rate * sum((self.coupon_maturities[i] - self.coupon_maturities[i-1]) * zero_curve(t) for i, t in enumerate(self.coupon_maturities) if i > 0)
            pv_float = 1 - zero_curve(self.coupon_maturities[-1])
            res.append(pv_fixed - pv_float)

            
            if len(x) > 1:
                interp_val = interpolation(curve | right_curve, unknown_dates[0])
                res.append(x[0] - interp_val)

            for i in range(1, len(x) - 1):
                left_curve[unknown_dates[i - 1]] = x[i - 1]
                del right_curve[unknown_dates[i]]
                interp_val = interpolation(left_curve | right_curve, unknown_dates[i])
                res.append(x[i] - interp_val)

            return np.array(res)

        guess = [list(curve.values())[-1]] * len(unknown_dates)
        solution = root(equation, guess)
        return unknown_dates, solution.x

class BondPricer:
    def __init__(self):
        self.price = None
        self.YTM = None
        self.duration = None
        self.modified_duration = None
        self.sensitivity = None
        self.convexity = None
        self.DV01 = None
        self.coupon_maturities = []

    def price_bond(self, bond: Bond, discount_curve: dict[float, float]) -> float:
        return sum(cf * discount_curve[t] for cf, t in zip(bond.cashflows, self.coupon_maturities))

    def compute_YTM(self, bond: Bond) -> float:
        def price_at_rate(r):
            return sum(cf / (1 + r)**t for cf, t in zip(bond.cashflows, self.coupon_maturities))

        def objective(r):
            return price_at_rate(r) - self.price

        return root(objective, 0.05).x[0]

    def compute_duration(self, bond: Bond) -> float:
        return sum(t * cf / (1 + self.YTM)**t for cf, t in zip(bond.cashflows, self.coupon_maturities)) / self.price

    def compute_modified_duration(self):
        return self.duration / (1 + self.YTM)

    def compute_sensitivity(self):
        return self.modified_duration

    def compute_convexity(self, bond: Bond) -> float:
        return sum(t * (t + 1) * cf / (1 + self.YTM)**t for cf, t in zip(bond.cashflows, self.coupon_maturities)) / (self.price * (1 + self.YTM)**2)

    def compute_DV01(self):
        return -self.modified_duration * self.price / 10000

    def compute_accrued_interest(self, bond: Bond, t: float) -> float:
        past = [m for m in self.coupon_maturities if m <= t]
        future = [m for m in self.coupon_maturities if m > t]
        if not past or not future:
            return 0.0
        t_last = past[-1]
        t_next = future[0]
        i = self.coupon_maturities.index(t_next)
        return bond.cashflows[i] * ((t - t_last) / (t_next - t_last))

    def compute_dirty_price(self, bond: Bond, t: float) -> float:
        return self.price + self.compute_accrued_interest(bond, t)

    def compute_clean_price(self) -> float:
        return self.price

    def compute_metrics(self, bond: Bond) -> None:
        self.YTM = self.compute_YTM(bond)
        self.duration = self.compute_duration(bond)
        self.modified_duration = self.compute_modified_duration()
        self.sensitivity = self.compute_sensitivity()
        self.convexity = self.compute_convexity(bond)
        self.DV01 = self.compute_DV01()

    def display_metrics(self, bond: Bond) -> None:
        self.compute_metrics(bond)
        display_tabular([
            ["Price", self.price],
            ["YTM", self.YTM],
            ["Duration", self.duration],
            ["Modified Duration", self.modified_duration],
            ["Sensitivity", self.sensitivity],
            ["Convexity", self.convexity],
            ["DV01", self.DV01]
        ], headers=["Metric", "Value"])

class ZeroCouponBondPricer(BondPricer):
    def compute_cashflows(self, bond: Bond, maturity: float):
        bond.cashflows = [bond.nominal]
        self.coupon_maturities = [maturity]
        bond.name = "Zero Coupon Bond"

class FixedRateBondPricer(BondPricer):
    def compute_cashflows(self, bond: Bond, valuation_date: date):
        self.coupon_maturities = []
        maturity_date = valuation_date + relativedelta(months=bond.months)
        current_date = valuation_date + relativedelta(months=bond.coupon_interval)

        while current_date <= maturity_date:
            self.coupon_maturities.append(bond.day_count(valuation_date, current_date))
            current_date += relativedelta(months=bond.coupon_interval)

        bond.cashflows = [bond.coupon_rate * bond.nominal] * len(self.coupon_maturities)
        bond.cashflows[-1] += bond.nominal
        bond.name = "Fixed Rate Bond"

class FloatingRateBondPricer(BondPricer):
    def compute_cashflows(self, bond: Bond, valuation_date: date, forward_curve: ZeroCouponCurve, spread: float):
        self.coupon_maturities = []
        maturity_date = valuation_date + relativedelta(months=bond.months)
        current_date = valuation_date + relativedelta(months=bond.coupon_interval)

        while current_date <= maturity_date:
            self.coupon_maturities.append(bond.day_count(valuation_date, current_date))
            current_date += relativedelta(months=bond.coupon_interval)

        bond.cashflows = [forward_curve.evaluate(t) + spread for t in self.coupon_maturities]
        bond.cashflows[-1] += bond.nominal
        bond.name = "Floating Rate Bond"

class InflationLinkedBondPricer(BondPricer):
    def compute_cashflows(self, bond: Bond, valuation_date: date, coupon_rates: list[float], inflation_curve: InflationCurve, cpi_initial: float, lag_months: int = 3):
        self.coupon_maturities = []
        lag = lag_months / 12

        def inflation_factor(t: float) -> float:
            return inflation_curve.get_cpi(t - lag) / cpi_initial

        maturity_date = valuation_date + relativedelta(months=bond.months)
        current_date = valuation_date + relativedelta(months=bond.coupon_interval)

        while current_date <= maturity_date:
            self.coupon_maturities.append(bond.day_count(valuation_date, current_date))
            current_date += relativedelta(months=bond.coupon_interval)

        bond.cashflows = [
            coupon_rates[i] * bond.nominal * inflation_factor(t)
            for i, t in enumerate(self.coupon_maturities)
        ]
        bond.cashflows[-1] += bond.nominal * inflation_factor(self.coupon_maturities[-1])
        bond.name = "Inflation Linked Bond"

class SwaptionPricer:
    def black_price(self, swaption: Swaption, valuation_date: date, forward: float, discount: float, vol: float) -> float:
        T = swaption.months / 12
        K = swaption.strike
        F = forward
        sigma = vol

        if sigma == 0 or T == 0:
            return max(0.0, discount * swaption.notional * (F - K if swaption.is_payer else K - F))

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if swaption.is_payer:
            return discount * swaption.notional * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return discount * swaption.notional * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    def implied_vol(self, swaption: Swaption, valuation_date: date, price_market: float, forward: float, discount: float) -> float:
        def objective(sigma_array):
            sigma = sigma_array[0]
            price_model = self.black_price(swaption, valuation_date, forward, discount, sigma)
            return price_model - price_market

        guess = [0.2]
        result = root(objective, guess)

        return result.x[0] if result.success else np.nan
