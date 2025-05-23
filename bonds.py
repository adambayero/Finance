from utils import Instrument, display_grid
from datetime import date
from typing import Callable
from scipy.optimize import root
import numpy as np
from bootstrapping import Curve, InflationCurve

class Bond(Instrument):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_dates: list[float], coupon_rates: list[float], discount_curve: Curve):
        super().__init__("bond", valuation_date, day_count, months)
        self.price = price
        self.nominal = nominal
        self.coupon_dates = coupon_dates
        self.coupon_rates = coupon_rates
        self.discounts = [discount_curve.evaluate(date) for date in coupon_dates]

        self.YTM = self.compute_YTM()
        self.duration = self.compute_duration()
        self.modified_duration = self.compute_modified_duration()
        self.sensitivity = self.compute_sensitivity()
        self.convexity = self.compute_convexity()
        self.DV01 = self.compute_DV01()
        self.clean_price = None
        self.dirty_price = None

        self.R = np.linspace(0, 1, 1000)
        self.P = np.linspace(50, 150, 1000)
    
    def compute_YTM(self) -> float:
        def price_at_rate(r: float) -> float:
            discounts = [(1 + r) ** (-t) for t in self.coupon_dates]
            total = sum(self.coupon_rates[i] * self.nominal * discounts[i] for i in range(len(self.coupon_dates)))
            total += self.nominal * discounts[-1]
            return total

        def equation(r: float) -> float:
            return price_at_rate(r) - self.price

        return root(equation, 0.05).x[0]
    
    def compute_duration(self) -> float:
        total = 0
        for i, t in enumerate(self.coupon_dates):
            total += (t * self.coupon_rates[i] * self.nominal / (1 + self.YTM) ** t)
        total += (self.coupon_dates[-1] * self.nominal / (1 + self.YTM) ** self.coupon_dates[-1])
        return total / self.price
    
    def compute_modified_duration(self) -> float:
        return self.duration / (1 + self.YTM)
    
    def compute_sensitivity(self) -> float:
        return self.modified_duration
    
    def compute_convexity(self) -> float:
        total = 0
        for i, t in enumerate(self.coupon_dates):
            total += t * (t + 1) * self.coupon_rates[i] * self.nominal / (1 + self.YTM) ** t
        total += self.coupon_dates[-1] * (self.coupon_dates[-1] + 1) * self.nominal / (1 + self.YTM) ** self.coupon_dates[-1]
        return total / (self.price * (1 + self.YTM) ** 2)
    
    def compute_DV01(self) -> float:
        return -(self.modified_duration * self.price) / 10000
    
    def display_metrics(self) -> None:
        print(f"Price: {self.price:.4f}")
        print(f"YTM: {self.YTM:.4%}")
        print(f"Duration: {self.duration:.4f}")
        print(f"Modified Duration: {self.modified_duration:.4f}")
        print(f"Sensitivity: {self.sensitivity:.4f}")
        print(f"Convexity: {self.convexity:.4f}")
        print(f"DV01: {self.DV01:.4f}")

class ZeroCouponBond(Bond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, maturity: float, discount_curve: Curve):
        super().__init__(price, valuation_date, day_count, months, nominal, [maturity], [1], discount_curve)
        self.maturity = maturity

    def price_bond(self) -> float:
        return self.nominal * self.discounts[0]
    
class FixedRateBond(Bond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_dates: list[float], coupon_rates: list[float], discount_curve: Curve):
        super().__init__(price, valuation_date, day_count, months, nominal, coupon_dates, coupon_rates, discount_curve)

    def price_bond(self) -> float:
        total = 0
        for i, t in enumerate(self.coupon_dates):
            total += self.coupon_rates[i] * self.nominal * self.discounts[i]
        total += self.nominal * self.discounts[-1]
        return total
    
class FloatingRateBond(Bond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_dates: list[float], spreads: list[float], discount_curve: Curve, forward_curve: Curve):
        super().__init__(price, valuation_date, day_count, months, nominal, coupon_dates, spreads, discount_curve)
        self.forwards = [forward_curve.forward_nss(date) for date in coupon_dates]

    def price_bond(self) -> float:
        total = 0
        for i, t in enumerate(self.coupon_dates):
            total += (self.forwards[i] + self.coupon_rates[i]) * self.nominal * self.discounts[i]
        total += self.nominal * self.discounts[-1]
        return total
    
class InflationLinkedBond(Bond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_dates: list[float], real_coupon_rates: list[float], discount_curve: Curve, inflation_curve: InflationCurve, cpi_initial: float, lag_months: int = 3):
        super().__init__(price, valuation_date, day_count, months, nominal, coupon_dates, real_coupon_rates, discount_curve)
        self.inflation_curve = inflation_curve
        self.cpi_initial = cpi_initial
        self.lag = lag_months / 12

    def inflation_factor(self, t: float) -> float:
        cpi_t = self.inflation_curve.get_cpi(t - self.lag)
        return cpi_t / self.cpi_initial

    def price_bond(self) -> float:
        total = 0
        for i, t in enumerate(self.coupon_dates):
            factor = self.inflation_factor(t)
            real_coupon = self.coupon_rates[i] * self.nominal * factor
            total += real_coupon * self.discounts[i]
        total += self.nominal * self.inflation_factor(self.coupon_dates[-1]) * self.discounts[-1]
        return total
