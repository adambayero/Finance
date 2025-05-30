from instruments import Deposit, Future, Swap, Bond
from utils import display_tabular
from market_data import ZeroCouponCurve, InflationCurve
import numpy as np
from scipy.optimize import root
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Callable

class MarketDeposit(Deposit):
    def __init__(self, rate: float, valuation_date: date, day_count: Callable, months: int):
        super().__init__(day_count, months)
        self.valuation_date = valuation_date
        self.rate = rate
        self.maturity = self.day_count(self.valuation_date, self.valuation_date + relativedelta(months=self.months))

class MarketFuture(Future):
    def __init__(self, rate: float, valuation_date: date, day_count: Callable, months: int, maturity: date):
        super().__init__(day_count, months)
        self.rate = rate
        self.valuation_date = valuation_date
        self.maturity = self.day_count(self.valuation_date, maturity)
        self.start = self.day_count(self.valuation_date, maturity - relativedelta(months=self.months))

class MarketSwap(Swap):
    def __init__(self, rate: float, valuation_date: date, day_count: Callable, months: int, coupon_interval: int):
        super().__init__(day_count, months, coupon_interval)
        self.rate = rate
        self.valuation_date = valuation_date
        self.coupon_interval = coupon_interval
        
        self.coupon_dates = []
        self.maturity = valuation_date + relativedelta(months=self.months)
        current_date = self.valuation_date
        while current_date <= self.maturity:
            self.coupon_dates.append(day_count(valuation_date, current_date))
            current_date += relativedelta(months=coupon_interval)
        self.maturity = self.coupon_dates[-1]

class MarketBond(Bond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_dates: list[float], cashflows: list[float]):
        super().__init__(day_count, months, nominal, cashflows)
        self.price = price
        self.valuation_date = valuation_date
        self.coupon_dates = coupon_dates

        self.YTM = self.compute_YTM()
        self.duration = self.compute_duration()
        self.modified_duration = self.compute_modified_duration()
        self.sensitivity = self.compute_sensitivity()
        self.convexity = self.compute_convexity()
        self.DV01 = self.compute_DV01()

        self.R = np.linspace(0, 1, 1000)
        self.P = np.linspace(50, 150, 1000)
    
    def compute_YTM(self) -> float:
        def price_at_rate(r: float) -> float:
            total = sum(c / (1 + r) ** t for c, t in zip(self.cashflows, self.coupon_dates))
            return total

        def equation(r: float) -> float:
            return price_at_rate(r) - self.price

        return root(equation, 0.05).x[0]
    
    def compute_duration(self) -> float:
        total = 0
        for c, t in zip(self.cashflows, self.coupon_dates):
            total += (t * c / (1 + self.YTM) ** t)
        return total / self.price
    
    def compute_modified_duration(self) -> float:
        return self.duration / (1 + self.YTM)
    
    def compute_sensitivity(self) -> float:
        return self.modified_duration
    
    def compute_convexity(self) -> float:
        total = 0
        for c, t in zip(self.cashflows, self.coupon_dates):
            total += t * (t + 1) * c / (1 + self.YTM) ** t
        return total / (self.price * (1 + self.YTM) ** 2)
    
    def compute_DV01(self) -> float:
        return -(self.modified_duration * self.price) / 10000
    
    def compute_accrued_interest(self, t) -> float:
        past_dates = [t for t in self.coupon_dates if t <= t]
        next_dates = [t for t in self.coupon_dates if t > t]

        if not past_dates or not next_dates:
            return 0.0

        t_last = past_dates[-1]
        t_next = next_dates[0]
        i = self.coupon_dates.index(t_next)

        time_since_last = t - t_last
        period = t_next - t_last

        accrued = self.cashflows[i] * (time_since_last / period)
        return accrued

    def compute_dirty_price(self, t) -> float:
        accrued = self.compute_accrued_interest(t)
        return self.price + accrued

    def compute_clean_price(self) -> float:
        return self.price
    
    def display_metrics(self) -> None:
        print(f"Metrics for {self.name} at {self.valuation_date}:\n")
        display_tabular([
            ["Price", self.price],
            ["YTM", self.YTM],
            ["Duration", self.duration],
            ["Modified Duration", self.modified_duration],
            ["Sensitivity", self.sensitivity],
            ["Convexity", self.convexity],
            ["DV01", self.DV01]
        ], headers=["Metric", "Value"])

class MarketZeroCouponBond(MarketBond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, maturity: float):
        super().__init__(price, valuation_date, day_count, months, nominal, [maturity], [nominal])
        self.maturity = maturity
        self.name = "Zero Coupon Bond"

class MarketFixedRateBond(MarketBond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_interval: float, coupon_rate: float):
        coupon_dates = []
        self.maturity = valuation_date + relativedelta(months=months)
        current_date = valuation_date + relativedelta(months=coupon_interval)
        while current_date <= self.maturity:
            coupon_dates.append(day_count(valuation_date, current_date))
            current_date += relativedelta(months=coupon_interval)
        self.maturity = coupon_dates[-1]

        cashflows = [coupon_rate * nominal] * len(coupon_dates)
        cashflows[-1] += nominal
        super().__init__(price, valuation_date, day_count, months, nominal, coupon_dates, cashflows)
        self.name = "Fixed Rate Bond"

class MarketFloatingRateBond(MarketBond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_interval: float, spread: float, forward_curve: ZeroCouponCurve):
        coupon_dates = []
        self.maturity = valuation_date + relativedelta(months=months)
        current_date = valuation_date + relativedelta(months=coupon_interval)
        while current_date <= self.maturity:
            coupon_dates.append(day_count(valuation_date, current_date))
            current_date += relativedelta(months=coupon_interval)
        self.maturity = coupon_dates[-1]
        
        cashflows = [forward_curve.evaluate(t) + spread for t in coupon_dates]
        cashflows[-1] += nominal
        super().__init__(price, valuation_date, day_count, months, nominal, coupon_dates, cashflows)
        self.name = "Floating Rate Bond"

class MarketInflationLinkedBond(MarketBond):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_interval: float, coupon_rates: list[float], inflation_curve: InflationCurve, cpi_initial: float, lag_months: int = 3):
        coupon_dates = []
        self.maturity = valuation_date + relativedelta(months=months)
        current_date = valuation_date + relativedelta(months=coupon_interval)
        while current_date <= self.maturity:
            coupon_dates.append(day_count(valuation_date, current_date))
            current_date += relativedelta(months=coupon_interval)
        self.maturity = coupon_dates[-1]
        
        self.inflation_curve = inflation_curve
        self.cpi_initial = cpi_initial
        self.lag = lag_months / 12
        
        cashflows = [coupon_rates[i] * nominal * self.inflation_factor(t) for i, t in enumerate(coupon_dates)]
        cashflows[-1] += nominal * self.inflation_factor(coupon_dates[-1])

        super().__init__(price, valuation_date, day_count, months, nominal, coupon_dates, cashflows)
        self.name = "Inflation Linked Bond"

    def inflation_factor(self, t: float) -> float:
        cpi_t = self.inflation_curve.get_cpi(t - self.lag)
        return cpi_t / self.cpi_initial