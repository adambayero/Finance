from instruments import Deposit, Future, Swap
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