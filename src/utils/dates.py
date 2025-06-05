from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Callable, List
import calendar

def actual_360(start: date, end: date) -> float:
    return (end - start).days / 360

def actual_365(start: date, end: date) -> float:
    return (end - start).days / 365

def third_wednesday(year: int, month: int) -> date:
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    wednesdays = [day for day in c.itermonthdays2(year, month) if day[0] != 0 and day[1] == calendar.WEDNESDAY]
    return date(year, month, wednesdays[2][0])

def compute_year_fractions(dates: list[date], valuation_date: date, day_count: Callable[[date, date], float]) -> list[float]:
    return [day_count(valuation_date, d) for d in dates]

def prepare_future_intervals(dates: List[date], valuation_date: date, day_count: Callable[[date, date], float], months: int) -> List[List[float]]:
    return [[day_count(valuation_date, d - relativedelta(months=months)), day_count(valuation_date, d)] for d in dates]

def prepare_swap_dates(swap_dates: list[date], valuation_date: date, day_count: Callable[[date, date], float], months: int) -> list[list[float]]:
    dates = []
    for swap_date in swap_dates:
        coupon_dates = []
        current_date = valuation_date
        while current_date <= swap_date:
            coupon_dates.append(current_date)
            current_date += relativedelta(months=months)
        dates.append([day_count(valuation_date, t) for t in coupon_dates])
    return dates