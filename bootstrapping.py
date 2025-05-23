from utils import nelson_siegel, nelson_siegel_svensson, Instrument, display_grid
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root, curve_fit
from scipy.interpolate import interp1d

class Deposit(Instrument):
    def __init__(self, rate: float, valuation_date: date, day_count: Callable, months: int):
        super().__init__("deposit", valuation_date, day_count, months)
        self.rate = rate
        self.maturity = self.day_count(self.valuation_date, self.valuation_date + relativedelta(months=self.months))
    
    def price(self) -> float:
        return 1 / (1 + self.rate * self.maturity)

class Future(Instrument):
    def __init__(self, rate: float, valuation_date: date, day_count: Callable, months: int, maturity: date):
        super().__init__("future", valuation_date, day_count, months)
        self.rate = rate
        self.maturity = self.day_count(self.valuation_date, maturity)
        self.start = self.day_count(self.valuation_date, maturity - relativedelta(months=self.months))

    def price(self, curve: dict, interpolation: Callable) -> float:
        D_start = interpolation(curve, self.start)
        return D_start / (1 + (self.maturity - self.start) * self.rate)

class Swap(Instrument):
    def __init__(self, rate: float, valuation_date: date, day_count: Callable, months: int, coupon_interval: int):
        super().__init__("swap", valuation_date, day_count, months)
        self.rate = rate
        self.coupon_dates = []
        self.maturity = valuation_date + relativedelta(months=self.months)
        current_date = self.valuation_date
        while current_date <= self.maturity:
            self.coupon_dates.append(day_count(valuation_date, current_date))
            current_date += relativedelta(months=coupon_interval)
        self.maturity = self.coupon_dates[-1]

    def price(self, curve: dict, interpolation: Callable) -> tuple[list[float], NDArray[np.float64]]:
        unknown_dates = [t for t in self.coupon_dates if t > max(curve.keys())]

        def equation(x):
            local_curve = curve.copy()
            for t, d in zip(unknown_dates, x):
                local_curve[t] = d
            zero_curve = lambda t: interpolation(local_curve, t)
            left_curve = curve.copy()
            right_curve = {unknown_dates[i]: x[i] for i in range(1, len(x))}
            res = []

            pv_fixed = self.rate * sum((self.coupon_dates[i] - self.coupon_dates[i-1]) * zero_curve(t) for i, t in enumerate(self.coupon_dates) if i > 0)
            pv_float = 1 - zero_curve(self.coupon_dates[-1])
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

class Curve:
    def __init__(self, interpolation: Callable, instruments: list[Instrument]):
        self.curve = {0.0: 1.0}
        self.interpolation = interpolation
        self.instruments = instruments

        self.bootstrap()

        self.T = None
        self.D = None
        self.ZC = None
        self.FWD = None

        self.compute_curve()

        self.popt_ns = self.fit(nelson_siegel)
        self.popt_nss = self.fit(nelson_siegel_svensson)

        self.D_ns = None
        self.D_nss = None
        self.ZC_ns = None
        self.ZC_nss = None
        self.FWD_ns = None
        self.FWD_nss = None

        self.adjust_curve(nelson_siegel)
        self.adjust_curve(nelson_siegel_svensson)

    def update_curve(self, instrument: Instrument) -> None:
        if instrument.name == "deposit":
            self.curve[instrument.maturity] = instrument.price()
        elif instrument.name == "future":
            self.curve[instrument.maturity] = instrument.price(self.curve, self.interpolation)
        elif instrument.name == "swap":
            unknown_dates, solution = instrument.price(self.curve, self.interpolation)
            for t, d in zip(unknown_dates, solution):
                self.curve[t] = d

    def evaluate(self, t: float) -> float:
        if t in self.curve:
            return self.curve[t]
        else:
            return self.interpolation(self.curve, t)
    
    def bootstrap(self) -> None:
        for instrument in self.instruments:
            print(f"⏳ Bootstrapping {instrument.name} with rate {instrument.rate:.4%}...")
            self.update_curve(instrument)
            print(f"✅ Computed discount factor D({instrument.maturity:.3f}) = {self.curve[instrument.maturity]:.6f} added to the curve.\n")
        print("🎯 Bootstrapping completed!\n")

    def compute_curve(self) -> None:
        T_sorted = sorted(self.curve.keys())
        self.T = np.linspace(T_sorted[1], T_sorted[-1], 1000)
        self.D = np.array([self.interpolation(self.curve, t) for t in self.T])
        self.ZC = -np.log(self.D) / self.T
        self.FWD = -np.gradient(np.log(self.D), self.T)

    def fit(self, method: Callable) -> NDArray[np.float64]:
        bounds = {
            nelson_siegel: ([-1, -10, -10, 0.01], [10, 10, 10, 10]),
            nelson_siegel_svensson: ([-1, -10, -10, -10, 0.01, 0.01], [10, 10, 10, 10, 10, 10])
        }

        popt, _ = curve_fit(method, self.T, self.ZC, bounds=bounds[method])
        return popt

    def adjust_curve(self, method: Callable) -> None:
        if method == nelson_siegel:
            self.popt_ns = self.fit(method)
            self.ZC_ns = nelson_siegel(self.T, *self.popt_ns)
            self.D_ns = np.exp(-self.ZC_ns * self.T)
            self.FWD_ns = -np.gradient(np.log(self.D_ns), self.T)
        elif method == nelson_siegel_svensson:
            self.popt_nss = self.fit(method)
            self.ZC_nss = nelson_siegel_svensson(self.T, *self.popt_nss)
            self.D_nss = np.exp(-self.ZC_nss * self.T)
            self.FWD_nss = -np.gradient(np.log(self.D_nss), self.T)

    def forward_nss(self, t: float) -> float:
        interp = interp1d(self.T, self.FWD_nss, kind='linear', fill_value="extrapolate")
        return float(interp(t))
    
class InflationCurve:
    def __init__(self, dates: list[float], cpi_values: list[float]):
        self.interpolator = interp1d(dates, cpi_values, kind="linear", fill_value="extrapolate")

    def get_cpi(self, t: float) -> float:
        return float(self.interpolator(t))

def display_bootstrap_result(curves: list[Curve], legends: list[str]) -> None:
    display_grid([[curve.T for curve in curves] for _ in range(2)], [[curve.D for curve in curves], [curve.ZC for curve in curves]], ["Discount", "Zero-Coupon"], "Maturity", ["Discount", "Zero-Coupon", "Forward"], [legends, legends])

def display_adjusted_curve(curves: list[Curve], legends: list[str]) -> None:
    display_grid([[curve.T for curve in curves] for _ in range(3)], [[curve.D_ns for curve in curves], [curve.ZC_ns for curve in curves], [curve.FWD_ns for curve in curves]], ["Discount NS", "Zero-Coupon NS", "Forward NS"], "Maturity", ["Discount NS", "Zero-Coupon NS", "Forward NS"], [legends, legends, legends])
    display_grid([[curve.T for curve in curves] for _ in range(3)], [[curve.D_nss for curve in curves], [curve.ZC_nss for curve in curves], [curve.FWD_nss for curve in curves]], ["Discount NSS", "Zero-Coupon NSS", "Forward NS"], "Maturity", ["Discount NSS", "Zero-Coupon NSS", "Forward NSS"], [legends, legends, legends])