from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Callable, List, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from tabulate import tabulate
import calendar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Instrument:
    def __init__(self, name: str, valuation_date: date, day_count: Callable, months: int):
        self.name = name
        self.valuation_date = valuation_date
        self.day_count = day_count
        self.months = months

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

def linear_interpolation(curve: dict, t: float) -> float:
    known_ts = sorted(curve.keys())
    if t in curve:
        return curve[t]
    for i in range(len(known_ts) - 1):
        t1, t2 = known_ts[i], known_ts[i+1]
        if t1 < t < t2:
            D1, D2 = curve[t1], curve[t2]
            return np.interp(t, [t1, t2], [D1, D2])
    raise ValueError(f"t={t} not in range")

def log_interpolation(curve: dict, t: float) -> float:
    known_ts = sorted(curve.keys())
    if t in curve:
        return curve[t]
    for i in range(len(known_ts) - 1):
        t1, t2 = known_ts[i], known_ts[i+1]
        if t1 < t < t2:
            D1, D2 = np.log(curve[t1]), np.log(curve[t2])
            return np.exp(np.interp(t, [t1, t2], [D1, D2]))
    raise ValueError(f"t={t} not in range")

def spline_interpolation(curve: dict, t: float) -> float:
    known_ts = sorted(curve.keys())

    try:
        discounts = np.array([curve[ti] for ti in known_ts])
        spline = CubicSpline(known_ts, discounts, bc_type='natural')
        return spline(t)
    except Exception as e:
        raise ValueError(f"Erreur lors de l'interpolation en t={t:.2f} : {e}")

def display_grid(X: List[List[NDArray[np.float64]]], Y: List[List[NDArray[np.float64]]], titles: Union[str, List[str]] = "", xlabels: Union[str, List[str]] = "", ylabels: Union[str, List[str]] = "", labels: Union[None, List[List[str]]] = None, alpha: float = 0.5, ncols: int = 2) -> None:
    
    n = len(X)
    nrows = (n + ncols - 1) // ncols
    colors = ["blue", "red", "purple", "orange", "cyan", "brown", "magenta", "olive", "teal"]

    fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for i in range(n):
        row, col = divmod(i, ncols)

        if i == n - 1 and (n % ncols != 0):
            ax = fig.add_subplot(gs[row, :])
        else:
            ax = fig.add_subplot(gs[row, col])

        for j, (x, y) in enumerate(zip(X[i], Y[i])):
            color = colors[j % len(colors)]
            label = labels[i][j] if labels and labels[i] and j < len(labels[i]) else None
            ax.plot(x, y, color=color, alpha=alpha, label=label)

        ax.set_title(titles[i] if isinstance(titles, list) else titles)
        ax.set_xlabel(xlabels[i] if isinstance(xlabels, list) else xlabels)
        ax.set_ylabel(ylabels[i] if isinstance(ylabels, list) else ylabels)
        ax.grid(True)
        if labels and labels[i]:
            ax.legend()

    plt.tight_layout()
    plt.show()

def display_tabular(params: list[float]) -> None:
    greek_betas = ["β₀", "β₁", "β₂", "β₃", "β₄", "β₅", "β₆"]
    lambdas = ["λ₁", "λ₂", "λ₃", "λ₄", "λ₅"]

    names = []
    for i in range(len(params)):
        if i < 2:
            names.append(greek_betas[i])
        elif i < 2 + (len(params) - 2) // 2:
            names.append(greek_betas[i])
        else:
            names.append(lambdas[i - (2 + (len(params) - 2) // 2)])

    rows = [(name, f"{val:.6f}") for name, val in zip(names, params)]
    print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="fancy_grid")+"\n")

def nelson_siegel(t: NDArray[np.float64], beta0: float, beta1: float, beta2: float, lambd: float) -> NDArray[np.float64]:
    term1 = (1 - np.exp(-lambd * t)) / (lambd * t)
    term2 = term1 - np.exp(-lambd * t)
    return beta0 + beta1 * term1 + beta2 * term2

def nelson_siegel_svensson(t: NDArray[np.float64], beta0: float, beta1: float, beta2: float, beta3: float, lambd1: float, lambd2: float) -> NDArray[np.float64]:
    term1 = (1 - np.exp(-lambd1 * t)) / (lambd1 * t)
    term2 = term1 - np.exp(-lambd1 * t)
    term3 = ((1 - np.exp(-lambd2 * t)) / (lambd2 * t)) - np.exp(-lambd2 * t)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

def derivative(f: callable, x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)