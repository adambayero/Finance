from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Callable, List, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import root, curve_fit
from tabulate import tabulate
import calendar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def prepare_future_intervals(dates: List[date], valuation_date: date, day_count: Callable[[date, date], float]) -> List[List[float]]:
    return [[day_count(valuation_date, d - relativedelta(months=3)), day_count(valuation_date, d)] for d in dates]

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

def initialize_curve() -> dict:
    return {0.0: 1.0}

def price_deposit(curve: dict, rate: float, end: float) -> None:
    curve[end] = 1 / (1 + rate * end)

def price_future(curve: dict, price: float, start: float, end: float, interpolation: Callable) -> None:
    rate = (100 - price) / 100
    D_start = interpolation(curve, start)
    curve[end] = D_start / (1 + (end - start) * rate)

def price_swap(curve: dict, rate: float, coupon_dates: list[float], interpolation: Callable) -> None:
    unknown_dates = [t for t in coupon_dates if t > max(curve.keys())]

    def equation(x):
        local_curve = curve.copy()
        for t, d in zip(unknown_dates, x):
            local_curve[t] = d
        zero_curve = lambda t: interpolation(local_curve, t)
        left_curve = curve.copy()
        right_curve = {unknown_dates[i]: x[i] for i in range(1, len(x))}
        res = []

        pv_fixed = rate * sum((coupon_dates[i] - coupon_dates[i-1]) * zero_curve(t) for i, t in enumerate(coupon_dates) if i > 0)
        pv_float = 1 - zero_curve(coupon_dates[-1])
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
    for t, d in zip(unknown_dates, solution.x):
        curve[t] = d

def bootstrapping(rate_deposit: float, end_deposit: float, future_prices: List[float], future_intervals: List[List[float]], swap_rates: List[float], swap_coupon_dates: List[List[float]], interpolation: Callable[[dict, float], float]) -> dict:
    
    curve = initialize_curve()

    print(f"⏳ Bootstrapping deposit with rate {rate_deposit:.4%} over {end_deposit:.3f} years...")
    price_deposit(curve, rate_deposit, end_deposit)
    print(f"✅ Computed discount factor D({end_deposit:.3f}) = {curve[end_deposit]:.6f} added to the curve.\n")

    for i, (price, interval) in enumerate(zip(future_prices, future_intervals)):
        start, end = interval
        print(f"⏳ Bootstrapping future {i+1} from {start:.3f}y to {end:.3f}y at price {price:.3f}...")
        price_future(curve, price, start, end, interpolation)
        print(f"✅ Computed discount factor D({end:.3f}) = {curve[end]:.6f} added to the curve.\n")

    for i, (rate, coupons) in enumerate(zip(swap_rates, swap_coupon_dates)):
        maturity = coupons[-1]
        print(f"⏳ Bootstrapping swap {i+1} with maturity {maturity:.3f} years and fixed rate {rate:.4%}...")

        unknowns = [t for t in swap_coupon_dates[i] if t > max(curve.keys())]
        if not unknowns:
            print(f"⚠️ Skip swap {i} (aucune nouvelle date à résoudre)\n")
            continue

        price_swap(curve, rate, coupons, interpolation)

        if unknowns:
            print(f"✅ Added {len(unknowns)} discount factors:")
            for t in unknowns:
                print(f"    • D({t:.3f}) = {curve[t]:.6f}")
            print()

    print("🎯 Bootstrapping completed!\n")
    return curve


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

def nelson_siegel(t: NDArray[np.float64], beta0: float, beta1: float, beta2: float, lambd: float) -> NDArray[np.float64]:
    term1 = (1 - np.exp(-lambd * t)) / (lambd * t)
    term2 = term1 - np.exp(-lambd * t)
    return beta0 + beta1 * term1 + beta2 * term2

def nelson_siegel_svensson(t: NDArray[np.float64], beta0: float, beta1: float, beta2: float, beta3: float, lambd1: float, lambd2: float) -> NDArray[np.float64]:
    term1 = (1 - np.exp(-lambd1 * t)) / (lambd1 * t)
    term2 = term1 - np.exp(-lambd1 * t)
    term3 = ((1 - np.exp(-lambd2 * t)) / (lambd2 * t)) - np.exp(-lambd2 * t)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

def fit(curve: dict, interpolation: Callable[[dict, float], float], method: Callable, n_points: int = 200) -> NDArray[np.float64]:
    T_sorted = sorted(curve.keys())
    T = np.linspace(T_sorted[1], T_sorted[-1], n_points)
    D_interp = np.array([interpolation(curve, t) for t in T])
    ZC = -np.log(D_interp) / T

    bounds = {
        nelson_siegel: ([-1, -10, -10, 0.01], [10, 10, 10, 10]),
        nelson_siegel_svensson: ([-1, -10, -10, -10, 0.01, 0.01], [10, 10, 10, 10, 10, 10])
    }

    popt, _ = curve_fit(method, T, ZC, bounds=bounds[method])
    return popt

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

def main() -> int:
    valuation_date = date(2025, 4, 25)

    rate_deposit = 0.0217	
    end_deposit = actual_360(valuation_date, valuation_date + relativedelta(months=3))

    future_prices = [98.195, 98.285, 98.35]
    future_maturities = [date(2025, 8, 18), date(2025, 10, 13), date(2025, 12, 15)]
    future_intervals = prepare_future_intervals(future_maturities, valuation_date, actual_360)

    swap_rates = [0.01932, 0.0187, 0.01936, 0.02095, 0.02234, 0.02394, 0.02524, 0.02384]
    swap_end_dates = [valuation_date + relativedelta(years=1), valuation_date + relativedelta(years=2), valuation_date + relativedelta(years=3), valuation_date + relativedelta(years=5), valuation_date + relativedelta(years=7), valuation_date + relativedelta(years=10), valuation_date + relativedelta(years=15), valuation_date + relativedelta(years=30)]
    swap_coupon_dates = prepare_swap_dates(swap_end_dates, valuation_date, actual_360, 6)

    """valuation_date = date(2025, 4, 28)

    rate_deposit = 0.0428
    end_deposit = actual_360(valuation_date, valuation_date + relativedelta(months=3))

    future_prices = [96.295, 96.61, 96.82]
    future_maturities = [third_wednesday(2025, 9), third_wednesday(2025, 12), third_wednesday(2026, 3) - relativedelta(days=1)]
    future_intervals = prepare_future_intervals(future_maturities, valuation_date, actual_360)

    swap_rates = [0.03837, 0.03523, 0.03458, 0.03508, 0.03611, 0.03746, 0.03904, 0.03870]
    swap_end_dates = [valuation_date + relativedelta(years=1), valuation_date + relativedelta(years=2), valuation_date + relativedelta(years=3), valuation_date + relativedelta(years=5), valuation_date + relativedelta(years=7), valuation_date + relativedelta(years=10), valuation_date + relativedelta(years=15), valuation_date + relativedelta(years=30)]
    swap_coupon_dates = prepare_swap_dates(swap_end_dates, valuation_date, actual_360, 12)"""

    lin_curve = bootstrapping(rate_deposit, end_deposit, future_prices, future_intervals, swap_rates, swap_coupon_dates, linear_interpolation)
    log_curve = bootstrapping(rate_deposit, end_deposit, future_prices, future_intervals, swap_rates, swap_coupon_dates, log_interpolation)
    spline_curve = bootstrapping(rate_deposit, end_deposit, future_prices, future_intervals, swap_rates, swap_coupon_dates, spline_interpolation)

    T_sorted = sorted(lin_curve.keys())
    T = np.linspace(T_sorted[1], T_sorted[-1], 300)

    lin_D = np.array([linear_interpolation(lin_curve, t) for t in T])
    lin_ZC = -np.log(lin_D) / T

    log_D = np.array([log_interpolation(log_curve, t) for t in T])
    log_ZC = -np.log(log_D) / T

    spline_D = np.array([spline_interpolation(spline_curve, t) for t in T])
    spline_ZC = -np.log(spline_D) / T

    display_grid(
        X=[[T, T, T], [T, T, T]],
        Y=[[lin_D, log_D, spline_D], [lin_ZC, log_ZC, spline_ZC]],
        titles=["Courbe d'actualisation", "Courbe de taux zéro-coupons"],
        xlabels="Maturité (années)",
        ylabels=["Taux d'actualisation", "Taux zéro-coupon"],
        labels=[["Interpolation linéaire", "Interpolation logarithmique", "Interpolation spline cubique"]] * 3,
        ncols=1
    )

    popt_ns = fit(lin_curve, spline_interpolation, nelson_siegel)
    popt_nss = fit(lin_curve, spline_interpolation, nelson_siegel_svensson)

    print("Fitting Nelson-Siegel parameters:")
    display_tabular(popt_ns)
    print("Fitting Nelson-Siegel-Svensson parameters:")
    display_tabular(popt_nss)

    ZC_ns = nelson_siegel(T, *popt_ns)
    ZC_nss = nelson_siegel_svensson(T, *popt_nss)

    D_ns = np.exp(-ZC_ns * T)
    D_nss = np.exp(-ZC_nss * T)

    fwd_ns = -np.gradient(np.log(D_ns), T)
    fwd_nss = -np.gradient(np.log(D_nss), T)

    display_grid(
        X=[[T, T], [T, T], [T, T]],
        Y=[[ZC_ns, ZC_nss], [D_ns, D_nss], [fwd_ns, fwd_nss]],
        titles=["Courbe de taux zéro-coupons", "Courbe d'actualisation", "Courbe de taux forward"],
        xlabels="Maturité (années)",
        ylabels=["Taux zéro-coupons", "Taux d'actualisation", "Taux forward"],
        labels=[["Nelson-Siegel", "Nelson-Siegel-Svensson"], ["Nelson-Siegel", "Nelson-Siegel-Svensson"], ["Nelson-Siegel", "Nelson-Siegel-Svensson"]],
        ncols=2
    )

    return 0

main()




