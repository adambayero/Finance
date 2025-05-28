from market_data import MarketDeposit,  MarketFuture, MarketSwap
from typing import Callable
from scipy.optimize import root
import numpy as np
from numpy.typing import NDArray

def price_deposit(deposit: MarketDeposit) -> float:
    return 1 / (1 + deposit.rate * deposit.maturity)

def price_future(future: MarketFuture, curve: dict, interpolation: Callable) -> float:
        D_start = interpolation(curve, future.start)
        return D_start / (1 + (future.maturity - future.start) * future.rate)

def price_swap(swap: MarketSwap, curve: dict, interpolation: Callable) -> tuple[list[float], NDArray[np.float64]]:
        unknown_dates = [t for t in swap.coupon_dates if t > max(curve.keys())]

        def equation(x):
            local_curve = curve.copy()
            for t, d in zip(unknown_dates, x):
                local_curve[t] = d
            zero_curve = lambda t: interpolation(local_curve, t)
            left_curve = curve.copy()
            right_curve = {unknown_dates[i]: x[i] for i in range(1, len(x))}
            res = []

            pv_fixed = swap.rate * sum((swap.coupon_dates[i] - swap.coupon_dates[i-1]) * zero_curve(t) for i, t in enumerate(swap.coupon_dates) if i > 0)
            pv_float = 1 - zero_curve(swap.coupon_dates[-1])
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