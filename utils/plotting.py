from market_data import ZeroCouponCurve
from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# Réfléchir à leur placement

def display_bootstrap_result(curves: list[ZeroCouponCurve], legends: list[str]) -> None:
    display_grid([[curve.T for curve in curves] for _ in range(2)], [[curve.D for curve in curves], [curve.ZC for curve in curves]], ["Discount", "Zero-Coupon"], "Maturity", ["Discount", "Zero-Coupon", "Forward"], [legends, legends])

def display_adjusted_curve(curves: list[ZeroCouponCurve], legends: list[str]) -> None:
    display_grid([[curve.T for curve in curves] for _ in range(3)], [[curve.D_ns for curve in curves], [curve.ZC_ns for curve in curves], [curve.FWD_ns for curve in curves]], ["Discount NS", "Zero-Coupon NS", "Forward NS"], "Maturity", ["Discount NS", "Zero-Coupon NS", "Forward NS"], [legends, legends, legends])
    display_grid([[curve.T for curve in curves] for _ in range(3)], [[curve.D_nss for curve in curves], [curve.ZC_nss for curve in curves], [curve.FWD_nss for curve in curves]], ["Discount NSS", "Zero-Coupon NSS", "Forward NSS"], "Maturity", ["Discount NSS", "Zero-Coupon NSS", "Forward NSS"], [legends, legends, legends])