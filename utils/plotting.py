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

def display_tabular(data: list[list], headers: list[str]) -> None:
    print(tabulate(data, headers=headers, tablefmt="fancy_grid") + "\n")