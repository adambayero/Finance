from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from matplotlib import cm

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

def display_3d_grid(X: list[NDArray[np.float64]], Y: list[NDArray[np.float64]], Z: list[NDArray[np.float64]], titles: Union[str, List[str]] = "", xlabels: Union[str, List[str]] = "", ylabels: Union[str, List[str]] = "", zlabels: Union[str, List[str]] = "", alpha: float = 0.8, ncols: int = 2, scatter: bool = False):

    n = len(X)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for i in range(n):
        row, col = divmod(i, ncols)

        if i == n - 1 and (n % ncols != 0):
            ax = fig.add_subplot(gs[row, :], projection='3d')
        else:
            ax = fig.add_subplot(gs[row, col], projection='3d')

        ax.plot_trisurf(X[i], Y[i], Z[i], cmap='viridis', edgecolor='none', alpha=alpha)
        if scatter:
            ax.scatter(X[i], Y[i], Z[i], color='k', s=8)

        ax.set_title(titles[i] if isinstance(titles, list) else titles)
        ax.set_xlabel(xlabels[i] if isinstance(xlabels, list) else xlabels)
        ax.set_ylabel(ylabels[i] if isinstance(ylabels, list) else ylabels)
        ax.set_zlabel(zlabels[i] if isinstance(zlabels, list) else zlabels)

        ax.view_init(elev=20, azim=-35)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def display_cube(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, values: np.ndarray, xlabel: str = "", ylabel: str = "", zlabel: str = "", value_label: str = "", title: str = "", resolution: int = 30) -> None:
    X, Y, Z, values = map(np.asarray, (X, Y, Z, values))
    
    xi = np.linspace(np.min(X), np.max(X), resolution)
    yi = np.linspace(np.min(Y), np.max(Y), resolution)
    zi = np.linspace(np.min(Z), np.max(Z), resolution)
    Xi, Yi, Zi = np.meshgrid(xi, yi, zi)

    values_grid = griddata(
        (X.flatten(), Y.flatten(), Z.flatten()),
        values.flatten(),
        (Xi, Yi, Zi),
        method='linear'
    )

    kw = {
        'vmin': np.nanmin(values_grid),
        'vmax': np.nanmax(values_grid),
        'levels': np.linspace(np.nanmin(values_grid), np.nanmax(values_grid), resolution),
    }

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    _ = ax.contourf(
        Xi[:, :, 0], Yi[:, :, 0], values_grid[:, :, 0],
        zdir='z', offset=zi[0], **kw
    )
    _ = ax.contourf(
        Xi[0, :, :], values_grid[0, :, :], Zi[0, :, :],
        zdir='y', offset=yi[0], **kw
    )
    C = ax.contourf(
        values_grid[:, -1, :], Yi[:, -1, :], Zi[:, -1, :],
        zdir='x', offset=xi[-1], **kw
    )

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        title=title,
    )

    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)

    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=value_label)
    plt.show()