from market_data import instruments
from curves import example_curves
from swaptions import display_vols, compute_sabr_fit
from models import simulate_sabr_paths, sabr_vol
from utils import display_grid
import numpy as np

def main():
    #example_curves(instruments)
    display_vols()
    """popt = compute_sabr_fit()
    F, alpha = simulate_sabr_paths(0.03, *popt, 1, 1000, 5)
    T = np.linspace(0, 5, 1000)
    vol = [[sabr_vol(F[j][i], 10, T[i], *popt) for i in range(len(T))] for j in range(5)]
    display_grid([[T] * 5, [T] * 5, [T] * 5], [F, alpha, vol], ["Forward Rate", "Alpha Paths", "Vol"], "Time (Years)", ["Forward Rate", "Alpha", "Vol"])"""
    return 0

main()