from utils import linear_interpolation, log_interpolation, spline_interpolation, display_bootstrap_result, display_adjusted_curve
from market_data import instruments
from models import create_curve

def main():
    curves = [
        create_curve(linear_interpolation, instruments),
        create_curve(log_interpolation, instruments),
        create_curve(spline_interpolation, instruments)
]
    display_bootstrap_result(curves, ["linear", "log", "spline"])
    display_adjusted_curve(curves, ["linear", "log", "spline"])
    return 0

main()