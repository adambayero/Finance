from instruments import Instrument
from utils import linear_interpolation, log_interpolation, spline_interpolation, actual_360
from models import create_curve, display_bootstrap_result, display_adjusted_curve

def example_curves(instruments: list[Instrument]) -> None:
    curves = [
        create_curve(linear_interpolation, instruments),
        create_curve(log_interpolation, instruments),
        create_curve(spline_interpolation, instruments)
    ]
    display_bootstrap_result(curves, ["linear", "log", "spline"])
    display_adjusted_curve(curves, ["linear", "log", "spline"])