from utils import nelson_siegel_svensson
from typing import Callable
from scipy.interpolate import interp1d

class ZeroCouponCurve:
    def __init__(self, interpolation: Callable):
        self.curve = {0.0: 1.0}
        self.interpolation = interpolation

        self.T = None
        self.D = None
        self.ZC = None
        self.FWD = None

        self.popt_ns = None
        self.popt_nss = None

        self.D_ns = None
        self.D_nss = None
        self.ZC_ns = None
        self.ZC_nss = None
        self.FWD_ns = None
        self.FWD_nss = None

    def evaluate(self, t: float) -> float:
        if t in self.curve:
            return self.curve[t]
        return nelson_siegel_svensson(t, *self.popt_nss)
    
class InflationCurve:
    def __init__(self, dates: list[float], cpi_values: list[float]):
        self.interpolator = interp1d(dates, cpi_values, kind="linear", fill_value="extrapolate")

    def get_cpi(self, t: float) -> float:
        return float(self.interpolator(t))