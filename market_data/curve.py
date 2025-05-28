from typing import Callable

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

    