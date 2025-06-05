from instruments import Instrument
from typing import Callable


class Swaption(Instrument):
    def __init__(self, day_count: Callable, option_months: int, swap_tenor_months: int, strike: float, is_payer: bool, notional: float = 1e6):
        super().__init__("swaption", day_count, option_months)
        self.swap_tenor_months = swap_tenor_months
        self.strike = strike
        self.is_payer = is_payer
        self.notional = notional
