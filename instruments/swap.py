from instruments import Instrument
from typing import Callable


class Swap(Instrument):
    def __init__(self, day_count: Callable, months: int, coupon_interval: int):
        super().__init__("swap", day_count, months)
        self.coupon_interval = coupon_interval