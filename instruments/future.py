from instruments import Instrument
from typing import Callable

class Future(Instrument):
    def __init__(self, day_count: Callable, months: int):
        super().__init__("future", day_count, months)
