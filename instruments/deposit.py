from instruments import Instrument
from typing import Callable

class Deposit(Instrument):
    def __init__(self, day_count: Callable, months: int):
        super().__init__("deposit", day_count, months)