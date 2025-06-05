from instruments import Instrument
from typing import Callable

class Bond(Instrument):
    def __init__(self, day_count: Callable, months: int, nominal: float, cashflows: list[float]):
        super().__init__("bond", day_count, months)
        self.nominal = nominal
        self.cashflows = cashflows