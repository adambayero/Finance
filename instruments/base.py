from datetime import date
from typing import Callable

class Instrument:
    def __init__(self, name: str, day_count: Callable, months: int):
        self.name = name
        self.day_count = day_count
        self.months = months
