from utils import Instrument
from datetime import date
from typing import Callable
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt

class Bond(Instrument):
    def __init__(self, price: float, valuation_date: date, day_count: Callable, months: int, nominal: float, coupon_dates: list[float], coupon_rates: list[float], rates: list[float]):
        super().__init__("bond", valuation_date, day_count, months)
        self.price = price
        self.nominal = nominal
        self.coupon_dates = coupon_dates
        self.coupon_rates = coupon_rates
        self.rates = rates

        self.YTM = self.compute_YTM()
        self.duration = self.compute_duration()
        self.modified_duration = self.compute_modified_duration()
        self.sensitivity = self.compute_sensitivity()
        self.convexity = self.compute_convexity()
        self.DV01 = self.compute_DV01()
        self.clean_price = None
        self.dirty_price = None

    def price_bond(self) -> float:
        total = 0
        for i, date in enumerate(self.coupon_dates):
            total += (self.coupon_rates[i] * self.nominal / (1 + self.rates[i]) ** date)
        total += (self.nominal / (1 + self.rates[-1]) ** self.coupon_dates[-1])
        return total
    
    def compute_YTM(self) -> float:
        rates = self.rates.copy()
        def equation(YTM: float) -> float:
            self.rates = [YTM] * len(self.coupon_dates)
            return self.price - self.price_bond()

        solution = root(equation, 0.05).x[0]
        self.rates = rates
        return solution
    
    def compute_duration(self) -> float:
        total = 0
        for i, date in enumerate(self.coupon_dates):
            total += (date * self.coupon_rates[i] * self.nominal / (1 + self.YTM) ** date)
        total += (self.coupon_dates[-1] * self.nominal / (1 + self.YTM) ** self.coupon_dates[-1])
        return total / self.price
    
    def compute_modified_duration(self) -> float:
        return self.duration / (1 + self.YTM)
    
    def compute_sensitivity(self) -> float:
        return self.modified_duration
    
    def compute_convexity(self) -> float:
        total = 0
        for i, t in enumerate(self.coupon_dates):
            total += t * (t + 1) * self.coupon_rates[i] * self.nominal / (1 + self.YTM) ** t
        total += self.coupon_dates[-1] * (self.coupon_dates[-1] + 1) * self.nominal / (1 + self.YTM) ** self.coupon_dates[-1]
        return total / (self.price * (1 + self.YTM) ** 2)
    
    def compute_DV01(self) -> float:
        return -(self.modified_duration * self.price) / 10000
    
    def display_metrics(self):
        print(f"Price: {self.price:.4f}")
        print(f"YTM: {self.YTM:.4%}")
        print(f"Duration: {self.duration:.4f}")
        print(f"Modified Duration: {self.modified_duration:.4f}")
        print(f"Sensitivity: {self.sensitivity:.4f}")
        print(f"Convexity: {self.convexity:.4f}")
        print(f"DV01: {self.DV01:.4f}")

"""R = np.linspace(0, 1, 1000)
P = np.linspace(50, 150, 1000)

plt.plot(R, [duration_modify(price, nominal, coupon_dates, coupon_rate, r) for r in R], color='blue', alpha=0.5)
plt.title('Modified Duration')
plt.xlabel('Rate')
plt.ylabel('Modified Duration')
plt.grid()
plt.show()

plt.plot(R, [price_bond(nominal, coupon_dates, coupon_rate, [r] * 7) for r in R], color='blue', alpha=0.5)
plt.title('Price of Bond')
plt.xlabel('Rate')
plt.ylabel('Price')
plt.grid()
plt.show()

plt.plot(P, [YTM(p, nominal, coupon_dates, coupon_rate) for p in P], color='blue', alpha=0.5)
plt.title('Yield to Maturity')
plt.xlabel('Price')
plt.ylabel('YTM')
plt.grid()
plt.show()

price_bond_simplfied = lambda rate: price_bond(nominal, coupon_dates, coupon_rate, [rate] * 7)

plt.plot(R, [derive(price_bond_simplfied, r) for r in R], color='blue', alpha=0.5)
plt.plot(R, [-duration_modify(price_bond(nominal, coupon_dates, coupon_rate, [r]*7), nominal, coupon_dates, coupon_rate, r) * price_bond(nominal, coupon_dates, coupon_rate, [r]*7) for r in R], color='red', alpha=0.5, linestyle='--')
plt.title('Derivative of Price wrt Rate vs Duration Approximation')
plt.xlabel('Rate')
plt.ylabel('dP/dr')
plt.legend(['Numerical Derivative', '-Duration × Price'])
plt.grid()
plt.show()

plt.plot(R, [price_bond(nominal, coupon_dates, coupon_rate, [r] * 7) for r in R], color='blue', alpha=0.5)
plt.plot(R, [price_bond(nominal, coupon_dates, coupon_rate, [0] * 7) * (1 - r * sensitivity(nominal, coupon_dates, coupon_rate, 0)) for r in R], color='red', alpha=0.5, linestyle='--')
plt.title('Price of Bond vs Duration Approximation')
plt.xlabel('Rate')
plt.ylabel('Price')
plt.legend(['Price', 'Duration Approximation'])
plt.grid()
plt.show()

plt.plot(R, [price_bond(nominal, coupon_dates, coupon_rate, [r] * 7) for r in R], color='blue', alpha=0.5)
plt.plot(R, [price_bond(nominal, coupon_dates, coupon_rate, [0] * 7) * (1 - r * sensitivity(nominal, coupon_dates, coupon_rate, 0) + 0.5 * convexity(price_bond(nominal, coupon_dates, coupon_rate, [0] * 7), nominal, coupon_dates, coupon_rate, 0) * r ** 2) for r in R], color='red', alpha=0.5, linestyle='--')
plt.title('Price of Bond vs Duration and Convexity Approximation')
plt.xlabel('Rate')
plt.ylabel('Price')
plt.legend(['Price', 'Duration and Convexity Approximation'])
plt.grid()
plt.show()"""