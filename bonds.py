from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt

price = 82.44
nominal = 100
coupon_dates = [1, 2, 3, 4, 5, 6, 7]
coupon_rate = 0.00625

def price_bond(nominal: float, coupon_dates: list[float], coupon_rate: float, rates: list[float]) -> float:
    total = 0
    for i, date in enumerate(coupon_dates):
        total += (coupon_rate * nominal / (1 + rates[i]) ** date)
    total += (nominal / (1 + rates[-1]) ** coupon_dates[-1])
    return total

def YTM(price: float, nominal: float, coupon_dates: list[float], coupon_rate: float) -> float:

    def equation(rate: float) -> float:
        return price - price_bond(nominal, coupon_dates, coupon_rate, [rate] * len(coupon_dates))

    return root(equation, 0.05).x[0]

def duration(price: float, nominal: float, coupon_dates: list[float], coupon_rate: float, rate: float) -> float:
    total = 0
    for date in coupon_dates:
        total += (date * coupon_rate * nominal / (1 + rate) ** date)
    total += (coupon_dates[-1] * nominal / (1 + rate) ** coupon_dates[-1])
    return total / price

def duration_modify(price: float, nominal: float, coupon_dates: list[float], coupon_rate: float, rate: float) -> float:
    return duration(price, nominal, coupon_dates, coupon_rate, rate) / (1 + rate)

def derive(f: callable, x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)

def sensitivity(nominal: float, coupon_dates: list[float], coupon_rate: float, rate: float) -> float:
    return duration_modify(price_bond(nominal, coupon_dates, coupon_rate, [rate] * 7), nominal, coupon_dates, coupon_rate, rate)

def convexity(price: float, nominal: float, coupon_dates: list[float], coupon_rate: float, rate: float) -> float:
    total = 0
    for date in coupon_dates:
        total += (date * (date + 1) * coupon_rate * nominal / (1 + rate) ** date)
    total += (coupon_dates[-1] * (coupon_dates[-1] + 1) * nominal / (1 + rate) ** coupon_dates[-1])
    return total / price

def DV01(price: float, nominal: float, coupon_dates: list[float], coupon_rate: float, rate: float) -> float:
    return -(duration_modify(price, nominal, coupon_dates, coupon_rate, rate) * price) / 10000

# Dirty price, clean price coupon couru

def main():
    rate = YTM(price, nominal, coupon_dates, coupon_rate)
    print(f"YTM : {rate}")
    print(f"Duration : {duration(price, nominal, coupon_dates, coupon_rate, rate)}")
    print(f"Modified Duration : {duration_modify(price, nominal, coupon_dates, coupon_rate, rate)}")
    print(f"Sensitivity : {sensitivity(nominal, coupon_dates, coupon_rate, rate)}")
    print(f"Convexity : {convexity(price_bond(nominal, coupon_dates, coupon_rate, [0] * 7), nominal, coupon_dates, coupon_rate, rate)}")
    print(f"DV01 : {DV01(price, nominal, coupon_dates, coupon_rate, rate)}")

main()

R = np.linspace(0, 1, 1000)
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
plt.show()