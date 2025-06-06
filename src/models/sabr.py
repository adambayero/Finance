import numpy as np
from scipy.optimize import curve_fit

def sabr_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    if F == K:
        term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / (F ** (2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
        term3 = ((2 - 3 * rho ** 2) * nu ** 2) / 24
        return alpha / (F ** (1 - beta)) * (1 + (term1 + term2 + term3) * T)
    
    logFK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    A = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * logFK ** 2 + ((1 - beta) ** 4 / 1920) * logFK ** 4))
    B = z / x_z

    term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / FK_beta
    term2 = ((2 - 3 * rho ** 2) * nu ** 2) / 24
    C = 1 + (term1 + term2) * T

    return A * B * C

def sabr_fit(F: float, T: float, strikes: np.ndarray, market_vols: np.ndarray) -> float:
    def wrapper(K: np.ndarray, alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
        return np.array([sabr_vol(k, F, T, alpha, beta, rho, nu) for k in K])
    
    bounds = ([1e-4, 0.0, -0.999, 1e-4], [1.0, 1.0, 0.999, 2.0])

    popt, _ = curve_fit(wrapper, strikes, market_vols, bounds=bounds)

    return popt

def simulate_sabr_paths(F0: float, alpha0: float, beta: float, rho: float, nu: float, T: float, n_steps: int, n_paths: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    dt = T / n_steps
    F = np.full((n_paths, n_steps), F0)
    alpha = np.full((n_paths, n_steps), alpha0)

    for i in range(n_steps - 1):
        Z1 = np.random.normal(0, 1, size=n_paths)
        Z2 = np.random.normal(0, 1, size=n_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        alpha[:, i+1] = alpha[:, i] * np.exp(nu * np.sqrt(dt) * W2 - 0.5 * nu**2 * dt)
        F[:, i+1] = F[:, i] + alpha[:, i] * F[:, i]**beta * np.sqrt(dt) * W1

    return F, alpha