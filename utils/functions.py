import numpy as np
from numpy.typing import NDArray

def nelson_siegel(t: NDArray[np.float64], beta0: float, beta1: float, beta2: float, lambd: float) -> NDArray[np.float64]:
    term1 = (1 - np.exp(-lambd * t)) / (lambd * t)
    term2 = term1 - np.exp(-lambd * t)
    return beta0 + beta1 * term1 + beta2 * term2

def nelson_siegel_svensson(t: NDArray[np.float64], beta0: float, beta1: float, beta2: float, beta3: float, lambd1: float, lambd2: float) -> NDArray[np.float64]:
    term1 = (1 - np.exp(-lambd1 * t)) / (lambd1 * t)
    term2 = term1 - np.exp(-lambd1 * t)
    term3 = ((1 - np.exp(-lambd2 * t)) / (lambd2 * t)) - np.exp(-lambd2 * t)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3