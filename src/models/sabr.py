import numpy as np

def sabr_vol(F, K, T, alpha, beta, rho, nu):
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

def sabr_loss(params, F, T, strikes, market_vols, beta=0.5):
    alpha, rho, nu = params
    model_vols = [sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
    return np.mean((np.array(model_vols) - np.array(market_vols)) ** 2)