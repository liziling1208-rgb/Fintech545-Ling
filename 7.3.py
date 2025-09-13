# 7.3
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

csv_path = '/Users/liziling/Desktop/Duke/fintech 545/Assignment 1/test7_3.csv'

df = pd.read_csv(csv_path)
X_raw = df[['x1', 'x2', 'x3']].to_numpy()
y = df['y'].to_numpy()
n = len(y)


X = np.column_stack([np.ones(n), X_raw])
p = X.shape[1]


def neg_log_likelihood_u(theta):
    log_sigma = theta[0]
    log_nu_m2 = theta[1]
    beta = theta[2:]
    sigma = np.exp(log_sigma)        # > 0
    nu = np.exp(log_nu_m2) + 2.0     # > 2
    e = y - X @ beta
    z = e / sigma

    ll = np.sum(t.logpdf(z, df=nu) - np.log(sigma))
    return -ll


alpha0 = 0.04263420272554886
b10 = 0.9748885991199169
b20 = 2.041192163366157
b30 = 3.1548013492762172
sigma0 = 0.04854807936130706
nu0 = 4.598293021968668

theta0 = np.array([
    np.log(sigma0),
    np.log(nu0 - 2.0),
    alpha0, b10, b20, b30
], dtype=float)


res = minimize(
    neg_log_likelihood_u,
    x0=theta0,
    method="L-BFGS-B",
    options={'maxiter': 20000, 'ftol': 1e-12}
)

log_sigma_hat, log_nu_m2_hat, *beta_hat = res.x
sigma_hat = float(np.exp(log_sigma_hat))
nu_hat = float(np.exp(log_nu_m2_hat) + 2.0)
alpha_hat = float(beta_hat[0])
b1, b2, b3 = map(float, beta_hat[1:4])


print("mu    =", f"{0.0:.17f}")
print("sigma =", f"{sigma_hat:.17f}")
print("nu    =", f"{nu_hat:.17f}")
print("Alpha =", f"{alpha_hat:.17f}")
print("B1    =", f"{b1:.17f}")
print("B2    =", f"{b2:.17f}")
print("B3    =", f"{b3:.17f}")
