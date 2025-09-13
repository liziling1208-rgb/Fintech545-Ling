# 7.2
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize


data = pd.read_csv('/Users/liziling/Desktop/Duke/fintech 545/Assignment 1/test7_2.csv',
                   skiprows=1, header=None).values.flatten()

# Maximize the log-likelihood by 'taking the negative and minimizing‘，NLL


def neg_log_likelihood(params, data):
    mu, sigma, nu = params
    if sigma <= 0 or nu <= 2:
        return np.inf
    ll = np.sum(t.logpdf((data - mu) / sigma, df=nu) - np.log(sigma))
    return -ll


init_params = [np.mean(data), np.std(data, ddof=1), 5.0]
result = minimize(neg_log_likelihood, init_params, args=(data,),
                  bounds=[(None, None), (1e-6, None), (2.01, None)])


mu_hat, sigma_hat, nu_hat = result.x
print("mu =", mu_hat)
print("sigma =", sigma_hat)
print("nu =", nu_hat)
