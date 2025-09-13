# 7.1
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/liziling/Desktop/Duke/fintech 545/Assignment 1/test7_1.csv",
                   skiprows=1, header=None).values.flatten()

# (unbiased estimate, ddof=1)
mu = np.mean(data)
sigma = np.std(data, ddof=1)

print("mu =", mu)
print("sigma =", sigma)
