import pandas as pd
import numpy as np

# read data
df = pd.read_csv('test12_1.csv')


def american_option_price(S, K, T, r, q, sigma, option_type, N=200):

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Initialize asset price list
    asset_prices = np.zeros(N + 1)
    for i in range(N + 1):
        asset_prices[i] = S * (u ** (N - i)) * (d ** i)

    # Initialize option value
    option_values = np.zeros(N + 1)
    if option_type == 'Call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            price = S * (u ** (j - i)) * (d ** i)
            hold_value = np.exp(-r * dt) * (p *
                                            option_values[i] + (1 - p) * option_values[i + 1])
            if option_type == 'Call':
                exercise_value = max(price - K, 0)
            else:
                exercise_value = max(K - price, 0)
            option_values[i] = max(hold_value, exercise_value)

    return option_values[0]


def calculate_greeks(S, K, T, r, q, sigma, option_type, N=200):

    # price
    price = american_option_price(S, K, T, r, q, sigma, option_type, N)

    # Delta
    dS = 0.01
    price_up = american_option_price(S + dS, K, T, r, q, sigma, option_type, N)
    price_down = american_option_price(
        S - dS, K, T, r, q, sigma, option_type, N)
    delta = (price_up - price_down) / (2 * dS)

    # Gamma
    dS_gamma = 1
    price_up_g = american_option_price(
        S + dS_gamma, K, T, r, q, sigma, option_type, N)
    price_down_g = american_option_price(
        S - dS_gamma, K, T, r, q, sigma, option_type, N)
    gamma = (price_up_g - 2 * price + price_down_g) / (dS_gamma ** 2)

    # Vega: ∂V/∂σ
    dsigma = 0.01
    price_sigma_up = american_option_price(
        S, K, T, r, q, sigma + dsigma, option_type, N)
    vega = (price_sigma_up - price) / dsigma

    # Rho
    dr = 0.01
    price_r_up = american_option_price(
        S, K, T, r + dr, q, sigma, option_type, N)
    price_r_down = american_option_price(
        S, K, T, r - dr, q, sigma, option_type, N)

    rho_numerical = (price_r_up - price_r_down) / (2 * dr)
    rho = -rho_numerical / 100

    # Theta
    dt_theta = 1 / 365
    if T > dt_theta:
        price_t = american_option_price(
            S, K, T - dt_theta, r, q, sigma, option_type, N)
        theta = (price - price_t) * 365
    else:
        theta = 0

    return price, delta, gamma, vega, rho, theta


# output
results = []

for idx, row in df.iterrows():
    if pd.isna(row['Option Type']):
        break

    S = row['Underlying']
    K = row['Strike']
    T = row['DaysToMaturity'] / row['DayPerYear']
    r = row['RiskFreeRate']
    q = row['DividendRate']
    sigma = row['ImpliedVol']
    option_type = row['Option Type']

    price, delta, gamma, vega, rho, theta = calculate_greeks(
        S, K, T, r, q, sigma, option_type)

    results.append({
        'ID': int(row['ID']),
        'Value': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    })

# 保存结果
output_df = pd.DataFrame(results)
output_df.to_csv('testout12.2——student.csv', index=False)

print("Done，results in file testout12_2.csv")
