using CSV
using DataFrames
using Distributions

# read data
inp = CSV.read("test12_1.csv", DataFrame)

# GBSM 
function gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
    d1 = (log(underlying / strike) + (b + ivol^2 / 2) * ttm) / (ivol * sqrt(ttm))
    d2 = d1 - ivol * sqrt(ttm)
    if call
        return underlying * exp((b - rf) * ttm) * cdf(Normal(), d1) -
               strike * exp(-rf * ttm) * cdf(Normal(), d2)
    else
        return strike * exp(-rf * ttm) * cdf(Normal(), -d2) -
               underlying * exp((b - rf) * ttm) * cdf(Normal(), -d1)
    end
end

# Greeks
function gbsm_greeks(call::Bool, S, K, T, r, b, σ)
    √T = sqrt(T)
    d1 = (log(S / K) + (b + σ^2 / 2) * T) / (σ * √T)
    d2 = d1 - σ * √T
    ϕ = pdf(Normal(), d1)
    Nd1 = cdf(Normal(), d1)
    Nd2 = cdf(Normal(), d2)
    disc_b_r = exp((b - r) * T)
    disc_r = exp(-r * T)

    # Value
    value = gbsm(call, S, K, T, r, b, σ)

    # Delta
    delta = call ? (disc_b_r * Nd1) : (disc_b_r * (Nd1 - 1.0))

    # Gamma
    gamma = disc_b_r * ϕ / (S * σ * √T)

    # Vega (per 1.0 volatility)
    vega = S * disc_b_r * ϕ * √T

    # Rho
    rho = call ? (K * T * disc_r * Nd2) : (-K * T * disc_r * cdf(Normal(), -d2))

    # Theta
    theta_time = -(S * disc_b_r * ϕ * σ) / (2 * √T)
    if call
        theta_carry = -(b - r) * S * disc_b_r * Nd1
        theta_rate = -r * K * disc_r * Nd2
        theta = theta_time + theta_carry + theta_rate
    else
        theta_carry = (b - r) * S * disc_b_r * cdf(Normal(), -d1)
        theta_rate = r * K * disc_r * cdf(Normal(), -d2)
        theta = theta_time + theta_carry + theta_rate
    end

    return value, delta, gamma, vega, rho, theta
end



# output
filter!(row -> !(ismissing(row.ID) || ismissing(row."Option Type")), inp)

inp[!, :T] = Float64.(inp.DaysToMaturity) ./ Float64.(inp.DayPerYear)
inp[!, :b] = Float64.(inp.RiskFreeRate) .- Float64.(inp.DividendRate)

vals = Vector{Float64}(undef, nrow(inp))
deltas = similar(vals)
gammas = similar(vals)
vegas = similar(vals)
rhos = similar(vals)
thetas = similar(vals)

for i in 1:nrow(inp)
    optype = String(inp[i, "Option Type"])
    call = lowercase(optype) == "call"

    S = Float64(inp[i, :Underlying])
    K = Float64(inp[i, :Strike])
    T = Float64(inp[i, :T])
    r = Float64(inp[i, :RiskFreeRate])
    b = Float64(inp[i, :b])
    σ = Float64(inp[i, :ImpliedVol])

    v, dlt, gmm, vga, rho, th = gbsm_greeks(call, S, K, T, r, b, σ)
    vals[i] = v
    deltas[i] = dlt
    gammas[i] = gmm
    vegas[i] = vga
    rhos[i] = rho
    thetas[i] = th
end

out = DataFrame(
    ID=Int.(inp.ID),
    Value=vals,
    Delta=deltas,
    Gamma=gammas,
    Vega=vegas,
    Rho=rhos,
    Theta=thetas
)



CSV.write("testout12.1———student.csv", out)
println("Done. Wrote testout12———1 student.csv")
