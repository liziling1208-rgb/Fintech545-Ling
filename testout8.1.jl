#8.1
using CSV
using DataFrames
using Distributions
using Statistics
using Printf

#read data
df = CSV.read("test7_1.csv", DataFrame)
x = df[:, 1]

# VaR
μ = mean(x)
σ = std(x)
VaR_abs = -quantile(Normal(μ, σ), 0.05)
VaR_diff = -quantile(Normal(0, σ), 0.05)

# print
println("VaR Absolute,VaR Diff from Mean")
@printf("%.5f,%.6f\n", VaR_abs, VaR_diff)
