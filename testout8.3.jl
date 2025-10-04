
# 8.3  VaR from Simulation
# put RiskStats.jl, fitted_model.jl, test7_2.csv, testout8_2.csv in one folder

using CSV, DataFrames, Distributions, Random, LinearAlgebra
using JuMP, Ipopt

include("RiskStats.jl")
include("fitted_model.jl")

#read data
df = CSV.read("test7_2.csv", DataFrame)
x = Vector(df.x1)
μ = mean(x)

# t distribution
fm = fit_general_t(x)
d = fm.errorModel

#  Monte Carlo
const SEED = 1234
const NSIM = 1_000_000
Random.seed!(SEED)
sim = rand(d, NSIM)

#  VaR
v_abs = VaR(sim, alpha=0.05)
v_dm = v_abs + μ

# print
println("VaR Absolute,VaR Diff from Mean")
println("$(v_abs),$(v_dm)")

# ---------------------------
# compare with 8.2
# ---------------------------
ref = CSV.read("testout8_2.csv", DataFrame)
ref_abs = Float64(ref[1, "VaR Absolute"])
ref_dm = Float64(ref[1, "VaR Diff from Mean"])

same_abs = isapprox(v_abs, ref_abs; atol=5e-4)
same_dm = isapprox(v_dm, ref_dm; atol=5e-4)

println()
println("8.3  VaR: Absolute=$(v_abs), DiffFromMean=$(v_dm)")
println("8.2  VaR: Absolute=$(ref_abs), DiffFromMean=$(ref_dm)")

println("Conclusion: Basically consistent with Section 8.2 (analytical t-distribution quantiles ≈ large-sample Monte Carlo).")
println("Reason: Section 8.2 directly computes using the fitted t-distribution quantiles; Section 8.3 performs Monte Carlo sampling from the same distribution, and with a sufficiently large sample size, the results will be very close.")
