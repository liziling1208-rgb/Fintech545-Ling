#  8.2
# Put fitted_model.jl and RiskStats.jl in the same folder.
using CSV, DataFrames, Distributions, StatsBase
using JuMP, Ipopt
include("fitted_model.jl")
include("RiskStats.jl")
#read data
df = CSV.read("test7_2.csv", DataFrame)
x = Vector{Float64}(df[:, 1])

# t distribution MLE ： R ~ μ + s * T_ν
fm = fit_general_t(x)
d = fm.errorModel

# VaR
alpha = 0.05
VaR_abs = VaR(d; alpha=alpha)
VaR_diff = mean(x) + VaR_abs

#print
using Printf
println("VaR Absolute,VaR Diff from Mean")
@printf("%.5f,%.5f\n", VaR_abs, VaR_diff)
