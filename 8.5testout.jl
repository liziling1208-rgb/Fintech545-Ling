#8.5
#put "RiskStats.jl" “fitted_model.jl" in the same file
using CSV, DataFrames, Distributions, Statistics
using QuadGK
using JuMP, Ipopt
using StatsBase
include("RiskStats.jl")
include("fitted_model.jl")

# read data
df = CSV.read("test7_2.csv", DataFrame)
x = Vector(df.x1)

α = 0.05
μ = mean(x)

# fit_general_t
fm = fit_general_t(x)
d = fm.errorModel   # LocationScale(TDist(ν), μ_d, σ_d)

#Diff from Mean
μd, σd, inner = params(d)
ν = params(inner)[1]
d0 = TDist(ν) * σd + 0.0

#ES/Absolute
es_diff_mean = ES(d0; alpha=α)
es_absolute = es_diff_mean - μ
# print
println("ES Absolute,ES Diff from Mean")
println("$(round(es_absolute, digits=6)),$(round(es_diff_mean, digits=6))")
