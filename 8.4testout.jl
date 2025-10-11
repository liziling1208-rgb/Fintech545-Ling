# 8.4
#put “RiskStats.jl”  "test7_1.csv" in the same file
using CSV, DataFrames, Distributions, Statistics
using QuadGK
include("RiskStats.jl")
# read data
df = CSV.read("test7_1.csv", DataFrame)
x = Vector(df.x1)

μ = mean(x)
σ = std(x)
α = 0.05

#different from mean
es_diff_mean = ES(Normal(0, σ); alpha=α)
# Absolute 
es_absolute = es_diff_mean - μ

# print
println("ES Absolute,ES Diff from Mean")
println("$(round(es_absolute, digits=6)),$(round(es_diff_mean, digits=6))")
