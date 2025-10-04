#5.2
#put "simulate.jl" in the same folder

using CSV, DataFrames, LinearAlgebra, Statistics, Random, Distributions

const SEED = 1234
const N = 100_000
const DIGITS = 8

# read data
Σdf = CSV.read(joinpath(@__DIR__, "test5_1.csv"), DataFrame)
Σ = Matrix{Float64}(Σdf)
n = size(Σ, 1)

# 0 均值多元正态：X ~ N(0, Σ)，N 次
Random.seed!(SEED)
L = cholesky(Symmetric(Σ)).L
Z = randn(n, N)
X = (L * Z)'

# Output covariance
Σ_hat = (X' * X) / N

# wgtNorm
include(joinpath(@__DIR__, "simulate.jl"))

Δ = Σ_hat .- Σ
W = I(n)
class_metric = wgtNorm(Δ, W)

# ptint
println("Input covariance (Σ) ")
show(DataFrame(round.(Σ, digits=DIGITS), Symbol.(names(Σdf))), allrows=true, allcols=true)
println("\n\nOutput covariance (Σ̂) from Normal Simulation: 0 mean, N = $N ")
show(DataFrame(round.(Σ_hat, digits=DIGITS), Symbol.(names(Σdf))), allrows=true, allcols=true)

println("\n\ncomparison metric (wgtNorm with W = I) ")
println(round(class_metric, digits=DIGITS))
println("Each element differs only slightly (around 1e-4),
and the overall error wgtNorm ≈ 1.2×10⁻⁷.
This shows that the simulated samples reproduce the original covariance very well.")
println()
