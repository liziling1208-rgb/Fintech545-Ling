# testout2.3
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.
using CSV
using DataFrames
using LinearAlgebra
using Printf
########### change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test2.csv"
# Reverse the time order
df = CSV.read(infile, DataFrame)
df_rev = df[end:-1:1, :]
X = Matrix(df_rev)
n, m = size(X)

# EWMAcovariance
function ewma_cov(X::AbstractMatrix{T}, λ::T) where {T<:AbstractFloat}
    n, m = size(X)
    w = [(one(T) - λ) * λ^(i - 1) for i in 1:n]
    w ./= sum(w)
    μ = vec(sum(X .* w, dims=1))
    Xc = X .- μ'
    return Xc' * Diagonal(w) * Xc
end

#  λ_var=0.97 
λ_var = 0.97
Σ97 = ewma_cov(X, λ_var)
vars97 = diag(Σ97)
D97_sqrt = Diagonal(sqrt.(vars97))

#EWMA correlation matrix λ_corr = 0.94
λ_corr = 0.94
Σ94 = ewma_cov(X, λ_corr)
D94_inv = Diagonal(1 ./ sqrt.(diag(Σ94)))
R94 = D94_inv * Σ94 * D94_inv

#cov
Cov = D97_sqrt * R94 * D97_sqrt

# demical 16
names_vec = names(df)
@printf("=== Hybrid EWMA Covariance (Var λ=%.2f, Corr λ=%.2f) ===\n", λ_var, λ_corr)
println("variable: ", names_vec)

header = ["      "; string.(names_vec)]
println(join(header, "  "))

for i in 1:m
    row = String[]
    push!(row, rpad(string(names_vec[i]), 6))
    for j in 1:m
        push!(row, @sprintf("%.17f", Cov[i, j]))   # ← 改成 "%.16f" 得到更高精度
    end
    println(join(row, "  "))
end


