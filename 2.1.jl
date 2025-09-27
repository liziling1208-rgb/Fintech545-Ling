# testout2.1
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using LinearAlgebra
using Printf

########### change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test2.csv"


# read，data prepare
df = CSV.read(infile, DataFrame)
df_rev = df[end:-1:1, :]
X = Matrix(df_rev)
n, m = size(X)
λ = 0.97

# weight：w_i = (1-λ) * λ^(i-1)
w = [(1 - λ) * λ^(i - 1) for i in 1:n]
w ./= sum(w)

# mean
μ = vec(sum(X .* w, dims=1))
Xc = X .- μ'
W = Diagonal(w)
Σ = Xc' * W * Xc

#demical
names_vec = names(df)
@printf("=== Exponentially Weighted Covariance Matrix (λ=%.2f) ===\n", λ)
println("variable: ", names_vec)

# header
header = ["      "; string.(names_vec)]
println(join(header, "  "))

# 16 demical
for i in 1:m
    row = String[]
    push!(row, rpad(string(names_vec[i]), 6))
    for j in 1:m
        push!(row, @sprintf("%.16f", Σ[i, j]))
    end
    println(join(row, "  "))
end
