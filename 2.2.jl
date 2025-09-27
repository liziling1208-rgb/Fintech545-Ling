# testout2.2
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using LinearAlgebra
using Printf

########### change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test2.csv"

# read,turn the order,data preprare
df = CSV.read(infile, DataFrame)
df_rev = df[end:-1:1, :]
X = Matrix(df_rev)                    # n × m
n, m = size(X)

# EWMA 
λ = 0.94

# weight
w = [(1 - λ) * λ^(i - 1) for i in 1:n]
w ./= sum(w)
μ = vec(sum(X .* w, dims=1))
Xc = X .- μ'
W = Diagonal(w)
Σ = Xc' * W * Xc

# transform
Dinv = Diagonal(1 ./ sqrt.(diag(Σ)))
R = Dinv * Σ * Dinv

# print
names_vec = names(df)
@printf("=== Exponentially Weighted Correlation Matrix (λ=%.2f) ===\n", λ)
println("variable", names_vec)

# header
header = ["      "; string.(names_vec)]
println(join(header, "  "))


for i in 1:m
    row = String[]
    push!(row, rpad(string(names_vec[i]), 6))
    for j in 1:m
        push!(row, @sprintf("%.17f", R[i, j]))
    end
    println(join(row, "  "))
end

