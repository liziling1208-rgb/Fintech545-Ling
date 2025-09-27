# testout3.1
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using LinearAlgebra
using Printf

########### change the absolute path in the infile field

infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/testout_1.3.csv"

# read
df = CSV.read(infile, DataFrame)
A = Matrix{Float64}(df)

#near_psd
function near_psd(a::AbstractMatrix{<:Real}; epsilon::Float64=0.0)
    out = copy(Matrix{Float64}(a))
    n = size(out, 1)

    invSD = nothing


    # Normalized into correlation matrix
    if count(x -> isapprox(x, 1.0; atol=1e-12), diag(out)) != n
        invSD = Diagonal(1.0 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    #  >= epsilon
    vals, vecs = eigen(Symmetric(out))
    vals = max.(vals, epsilon)

    #T, Λ^{1/2} structure）
    T = 1.0 ./ ((vecs .* vecs) * vals)
    T = Diagonal(sqrt.(T))
    l = Diagonal(sqrt.(vals))
    B = T * vecs * l
    out = B * B'
    #Scale back to the covariance
    if invSD !== nothing
        invSD = Diagonal(1.0 ./ diag(invSD))  # inv(invSD)
        out = invSD * out * invSD
    end
    return out
end

#Near PSD Covariance
Σ_psd = near_psd(A; epsilon=0.0)

# print
println("=== Near-PSD Covariance (from testout_1.3) ===")
m = size(Σ_psd, 1)
for i in 1:m
    for j in 1:m
        @printf("% .16f%s", Σ_psd[i, j], j == m ? "\n" : ", ")
    end
end
