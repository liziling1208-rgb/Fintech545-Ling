
# 3.3 Higham 
# testout3.3
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using LinearAlgebra
using Printf


########### infile to the "absolute path" on your computer
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/testout_1.3.csv"

# read data
A_df = CSV.read(infile, DataFrame)
A = Matrix{Float64}(A_df)

# Determine 
is_cov = any(abs.(diag(A) .- 1.0) .> 1e-12)

# it is covariance: standardize it 
D = nothing
C0 = copy(A)
if is_cov
    stds = sqrt.(diag(A))
    D = Diagonal(stds .+ 0.0)
    invD = Diagonal(1.0 ./ stds)
    C0 = invD * A * invD
end

# Higham covariance
function project_U(M::AbstractMatrix{<:Real})
    Y = Matrix{Float64}(M)
    @inbounds for i in 1:size(Y, 1)
        Y[i, i] = 1.0
    end
    return Y
end

function project_S(M::AbstractMatrix{<:Real})
    vals, vecs = eigen(Symmetric(Matrix{Float64}(M)))
    vals = max.(vals, 0.0)
    return vecs * Diagonal(vals) * vecs'
end

function higham_nearest_correlation(C::AbstractMatrix{<:Real}; max_iter=100, tol=1e-10)
    Y = project_U(C)
    ΔS = zeros(size(C))
    γ_prev = Inf
    for _ in 1:max_iter
        R = Y .- ΔS
        X = project_S(R)
        ΔS = X .- R
        Y = project_U(X)
        γ = norm(Y .- X)
        if abs(γ - γ_prev) < tol
            break
        end
        γ_prev = γ
    end
    return Symmetric(Y) |> Matrix
end

#  Higham
C_higham = higham_nearest_correlation(C0; max_iter=200, tol=1e-12)
Σ_higham = is_cov ? (D * C_higham * D) : C_higham

#print
println("=== Higham Nearest-PSD Covariance from testout_1.3 ===")
m = size(Σ_higham, 1)
for i in 1:m
    for j in 1:m
        @printf("% .16f%s", Σ_higham[i, j], j == m ? "\n" : ", ")
    end
end
