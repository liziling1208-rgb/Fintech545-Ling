# testout4.1
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using LinearAlgebra
using Printf

########### change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/testout_3.1.csv"

#

df = CSV.read(infile, DataFrame)
A = Matrix{Float64}(df)
n = size(A, 1)

# PSD Cholesky
function chol_psd!(root::AbstractMatrix{<:Real}, a::AbstractMatrix{<:Real})
    n = size(a, 1)
    root .= 0.0
    @inbounds for j in 1:n
        s = 0.0
        if j > 1
            s = root[j, 1:(j-1)]' * root[j, 1:(j-1)]
        end
        temp = a[j, j] - s
        if (temp ≤ 0.0) && (temp ≥ -1e-8)
            temp = 0.0
        end
        rjj = sqrt(temp)
        root[j, j] = rjj

        if rjj != 0.0
            ir = 1.0 / rjj
            @inbounds for i in (j+1):n
                s = 0.0
                if j > 1
                    s = root[i, 1:(j-1)]' * root[j, 1:(j-1)]
                end
                root[i, j] = (a[i, j] - s) * ir
            end
        end
    end
    return root
end


L = Array{Float64}(undef, n, n)
chol_psd!(L, A)
function clamp_small!(M; tol=1e-14)
    @inbounds for i in eachindex(M)
        if abs(M[i]) < tol
            M[i] = 0.0
        end
    end
    return M
end
clamp_small!(L)

println("=== Cholesky PSD Root (precision = 16 decimals) ===")
for i in 1:n
    for j in 1:n
        @printf("% .16f", L[i, j])
        if j < n
            print("  ")
        else
            print('\n')
        end
    end
end
