# testout3.2
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.
using CSV
using DataFrames
using LinearAlgebra
using Printf

########### change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/testout_1.4.csv"

# read data
raw_txt = open(infile, "r") do io
    read(io, String)
end
A_df = CSV.read(infile, DataFrame)
A = Matrix{Float64}(A_df)

# check PSD
tol = 1e-12
is_psd_already = eigmin(Symmetric(A)) >= -tol

#not PSD
function near_psd_spectral_truncation(C::AbstractMatrix{<:Real}; epsilon::Float64=0.0)
    vals, vecs = eigen(Symmetric(Matrix{Float64}(C)))
    vals = max.(vals, epsilon)
    return vecs * Diagonal(vals) * vecs'
end

if is_psd_already


    #print 
    println("=== Near-PSD Covariance (from testout1.4) — echoed (already PSD) ===")
    print(raw_txt)
else
    Σ = near_psd_spectral_truncation(A; epsilon=0.0)
    println("=== Near-PSD Covariance (from testout1.4) — spectral truncation ===")
    m = size(Σ, 1)
    for i in 1:m
        for j in 1:m
            @printf("% .16f%s", Σ[i, j], j == m ? "\n" : ", ")
        end
    end
end
