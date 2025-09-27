# testout1.3
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.


using CSV
using DataFrames
using Statistics

########### change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test1.csv"

# read
df = CSV.read(infile, DataFrame)

#  Pairwise deleted sample covariance matrix 
colnames = names(df)
p = length(colnames)
C = Matrix{Float64}(undef, p, p)

for i in 1:p
    xi = df[!, i]
    for j in i:p
        xj = df[!, j]
        mask = .!ismissing.(xi) .& .!ismissing.(xj)
        xi2 = collect(skipmissing(xi[mask]))
        xj2 = collect(skipmissing(xj[mask]))
        cij = cov(xi2, xj2)
        C[i, j] = cij
        C[j, i] = cij
    end
end

# print
println("=== Pairwise-Deletion Sample Covariance Matrix ===")
display(DataFrame(C, Symbol.(colnames)))
println()
