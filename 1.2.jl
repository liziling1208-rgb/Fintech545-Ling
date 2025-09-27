# testout1.2
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using Statistics
using LinearAlgebra


########## change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test1.csv"

df = CSV.read(infile, DataFrame)
df_clean = dropmissing(df)

# Pearson 
X = Matrix(df_clean)
C = cov(X; dims=1)
s = sqrt.(diag(C))
R = C ./ (s * s')

#print
println("=== Pearson Correlation Matrix ===")
println(R)