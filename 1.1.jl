# testout1.1
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using Statistics

########## change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test1.csv"

df = CSV.read(infile, DataFrame)
df_clean = dropmissing(df)

# Sample correlation matrix (Pearson)
using LinearAlgebra

X = Matrix(df_clean)
C = cov(X; dims=1)

# print
println("=== Sample Covariance Matrix ===")
colnames = names(df_clean)
outdf = DataFrame(C, Symbol.(colnames))
show(outdf; allrows=true, allcols=true)
println()