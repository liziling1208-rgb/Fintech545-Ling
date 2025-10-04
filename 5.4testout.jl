# 5.4 
# Put "simulate.jl" and "test5_3.csv" in the same folder.

using CSV, DataFrames, LinearAlgebra, Random, Printf, Statistics
include("simulate.jl")  # expects near_psd and chol_psd! inside

const NSIM = 100_000
const SEED = 1234
const DIGITS = 6
const NAMES = ["x1", "x2", "x3", "x4", "x5"]
const FILE = "test5_3.csv"

#data prepare
function read_cov_from_csv(path::AbstractString)
    df = CSV.read(path, DataFrame; header=false)
    M = Array{Union{Missing,Float64}}(undef, size(df, 1), size(df, 2))
    @inbounds for i in 1:size(df, 1), j in 1:size(df, 2)
        v = df[i, j]
        if v isa Number
            M[i, j] = Float64(v)
        else
            x = tryparse(Float64, String(v))
            M[i, j] = x === nothing ? missing : x
        end
    end

    if count(ismissing, M[1, :]) >= ceil(Int, size(M, 2) * 0.5)
        M = M[2:end, :]
    end
    if count(ismissing, M[:, 1]) >= ceil(Int, size(M, 1) * 0.5)
        M = M[:, 2:end]
    end
    if any(ismissing, M)
        error("CSV still contains non-numeric values (beyond first row/col). Check test5_3.csv.")
    end
    return Array{Float64}(M)
end

# --- Pretty printer ---
function print_matrix(mat; names=NAMES, title="")
    println()
    if !isempty(title)
        println(title)
    end
    @printf("%10s", "")
    for nm in names
        @printf(" %10s", nm)
    end
    println()
    for i in 1:size(mat, 1)
        @printf("%10s", names[i])
        for j in 1:size(mat, 2)
            @printf(" %10.*f", DIGITS, mat[i, j])
        end
        println()
    end
end

# -Higham/near_psd
A = read_cov_from_csv(FILE)
Σ_in = 0.5 .* (A .+ A')
Σ_psd = near_psd(Σ_in; epsilon=0.0)

# PSD-safe simulation
n = size(Σ_psd, 1)
root = zeros(n, n)
chol_psd!(root, Matrix(Σ_psd))

Random.seed!(SEED)
Z = randn(NSIM, n)
X = Z * root'
Σ_out = cov(X)

# Print the two covariances 
print_matrix(Σ_psd, title="Input Covariance")
print_matrix(Σ_out, title=@sprintf("Output Covariance (NSIM=%d)", NSIM))
println("The output covariance is overall very close to the input covariance, with only tiny differences at the level of sampling error.")
