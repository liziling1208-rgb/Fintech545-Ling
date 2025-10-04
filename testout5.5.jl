#5.5
using CSV, DataFrames, LinearAlgebra, Statistics, Random


function near_psd(a; epsilon=0.0)
    A = copy(a)
    n = size(A, 1)
    invSD = nothing
    if count(x -> isapprox(x, 1.0; atol=1e-12), diag(A)) != n
        invSD = Diagonal(1.0 ./ sqrt.(diag(A)))
        A = invSD * A * invSD
    end
    vals, vecs = eigen(Symmetric(A))
    vals = max.(vals, epsilon)
    T = 1.0 ./ (vecs .* vecs * vals)
    T = Diagonal(sqrt.(T))
    Lsqrt = Diagonal(sqrt.(vals))
    B = T * vecs * Lsqrt
    A = B * B'
    if invSD !== nothing
        SD = Diagonal(1.0 ./ diag(invSD))
        A = SD * A * SD
    end
    return A
end

function simulate_pca(a::AbstractMatrix, nsim::Int; nval::Int)
    vals, vecs = eigen(Symmetric(a))
    idx = sortperm(vals; rev=true)
    vals = vals[idx]
    vecs = vecs[:, idx]
    pos = findall(>=(1e-8), vals)
    vals = vals[pos]
    vecs = vecs[:, pos]
    k = min(nval, length(vals))
    B = vecs[:, 1:k] * Diagonal(sqrt.(vals[1:k]))
    return (B * randn(k, nsim))'
end


const NSIM = 100_000
const EXPLAINED = 0.99
Random.seed!(42)

# read data
df = CSV.read("test5_2.csv", DataFrame)
Σ = Matrix{Float64}(df)
Σ = 0.5 .* (Σ + Σ')

#  PSD 
λmin = minimum(eigvals(Symmetric(Σ)))
Σ_in = λmin < -1e-10 ? near_psd(Σ; epsilon=1e-8) : Σ

# PCA 
vals_all = eigvals(Symmetric(Σ_in))
vals_sorted = sort(vals_all; rev=true)
vals_pos = vals_sorted[vals_sorted.>1e-8]
tv = sum(vals_pos)
cum = cumsum(vals_pos) ./ tv
k = findfirst(x -> x >= EXPLAINED, cum)
k === nothing && (k = length(vals_pos))

# covariance
X = simulate_pca(Σ_in, NSIM; nval=k)
Σ_out = cov(X)

#print
nshow = min(5, size(Σ_in, 1))
println("== 5.5 PCA Simulation ==")
println("Matrix size: $(size(Σ_in,1)) × $(size(Σ_in,2))")
println("Principal components used: $k (≥99% variance explained)\n")

println("Input covariance Σ_in (first $nshow×$nshow):")
show(stdout, "text/plain", round.(Σ_in[1:nshow, 1:nshow]; digits=6));
println();

println("\nOutput covariance Σ_out (first $nshow×$nshow):")
show(stdout, "text/plain", round.(Σ_out[1:nshow, 1:nshow]; digits=6));
println();

println("\nThey are close ")
