#5.1
using CSV
using DataFrames
using LinearAlgebra
using Random
using Statistics
using Printf

# read Data
df = CSV.read("test5_2.csv", DataFrame)
A = Matrix{Float64}(df)
A = Symmetric(0.5 .* (A + A'))

#  Higham/T-scaling PSD
function near_psd_higham(A; epsilon=0.00866)
    sd = sqrt.(diag(A))
    Dinv = Diagonal(1.0 ./ sd)
    C = Symmetric(Dinv * A * Dinv)

    vals, vecs = eigen(C)
    vals = max.(vals, epsilon)
    Tvec = 1.0 ./ ((vecs .* vecs) * vals)
    T = Diagonal(sqrt.(Tvec))
    L = Diagonal(sqrt.(vals))
    Cpsd = Symmetric((T * vecs * L) * (T * vecs * L)')

    return Symmetric(Diagonal(sd) * Cpsd * Diagonal(sd))
end

Σ = near_psd_higham(A; epsilon=0.00866)

#  PCA Normal Simulation PD Input 0 mean - 100,000 simulations
function simulate_normal_pca(Σ::AbstractMatrix{<:Real}, nsim::Int; rng=MersenneTwister(545))
    vals, vecs = eigen(Symmetric(Σ))
    pos = findall(>=(1e-12), vals)
    B = vecs[:, pos] * Diagonal(sqrt.(vals[pos]))
    r = randn(rng, length(pos), nsim)
    return (B * r)'
end

nsim = 100_000
X = simulate_normal_pca(Σ, nsim)
Σ̂ = Symmetric(cov(X))

# cpmpare: wgtNorm  
function wgtNorm(E::AbstractMatrix{<:Real}, W::Union{UniformScaling,AbstractMatrix}=I)
    if W === I
        return norm(E)
    else
        Wsym = Symmetric(Matrix(W))
        F = cholesky(Wsym; check=false)
        return norm(F.U' * E * F.U)
    end
end

# print
names = [:x1, :x2, :x3, :x4, :x5]

println("Input covariance Σ (near-PSD, used for simulation):")
@printf("%10s %10s %10s %10s %10s\n", string.(names)...)
for i in 1:5
    @printf("%10.6f %10.6f %10.6f %10.6f %10.6f\n",
        Σ[i, 1], Σ[i, 2], Σ[i, 3], Σ[i, 4], Σ[i, 5])
end

println("\nOutput empirical covariance Σ̂ from Normal(0, Σ) with 100,000 sims:")
@printf("%10s %10s %10s %10s %10s\n", string.(names)...)
for i in 1:5
    @printf("%10.6f %10.6f %10.6f %10.6f %10.6f\n",
        Σ̂[i, 1], Σ̂[i, 2], Σ̂[i, 3], Σ̂[i, 4], Σ̂[i, 5])
end

Δ = Array(Σ̂ .- Σ)
rmse = sqrt(mean(Δ .^ 2))
maxe = maximum(abs.(Δ))
wn_I = wgtNorm(Δ, I)

@printf("\nRMSE(Σ̂−Σ): %.8e\n", rmse)
@printf("Max |Σ̂−Σ|: %.8e\n", maxe)
@printf("wgtNorm(Σ̂−Σ, I): %.8e\n", wn_I)
@printf("The output covariance is very close to the input covariance.")