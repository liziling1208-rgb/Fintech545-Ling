
# 3.4 Higham correlation 

using CSV
using DataFrames
using LinearAlgebra
using Printf


# 备注：请把 infile 改为您本机“绝对路径”，指向 testout_1.4.csv
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/testout_1.4.csv"


const DECIMALS = 16
const PRINT_HEADER = false
const FORCE_HIGHAM = false

# ---------- 读入 ----------
df = CSV.read(infile, DataFrame)
A = Matrix{Float64}(df)

# ---------- 若为协方差，则先标准化为相关矩阵 ----------
is_cov = any(abs.(diag(A) .- 1.0) .> 1e-12)
C0 = if is_cov
    invD = Diagonal(1.0 ./ sqrt.(diag(A)))
    invD * A * invD
else
    A
end

# ---------- Higham 交替投影（单位对角 + PSD） ----------
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
        γ = norm(Y .- X)   # Frobenius 范数（默认）
        if abs(γ - γ_prev) < tol
            break
        end
        γ_prev = γ
    end
    return Symmetric(Y) |> Matrix
end

# ---------- 生成要输出的相关矩阵 C_out ----------
tol_psd = 1e-12
already_correlation = all(abs.(diag(C0) .- 1.0) .<= 1e-12)
already_psd = eigmin(Symmetric(C0)) >= -tol_psd

C_out = if FORCE_HIGHAM
    higham_nearest_correlation(C0; max_iter=200, tol=1e-12)
elseif already_correlation && already_psd
    C0
else
    higham_nearest_correlation(C0; max_iter=200, tol=1e-12)
end

#print
const DECIMALS = 16
m = size(C_out, 1)

println("=== Higham correlation matrix (precision = 16 decimals) ===")

m = size(C_out, 1)
for i in 1:m
    for j in 1:m
        # 16 decimals
        @printf("%.16f", C_out[i, j])
        if j < m
            print("  ")
        else
            print("\n")
        end
    end
end
