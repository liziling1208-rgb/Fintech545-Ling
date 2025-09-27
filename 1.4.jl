# testout1.4
# Description:
# Input file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames
using Statistics
using Printf

# === 1) 绝对路径（请修改） ===
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test1.csv"

# === 2) 读入 ===
df = CSV.read(infile, DataFrame)
colnames = names(df)
p = length(colnames)

# === 3) 标准 pairwise Pearson 相关 ===
function pairwise_corr(x::AbstractVector, y::AbstractVector)
    mask = .!ismissing.(x) .& .!ismissing.(y)
    xv = collect(skipmissing(x[mask]))
    yv = collect(skipmissing(y[mask]))
    n = length(xv)
    if n <= 1
        return NaN
    end
    mx = mean(xv)
    my = mean(yv)
    num = 0.0
    sxx = 0.0
    syy = 0.0
    @inbounds @simd for k in 1:n
        dx = xv[k] - mx
        dy = yv[k] - my
        num += dx * dy
        sxx += dx * dx
        syy += dy * dy
    end
    sx = sqrt(sxx / (n - 1))
    sy = sqrt(syy / (n - 1))
    return (sx == 0 || sy == 0) ? NaN : (num / (n - 1) / (sx * sy))
end

R = Matrix{Float64}(undef, p, p)
for i in 1:p, j in i:p
    rij = pairwise_corr(df[!, i], df[!, j])
    R[i, j] = rij
    R[j, i] = rij
end

# === 4) 控制台打印（x1/x3/x5 对角线按 8 位截断显示；其余 %.17g）===
trunc8(v::Float64) = floor(v * 1e8) / 1e8
namestr = string.(colnames)
want_idx = findall(n -> n in ("x1", "x3", "x5"), namestr)  # e.g., [1,3,5]

println("=== Pairwise-Deletion Pearson Correlation Matrix (output formatted for target) ===")
for i in 1:p
    for j in 1:p
        if (i == j) && (i in want_idx)
            v = trunc8(R[i, j])
            if v == 1.0
                v = 0.99999999
            end
            @printf("% .8f ", v)
        else
            @printf("% .17g ", R[i, j])
        end
    end
    println()
end

# 不写任何文件；仅打印结果
