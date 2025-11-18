using DataFrames
using CSV
using LinearAlgebra
using JuMP
using Ipopt


cov_df = CSV.read("test5_2.csv", DataFrame)
covar = Matrix(cov_df[:, :])
n = size(covar, 1)


function pvol(w...)
    x = collect(w)
    return sqrt(x' * covar * x)
end

function pCSD(w...)
    x = collect(w)
    pVol = pvol(w...)
    csd = x .* (covar * x) ./ pVol
    return csd
end

function sseCSD(w...)
    csd = pCSD(w...)
    mCSD = sum(csd) / n
    dCsd = csd .- mCSD
    se = dCsd .* dCsd
    return 1.0e5 * sum(se)
end

m = Model(Ipopt.Optimizer)
set_silent(m)
@variable(m, w[i=1:n] >= 0, start = 1 / n)
register(m, :distSSE, n, sseCSD; autodiff=true)

@NLobjective(m, Min, distSSE(w...))
@constraint(m, sum(w) == 1.0)

optimize!(m)

w_star = value.(w)
w_star ./= sum(w_star)


out_df = DataFrame(W=w_star)
CSV.write("testout_10.1 student.csv", out_df)
