using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Ipopt

cov_df = CSV.read("test5_2.csv", DataFrame)
covar = Matrix(cov_df[:, :])
n = size(covar, 1)


rb = ones(n)
rb[end] = 0.5


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


function sseCSD2(w...)
    csd = pCSD(w...) ./ rb
    mCSD = sum(csd) / n
    dCsd = csd .- mCSD
    se = dCsd .* dCsd
    return 1.0e5 * sum(se)
end


m = Model(Ipopt.Optimizer)
@variable(m, w[i=1:n] >= 0, start = 1 / n)
register(m, :distSSE, n, sseCSD2; autodiff=true)

@NLobjective(m, Min, distSSE(w...))
@constraint(m, sum(w) == 1.0)

optimize!(m)

w_opt = value.(w)


out = DataFrame(W=w_opt)
CSV.write("testout_10.2 student.csv", out)
