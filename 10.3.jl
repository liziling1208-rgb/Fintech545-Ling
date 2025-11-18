using DataFrames
using CSV
using LinearAlgebra
using JuMP
using Ipopt
using ForwardDiff

rf = 0.04
cov_df = CSV.read("test5_2.csv", DataFrame)
covar = Matrix(cov_df)
mu_df = CSV.read("test10_3_means.csv", DataFrame)
mu = mu_df.Mean


# Sharpe Ratio 
function sr(w...)
    _w = collect(w)
    m = _w' * mu - rf
    s = sqrt(_w' * covar * _w)
    return m / s

    n = length(mu)


    m = Model(Ipopt.Optimizer)


    @variable(m, w[i=1:n] >= 0.0, start = 1.0 / n)
    register(m, :sr, n, sr; autodiff=true)

    @NLobjective(m, Max, sr(w...))
    @constraint(m, sum(w) == 1.0)

    optimize!(m)
    w_opt = value.(w)
    out = DataFrame(W=w_opt)
    CSV.write("testout_10.3 student.csv", out)
end