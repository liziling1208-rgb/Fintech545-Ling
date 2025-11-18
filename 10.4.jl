using CSV
using DataFrames
using LinearAlgebra
using JuMP
using Ipopt
using ForwardDiff
using Printf


cov_df = CSV.read("test5_2.csv", DataFrame)
covar = Matrix(cov_df[:, :])

mean_df = CSV.read("test10_3_means.csv", DataFrame)
stockMeans = Vector(mean_df.Mean)

rf = 0.04
n = length(stockMeans)

# Sharpe Ratio 
function sr(w...)
    _w = collect(w)
    m = _w' * stockMeans - rf
    s = sqrt(_w' * covar * _w)
    return m / s
end

m = Model(Ipopt.Optimizer)
@variable(m, 0.1 <= w[1:n] <= 0.5, start = 1 / n)

register(m, :sr, n, sr; autodiff=true)
@NLobjective(m, Max, sr(w...))
@constraint(m, sum(w[i] for i in 1:n) == 1.0)

optimize!(m)

w_opt = value.(w)

println("full precision w：")
for i in 1:n
    @printf("w[%d] = %.16f\n", i, w_opt[i])
end

out_df = DataFrame(W=w_opt)
CSV.write("testout10_4 student.csv", out_df)
println(out_df)
