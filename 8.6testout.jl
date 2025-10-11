#8.6 Simulation 
using CSV, DataFrames, Distributions, LinearAlgebra
using JuMP, Ipopt

include("RiskStats.jl")
include("fitted_model.jl")

function ES_from_simulation_86(data_path::String, ref_path::String)
    # read data
    df = CSV.read("test7_2.csv", DataFrame)
    x = Vector(df.x1)

    μ = mean(x)

    # 2)  t distribution（MLE ）
    fm = fit_general_t(x)
    d = fm.errorModel

    # Simulation 
    alpha = 0.05
    N = 10_000_000
    u = range(1 / (2N), step=1 / N, length=N)
    sim = quantile.(Ref(d), u)

    # ES
    es_abs = ES(sim; alpha=alpha)
    es_dm = es_abs + μ

    # 5)print
    println("ES Absolute,ES Diff from Mean")
    println("$(es_abs),$(es_dm)")

end


ES_from_simulation_86("test7_2.csv", "")

println("These two numbers being very close indicates that the simulation results are stable. ES Absolute focuses more on absolute losses, while ES Diff from Mean focuses more on relative performance.")

