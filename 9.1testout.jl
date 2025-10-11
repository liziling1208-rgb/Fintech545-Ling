#9.1
# put “fitted_model.jl”, “simulate.jl”, “RiskStats.jl”in the same file


using CSV, DataFrames, Distributions, LinearAlgebra, StatsBase, Random
using JuMP, Ipopt

include("simulate.jl")
include("RiskStats.jl")
include("fitted_model.jl")
const ALPHA = 0.05
const NSIM = 1_000_000
const SEED = 1234

# read data
portfolio = CSV.read("test9_1_portfolio.csv", DataFrame)
returns = CSV.read("test9_1_returns.csv", DataFrame)
rename!(portfolio, Symbol.(lowercase.(string.(names(portfolio)))))
rename!(returns, Symbol.(replace.(string.(names(returns)), " " => "_")))


# A=2000、B=3000
let s = Symbol.(portfolio.stock)
    for (nm, target) in ((:A, 2000.0), (:B, 3000.0))
        idx = findfirst(==(String(nm)), portfolio.stock)
        if idx === nothing
            push!(portfolio, (; stock=String(nm), holding=target))
        else
            portfolio.holding[idx] = target
        end
    end
end

asset_cols = Symbol.(filter(n -> lowercase(n) != "date", string.(names(returns))))
foreach(nm -> returns[!, nm] = Float64.(returns[!, nm]), asset_cols)



use_nms = [Symbol("A"), Symbol("B")]
market = Symbol("SPY")
if !(market in asset_cols)
    market = Symbol("A")
end

# De-meaning
for nm in union(use_nms, [market])
    v = returns[!, nm]
    returns[!, nm] = v .- mean(v)
end

#A/B Regression on SPY
nms = vcat([market], [nm for nm in use_nms if nm != market])  # 顺序：SPY, A, B
fitted = Dict{Symbol,FittedModel}()
fitted[market] = fit_normal(returns[!, market])
for nm in nms
    if nm == market
        continue
    end
    fitted[nm] = fit_regression_t(returns[!, nm], returns[!, market])
end

#Spearman
U = DataFrame()
for nm in nms
    U[!, nm] = fitted[nm].u
end
R = corspearman(Matrix(U))

#Copula
Random.seed!(SEED)
Z = simulate_pca(R, NSIM; seed=SEED)
simU = DataFrame(cdf.(Normal(), Z), nms)

# return
simRet = DataFrame()
simRet[!, market] = fitted[market].eval(simU[!, market])
for nm in nms
    if nm == market
        continue
    end
    simRet[!, nm] = fitted[nm].eval(simRet[!, market], simU[!, nm])
end

# P&L VaR/ES 
simAB = select(simRet, use_nms)
holding_map = Dict{Symbol,Float64}(Symbol.(portfolio.stock) .=> Float64.(portfolio.holding))
HA = get(holding_map, :A, 0.0)
HB = get(holding_map, :B, 0.0)

pnlA = simAB[:, 1] .* HA
pnlB = simAB[:, 2] .* HB
pnlT = pnlA .+ pnlB
PV = HA + HB

#print
safe_div(x, y) = y == 0 ? NaN : x / y

VaR95_A = VaR(pnlA, alpha=ALPHA);
ES95_A = ES(pnlA, alpha=ALPHA);
VaR95_B = VaR(pnlB, alpha=ALPHA);
ES95_B = ES(pnlB, alpha=ALPHA);
VaR95_T = VaR(pnlT, alpha=ALPHA);
ES95_T = ES(pnlT, alpha=ALPHA);

rowA = ("A", VaR95_A, ES95_A, safe_div(VaR95_A, HA), safe_div(ES95_A, HA))
rowB = ("B", VaR95_B, ES95_B, safe_div(VaR95_B, HB), safe_div(ES95_B, HB))
rowT = ("Total", VaR95_T, ES95_T, safe_div(VaR95_T, PV), safe_div(ES95_T, PV))

out = DataFrame([rowA, rowB, rowT], [:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])

show(out; allrows=true, allcols=true)
println()
