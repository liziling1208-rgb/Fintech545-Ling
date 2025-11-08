using CSV
using DataFrames

# input
infile = "test12_3.csv"


# No dividends
function bt_american(call::Bool, S, K, T, r, b, sigma, N)
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    pu = (exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = exp(-r * dt)
    z = call ? 1.0 : -1.0

    nNode(n) = ((n + 1) * (n + 2)) ÷ 2
    idx(i, j) = nNode(j - 1) + i + 1
    V = Vector{Float64}(undef, nNode(N))

    for j in N:-1:0
        for i in j:-1:0
            k = idx(i, j)
            St = S * u^i * d^(j - i)
            ex = max(0.0, z * (St - K))
            if j == N
                V[k] = ex
            else
                cont = df * (pu * V[idx(i + 1, j + 1)] + pd * V[idx(i, j + 1)])
                V[k] = max(ex, cont)
            end
        end
    end
    return V[1]
end

#Discrete Cash Dividends
function bt_american(call::Bool, S, K, T, r,
    divAmts::Vector{Float64}, divSteps::Vector{Int64},
    sigma, N)
    if isempty(divAmts) || isempty(divSteps) || divSteps[1] > N
        return bt_american(call, S, K, T, r, r, sigma, N)
    end

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    pu = (exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = exp(-r * dt)
    z = call ? 1.0 : -1.0

    nNode(n) = ((n + 1) * (n + 2)) ÷ 2
    idx(i, j) = nNode(j - 1) + i + 1

    nStop = divSteps[1]
    V = Vector{Float64}(undef, nNode(nStop))

    for j in nStop:-1:0
        for i in j:-1:0
            k = idx(i, j)
            St = S * u^i * d^(j - i)
            if j < nStop
                ex = max(0.0, z * (St - K))
                cont = df * (pu * V[idx(i + 1, j + 1)] + pd * V[idx(i, j + 1)])
                V[k] = max(ex, cont)
            else
                S_after = max(St - divAmts[1], 0.0)
                nextVal = bt_american(
                    call,
                    S_after,
                    K,
                    T - nStop * (T / N),
                    r,
                    length(divSteps) > 1 ? divAmts[2:end] : Float64[],
                    length(divSteps) > 1 ? (divSteps[2:end] .- nStop) : Int64[],
                    sigma,
                    N - nStop
                )
                ex = max(0.0, z * (St - K))
                V[k] = max(nextVal, ex)
            end
        end
    end
    return V[1]
end


parse_vec_float(s) = (ismissing(s) || s == "" ? Float64[] :
                      parse.(Float64, split(replace(String(s), '；' => ';', '，' => ','), r"[;,]"; keepempty=false)))
parse_vec_int(s) = (ismissing(s) || s == "" ? Int64[] :
                    parse.(Int64, split(replace(String(s), '；' => ';', '，' => ','), r"[;,]"; keepempty=false)))
days_to_steps(divDays::Vector{Int}, N::Int) = [clamp(d, 1, N) for d in divDays]


outfile = "testout12.3———student.csv"  

df = CSV.read(infile, DataFrame)

@assert all([:ID, Symbol("Option Type"), :Underlying, :Strike, :DaysToMaturity, :DayPerYear, :RiskFreeRate, :ImpliedVol, :DividendDates, :DividendAmts] .∈ Ref(Symbol.(names(df)))) "输入列名与示例不一致"

prices = Vector{Float64}(undef, nrow(df))

for r in 1:nrow(df)
    S = Float64(df[r, :Underlying])
    K = Float64(df[r, :Strike])
    days = Int(df[r, :DaysToMaturity])
    dpy = Float64(df[r, :DayPerYear])
    T = days / dpy
    rate = Float64(df[r, :RiskFreeRate])
    sigma = Float64(df[r, :ImpliedVol])
    N = max(days, 1)
    isCall = uppercase(strip(String(df[r, Symbol("Option Type")]))) |> x -> x[1] == 'C'

    divDays = parse_vec_int(df[r, :DividendDates])
    divAmts = parse_vec_float(df[r, :DividendAmts])
    m = min(length(divDays), length(divAmts))
    divSteps = days_to_steps(m > 0 ? divDays[1:m] : Int[], N)
    divCash = m > 0 ? divAmts[1:m] : Float64[]

    prices[r] = bt_american(isCall, S, K, T, rate, divCash, divSteps, sigma, N)
end

out = df[:, [:ID, Symbol("Option Type"), :Underlying, :Strike, :DaysToMaturity, :DayPerYear, :RiskFreeRate, :ImpliedVol, :DividendDates, :DividendAmts]]
out.Price = prices
CSV.write(outfile, out)
println("Done，results in file12.3——student)
