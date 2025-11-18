using DataFrames
using CSV
using LinearAlgebra
using StatsBase

function run_expost_11_2()

    stockRet = CSV.read("test11_2_stock_returns.csv", DataFrame)
    factorRet = CSV.read("test11_2_factor_returns.csv", DataFrame)
    betaDF = CSV.read("test11_2_beta.csv", DataFrame)
    stocks = names(stockRet)
    factors = names(factorRet)

    betaStocks = String.(betaDF.Stock)
    order = [findfirst(==(s), betaStocks) for s in stocks]
    Betas = Matrix(betaDF[order, factors])


    matReturns = Matrix(stockRet[:, stocks])
    ffReturns = Matrix(factorRet[:, factors])
    T, nStock = size(matReturns)
    _, nFactor = size(ffReturns)

    lastW = fill(1.0 / nStock, nStock)
    weights = Array{Float64}(undef, T, nStock)
    factorWeights = Array{Float64}(undef, T, nFactor)
    pReturn = zeros(T)
    residReturn = zeros(T)

    for t in 1:T

        weights[t, :] = lastW

        factorWeights[t, :] = (lastW' * Betas)'

        lastW = lastW .* (1 .+ matReturns[t, :])
        pR = sum(lastW)
        lastW = lastW ./ pR
        pReturn[t] = pR - 1.0

        residReturn[t] = pReturn[t] - factorWeights[t, :]' * ffReturns[t, :]
    end


    totalRet = exp(sum(log.(pReturn .+ 1.0))) - 1.0


    k = log(totalRet + 1.0) / totalRet
    carinoK = log.(1.0 .+ pReturn) ./ pReturn ./ k


    attrib = DataFrame(ffReturns .* factorWeights .* carinoK, factors)
    attrib[!, :Alpha] = residReturn .* carinoK


    retDF = DataFrame(factorRet)
    retDF[!, :Alpha] = residReturn
    retDF[!, :Portfolio] = pReturn

    newFactors = vcat(factors, "Alpha")

    Attribution = DataFrame(Value=["TotalReturn", "Return Attribution"])

    for s in vcat(newFactors, "Portfolio")
        col = Symbol(s)
        # 每个因子/Alpha/组合的几何总收益
        tr = exp(sum(log.(retDF[!, col] .+ 1.0))) - 1.0
        # 归因收益（组合本身就等于总收益）
        atr = s == "Portfolio" ? tr : sum(attrib[!, col])
        Attribution[!, col] = [tr, atr]
    end

    # 6. 波动率归因（Vol Attribution）
    Y = hcat(ffReturns .* factorWeights, residReturn)
    X = hcat(fill(1.0, T), pReturn)
    B = (inv(X' * X)*X'*Y)[2, :]
    pStd = std(pReturn)
    cSD = B * pStd


    volRow = DataFrame(Value=["Vol Attribution"])
    for (j, s) in enumerate(newFactors)
        volRow[!, Symbol(s)] = [cSD[j]]
    end
    volRow[!, :Portfolio] = [pStd]

    Attribution = vcat(Attribution, volRow)



    CSV.write("testout_11.2 student.csv", Attribution)
    println(Attribution)
end


run_expost_11_2()
