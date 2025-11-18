using DataFrames
using CSV
using LinearAlgebra

function main()

    returns = CSV.read("test11_1_returns.csv", DataFrame)
    weights_df = CSV.read("test11_1_weights.csv", DataFrame)

    asset_names = names(returns)

    matReturns = Matrix(returns)
    n, m = size(matReturns)

    w0 = Vector{Float64}(weights_df[:, 1])
    weights = Array{Float64,2}(undef, n, m)
    pReturn = Vector{Float64}(undef, n)

    lastW = copy(w0)

    for i in 1:n

        weights[i, :] = lastW

        lastW = lastW .* (1.0 .+ matReturns[i, :])
        pVal = sum(lastW)
        lastW = lastW ./ pVal

        pReturn[i] = pVal - 1.0
    end


    totalAsset = exp.(sum(log.(1 .+ matReturns), dims=1))[:] .- 1.0
    totalRet = exp(sum(log.(1 .+ pReturn))) - 1.0

    # Carino Return Attribution
    k = log(1.0 + totalRet) / totalRet
    carinoK = (log.(1.0 .+ pReturn) ./ pReturn) ./ k

    attribMat = matReturns .* weights .* carinoK
    retAttrib = vec(sum(attribMat, dims=1))



    μp = sum(pReturn) / n
    pStd = sqrt(sum((pReturn .- μp) .^ 2) / (n - 1))
    X = hcat(ones(n), pReturn)
    Y = matReturns .* weights
    B = (X' * X) \ (X' * Y)
    beta = B[2, :]
    volAttrib = beta .* pStd



    out = DataFrame(Value=["TotalReturn", "Return Attribution", "Vol Attribution"])

    for j in 1:m
        nm = asset_names[j]
        out[!, nm] = [totalAsset[j], retAttrib[j], volAttrib[j]]
    end

    out[!, :Portfolio] = [totalRet, totalRet, pStd]
    CSV.write("testout11_1 student.csv", out)
    println(out)
end

main()
