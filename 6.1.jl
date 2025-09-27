# testout6.1
# Description:
# Input/out file test1.csv: Please enter the absolute path in the infile field.

using CSV
using DataFrames

############ change the absolute path in the infile field
infile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/test6.csv"
outfile = "/Users/liziling/Desktop/Duke/fintech 545/Assignment 2/testout6_1.csv"

# check latest
latest_first = false
#read data
df0 = CSV.read(infile, DataFrame; delim=',', normalizenames=false)
df = latest_first ? df0[end:-1:1, :] : df0

# date/price prepare
names_lower = lowercase.(string.(names(df)))
date_idx = findfirst(n -> n in ("date", "time", "datetime"), names_lower)
date_idx === nothing && (date_idx = 1)

dates_all = df[!, date_idx]
prices_df = select(df, Not(date_idx))

# return
function arithmetic_returns(df_prices::DataFrame)
    n, m = size(df_prices)
    n < 2 && error("lack of data")
    ret = DataFrame()
    for cname in names(df_prices)
        col = df_prices[!, cname]
        r = Vector{Union{Missing,Float64}}(undef, n - 1)
        @inbounds for i in 2:n
            p0, p1 = col[i-1], col[i]
            r[i-1] = (ismissing(p0) || ismissing(p1)) ? missing : (p1 / p0 - 1.0)
        end
        ret[!, cname] = r
    end
    return ret
end

retdf = arithmetic_returns(prices_df)
dates_aligned = dates_all[2:end]

# output
out = DataFrame()
out[!, "Date"] = string.(dates_aligned)
for cname in names(retdf)
    out[!, string(cname)] = retdf[!, cname]
end

CSV.write(outfile, out; writeheader=true)

println("finished, results in : ", outfile)
