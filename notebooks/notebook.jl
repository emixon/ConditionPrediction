### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
begin
    import Pkg

    Pkg.activate(Base.current_project())
    Pkg.instantiate()

    using ConditionPrediction
    using DataFrames
    using StatsPlots
    using Statistics
end

# ╔═╡ af5b14a9-9e5b-42d4-bc68-abf314fc3928
begin
path = joinpath( @__DIR__, "..", "data", "observations.csv")

df = load_data(path)

age_hist = histogram(df.NormalizedAge, bins=500, title="Normalized Age")

ci_hist = histogram(df.PredictedCI, bins=500, title="Predicted CI")

score_hist = histogram(df.Score, bins=500, title="Assessed Score")

crv_hist = histogram(df.Recency, bins=500, title="Replacement Value")

combined = plot( age_hist, ci_hist, score_hist, crv_hist; layout=(2, 2), size=(2000, 1600))
end

# ╔═╡ 43ec02bf-1aa0-47b0-ad05-981fceefc222
begin
	sdf = combine(groupby(df, :MEC), nrow => :Count)
	ssdf = sort(sdf, :Count, rev=true)
end

# ╔═╡ 763c68c0-223c-4cf5-81fc-45d78a379935
begin
	function getMEC(df, MEC)
        df[df.MEC .== MEC, :]
    end

    function getHistQuad(d, mec)
        plot(
            histogram(d.NormalizedAge; bins=100, title="$mec Normalized Age"),
            histogram(d.PredictedCI;  bins=100, title="$mec Predicted CI"),
            histogram(d.Score;  bins=100, title="$mec Assessed Score"),
            histogram(d.Recency;      bins=100, title="$mec Recency"),
            layout=(2, 2),
        )
    end

    top_MECs = ssdf.MEC[1:min(12, nrow(ssdf))]
    
    quads = [
        getHistQuad(getMEC(df, mec), mec)
        for mec in top_MECs
    ]
    
    plot(quads...; layout=(4, 3), size=(2400, 1800))
end

# ╔═╡ 2c52a013-0c51-4e48-aaf4-8621f52c7c52
begin
	# Boilers
	plot(getHistQuad(getMEC(df, 171), 171))
end

# ╔═╡ 305f5ea8-86c0-44ca-9eae-8fe28f84dc56
begin
	# Exterior Closures
	plot(getHistQuad(getMEC(df, 28), 28))
end

# ╔═╡ Cell order:
# ╠═e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
# ╠═af5b14a9-9e5b-42d4-bc68-abf314fc3928
# ╠═43ec02bf-1aa0-47b0-ad05-981fceefc222
# ╠═763c68c0-223c-4cf5-81fc-45d78a379935
# ╠═2c52a013-0c51-4e48-aaf4-8621f52c7c52
# ╠═305f5ea8-86c0-44ca-9eae-8fe28f84dc56
