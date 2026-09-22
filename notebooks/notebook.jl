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

score_hist = histogram(df.ActualScore, bins=500, title="Assessed Score")

crv_hist = histogram(df.Recency, bins=500, title="Replacement Value")

combined = plot( age_hist, ci_hist, score_hist, crv_hist; layout=(2, 2), size=(2000, 1600))
end

# ╔═╡ 43ec02bf-1aa0-47b0-ad05-981fceefc222
begin
	sdf = combine(groupby(df, :Spec), nrow => :Count)
	ssdf = sort(sdf, :Count, rev=true)
end

# ╔═╡ 763c68c0-223c-4cf5-81fc-45d78a379935
begin
	function getSpec(df, spec)
        df[df.Spec .== spec, :]
    end

    function getHistQuad(d, spec)
        plot(
            histogram(d.NormalizedAge; bins=100, title="$spec Normalized Age"),
            histogram(d.PredictedCI;  bins=100, title="$spec Predicted CI"),
            histogram(d.ActualScore;  bins=100, title="$spec Assessed Score"),
            histogram(d.Recency;      bins=100, title="$spec Recency"),
            layout=(2, 2),
        )
    end

    top_specs = ssdf.Spec[1:min(12, nrow(ssdf))]
    
    quads = [
        getHistQuad(getSpec(df, spec), spec)
        for spec in top_specs
    ]
    
    plot(quads...; layout=(4, 3), size=(2400, 1800))
end

# ╔═╡ Cell order:
# ╠═e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
# ╠═af5b14a9-9e5b-42d4-bc68-abf314fc3928
# ╠═43ec02bf-1aa0-47b0-ad05-981fceefc222
# ╠═763c68c0-223c-4cf5-81fc-45d78a379935
