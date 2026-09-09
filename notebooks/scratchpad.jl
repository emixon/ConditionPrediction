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

crv_hist = histogram(df.CRV, bins=500, title="Replacement Value")

combined = plot( age_hist, ci_hist, score_hist, crv_hist; layout=(2, 2), size=(2000, 1600))
end

# ╔═╡ Cell order:
# ╠═e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
# ╠═af5b14a9-9e5b-42d4-bc68-abf314fc3928
