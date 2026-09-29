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

# ╔═╡ f92de274-9c7f-4ea2-84a5-d18f97044498
md"""
This will load the dataset and filters the data of interest, saving to a dataframe called "boilers".

It also defines a helper function called getHistAll that plots 12 basic histograms of the data set.
"""

# ╔═╡ af5b14a9-9e5b-42d4-bc68-abf314fc3928
begin
    path = joinpath( @__DIR__, "..", "data", "observations.csv")

    # MEC analysis for 171 "Boilers"
    MEC = 171
    MEC_Name = "Boilers"

    df = load_data(path)
    boilers = df[df.MEC .== MEC, :]
    function getHistAll(d)
        plot(
            histogram(d.NormalizedAge; bins=100, title="$MEC_Name Normalized Age"),
            histogram(d.PredictedCI;  bins=100, title="$MEC_Name Predicted CI"),
            histogram(d.Score;  bins=100, title="$MEC_Name Assessed Score"),
            histogram(d.Recency;      bins=100, title="$MEC_Name Recency"),
            histogram(d.Spec, title="$MEC_Name Specification"),
            histogram(d.DesignLife, title="$MEC_Name Design Life"),
            histogram(d.Qty, title="$MEC_Name Quantity"),
            histogram(d.EffectiveAge; bins=100, title="$MEC_Name Effective Age"),
            histogram(d.RSL; bins=100, title="$MEC_Name Remaining Service Life"),
            histogram(d.CRV; bins=100, title="$MEC_Name Component Replacement Value"),
            histogram(d.Fac, title="$MEC_Name Facility Code"),
            histogram(d.AssetAge, title="$MEC_Name Asset Age"),
            layout=(4, 3), size=(2400, 1800)
        )
    end

    getHistAll(boilers)
end

# ╔═╡ 5ca1e039-ee02-4467-95e0-8f44cb4995db
md"""
Describe the boilers data frame to show the initially available columns and types
"""

# ╔═╡ 9371dd2a-8673-4104-b04b-4d91a597427a
describe(boilers)

# ╔═╡ Cell order:
# ╟─e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
# ╟─f92de274-9c7f-4ea2-84a5-d18f97044498
# ╟─af5b14a9-9e5b-42d4-bc68-abf314fc3928
# ╟─5ca1e039-ee02-4467-95e0-8f44cb4995db
# ╟─9371dd2a-8673-4104-b04b-4d91a597427a
