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
### Load Data and Overview
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
### Describe The Boilers Data Frame
Shows the initially available columns and types as well as basic stats.
"""

# ╔═╡ 9371dd2a-8673-4104-b04b-4d91a597427a
describe(boilers, :all)

# ╔═╡ 43ee48fd-370d-412e-968f-3df0c2b7c422
md"""
# Begin Exploratory Analysis
"""

# ╔═╡ 3426ad4c-a340-46c3-994a-0e30dccef086
md"""
## Data Wrangling
"""

# ╔═╡ 64dbadfb-21b1-4ecd-b157-efd6f3a96983


# ╔═╡ b62c7f0d-f9e8-41b6-9eb6-e77f55345806
md"""
## Principal Component Analysis
In our goal to predict assessment scores for a given component, it is useful to perform a principal component analysis to identify which properties of the observation provide the majority of the variance. This allows us to de-noise our prediction models by omitting properties that have very little contribution to the information.
"""

# ╔═╡ 5d26ccac-1d20-44f7-8696-440e91eeabf1


# ╔═╡ 2a9579fc-6ec0-4741-859a-6c9105dc8f7d
md"""
## Regression Analysis
Looking for correlations.
"""

# ╔═╡ 17d5f78e-7188-4496-b00e-36be135fe37f


# ╔═╡ e553e6b3-7b8e-4029-b823-eb98362c49a4
md"""
## Artificial Neural Network (ANN) Predictive Model
"""

# ╔═╡ c4f0042e-dac8-4caf-b983-8bc9678b4d1b


# ╔═╡ c2324815-aa18-43a0-b9f3-cd7dbcde5647
md"""
## Decision Tree (XGBoost) Predictive Model
"""

# ╔═╡ 8bc9c3b1-0ffb-451b-bae9-bf3d209355b6


# ╔═╡ 2defcea8-94d2-4182-b313-7b242d185c08
md"""
## Quantile Distribution Model
"""

# ╔═╡ Cell order:
# ╠═e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
# ╟─f92de274-9c7f-4ea2-84a5-d18f97044498
# ╟─af5b14a9-9e5b-42d4-bc68-abf314fc3928
# ╟─5ca1e039-ee02-4467-95e0-8f44cb4995db
# ╟─9371dd2a-8673-4104-b04b-4d91a597427a
# ╟─43ee48fd-370d-412e-968f-3df0c2b7c422
# ╟─3426ad4c-a340-46c3-994a-0e30dccef086
# ╠═64dbadfb-21b1-4ecd-b157-efd6f3a96983
# ╟─b62c7f0d-f9e8-41b6-9eb6-e77f55345806
# ╠═5d26ccac-1d20-44f7-8696-440e91eeabf1
# ╟─2a9579fc-6ec0-4741-859a-6c9105dc8f7d
# ╠═17d5f78e-7188-4496-b00e-36be135fe37f
# ╟─e553e6b3-7b8e-4029-b823-eb98362c49a4
# ╠═c4f0042e-dac8-4caf-b983-8bc9678b4d1b
# ╟─c2324815-aa18-43a0-b9f3-cd7dbcde5647
# ╠═8bc9c3b1-0ffb-451b-bae9-bf3d209355b6
# ╟─2defcea8-94d2-4182-b313-7b242d185c08
