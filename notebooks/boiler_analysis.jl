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
    using LinearAlgebra
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

# ╔═╡ 942ad3ee-eab0-465e-b6d8-f36112bc4bec
md"""
### Removing unecessary properties
With the dataset filtered on MEC 171 (Boilers), we will only have one value in all observations for the :MEC, :System, and :UOM properties.  Those columns effectively carry no information and can be ommitted from PCA.
"""

# ╔═╡ 8e86278f-bbec-4f50-8961-41a45bf2aa94
select!(boilers, Not([:MEC, :System, :UOM]))

# ╔═╡ 745d02a7-6b60-47d3-973c-0f428013c265
md"""
### Addressing observations with missing data
The :Recency and :PrevScore columns of the data will have "missing" data for cases where the component has never been assessed before. Since knowing the component has never been assessed could carry its own predictive value, we cannot simply replace "missing" with the mean values in this case. We must first create a new column of Type Bit {0, 1} that preserves the information of a component having previously been assessed or "never assessed".  In this case we will create a column named :PrevAssessed set to 0 for "missing" values and 1 for values that have been assessed in the past.

Once the new :PrevAssessed column is created, we will replace "missing" values with the mean of their respective columns, thus addressing the missing values without losing predictive information.
"""

# ╔═╡ b078a14b-049e-4398-9c79-85ef7fde2843
begin
	boilers.PrevAssessed = Int.(ismissing.(boilers.PrevScore))
	
	mean_PrevScore = mean(skipmissing(boilers.PrevScore))
	mean_Recency = mean(skipmissing(boilers.Recency))
	
	boilers.PrevScore = coalesce.(boilers.PrevScore, mean_PrevScore)
	boilers.Recency = coalesce.(boilers.Recency, mean_Recency)

	# New Boilers df
	boilers
end

# ╔═╡ 78c6cc72-5658-495d-9444-cfa4b4bc582a
md"""
### The \$1,000,000 boiler
"""

# ╔═╡ 64dbadfb-21b1-4ecd-b157-efd6f3a96983
begin
	crv = boilers[!, :CRV]
	scrv = sort(crv, rev=true)
	plot(scrv, xlabel="Index", ylabel="Cost", st=:scatter)
end

# ╔═╡ 82c5ac02-1bfe-459a-ad49-b5eb84aca954
begin
	# That special guy
	subset(boilers, :CRV => ByRow(==(902540)))
end

# ╔═╡ b62c7f0d-f9e8-41b6-9eb6-e77f55345806
md"""
## Principal Component Analysis
In our goal to predict assessment scores for a given component, it is useful to perform a principal component analysis to identify which properties of the observation provide the majority of the variance. This allows us to de-noise our prediction models by omitting properties that have very little contribution to the information.
"""

# ╔═╡ b23788a2-2890-4417-bb00-9ebd4565b449
md"""
### Removing Id Column
The data wrangling process took care of most of the data reshaping, for the PCA, we will still need to remove the :Id column and the transform the matrix.
"""

# ╔═╡ 0b5304f0-3304-4f32-9070-3348c05797d3
begin
	modes = 15
	pca_boilers = select(boilers, Not([:Id]))
	X = Matrix(pca_boilers)' # Transform df into Matrix
	X̄ = mean(X, dims=2) # Get Mean
	B = X .- X̄ # Get Normalized Matrix
	F = svd(B) # SVD
end

# ╔═╡ 44044a33-0f6f-4159-ac89-2d8387ba781a
begin
	# Get the feature names
    column_names = names(pca_boilers) 
    
    # 1. Create a DataFrame from the loadings matrix (F.U)
    # This automatically names columns "x1", "x2", etc.
    pc_df = DataFrame(F.U, :auto)
    
    # 2. Rename the columns to "PC1", "PC2", etc., based on total PCs available
    num_pcs = size(F.U, 2)
    rename!(pc_df, [Symbol("PC$i") for i in 1:num_pcs])
    
    # 3. Insert the Feature names as the first column
    insertcols!(pc_df, 1, :Feature => column_names)
end

# ╔═╡ 889dc13a-0250-4051-8b15-157085c2936d
total_variance = F.S .^ 2 / (size(B, 2) - 1)

# ╔═╡ afac8034-878f-4fa4-9346-549ae4f6417e
explained_variance_ratio = total_variance[1:modes] / sum(total_variance)

# ╔═╡ 547e2500-c5e8-4e5b-8b37-fe469b7e85fe
begin
    for i in 1:modes
        println("PC $i: $(round(explained_variance_ratio[i] * 100, digits=2))% variance explained")
    end
end

# ╔═╡ 8a5c0007-9b6a-4ad4-8bd2-ef6fa34bdba6
begin
	# (Since it was excluded from Id, its index matches its position in names(pca_boilers))
    score_idx = findfirst(==("Score"), names(pca_boilers))
    
    # 2. Extract the loadings for the "Score" variable across your modes
    # In your row-variable setup, F.Vt columns correspond to original variables
    score_loadings = F.Vt[1:modes, score_idx]
    
    # 3. Calculate the squared loadings (Contribution to the variable's total variance)
    # This shows how much of "Score" is captured by each individual PC
    score_contributions = score_loadings .^ 2
    
    # Print the exact order of variables in your matrix X
    for (idx, name) in enumerate(names(pca_boilers))
        println("  Row $idx in Matrix X = Column '$name' from DataFrame")
    end
    println("")
    # Output the results
    println("Contribution of each PC to explaining 'Score':")
    for i in 1:modes
        pct = round(score_contributions[i] * 100, digits=2)
        println("  PC $i explains $pct% of the variance in Score")
    end
    println("\nTotal variance of 'Score' captured by these $modes components: ", 
            round(sum(score_contributions) * 100, digits=2), "%")

    #V2
    # 1. Find the row index of the "Score" column in your matrix B
    score_idx = findfirst(==("Score"), names(pca_boilers))
    
    # 2. Extract the loadings for the "Score" variable across your modes from F.U
    # F.U[score_idx, 1:modes] gives the weights of "Score" for each principal component
    score_loadings = F.U[score_idx, 1:modes]
    
    # 3. Calculate the squared loadings
    # Because U is orthogonal, the square of the loading represents the proportion of the variable's variance aligned with that component.
    score_contributions = score_loadings .^ 2
    
    # Output the results
    println("Contribution of each PC to explaining 'Score':")
    for i in 1:modes
        pct = round(score_contributions[i] * 100, digits=2)
        println("  PC $i accounts for $pct% of the loading profile for Score")
    end
end

# ╔═╡ 5d26ccac-1d20-44f7-8696-440e91eeabf1
md"""
## Interesting Plot 1
"""

# ╔═╡ 5977499b-030f-49b6-a329-a6f746ff1b2b


# ╔═╡ 345182d1-e582-4498-ad3a-daed465891ec
md"""
## Interesting Plot 2
"""

# ╔═╡ 58e54b9a-1e54-4354-b8f7-005b58e7ff67


# ╔═╡ 409978e5-8f83-49f9-b46b-2f39fac7570f
md"""
## Interesting Plot 3
"""

# ╔═╡ 2d22cc12-a322-40de-991b-909889ee6125


# ╔═╡ 2a9579fc-6ec0-4741-859a-6c9105dc8f7d
md"""
## Regression Analysis
Looking for correlations.
"""

# ╔═╡ 17d5f78e-7188-4496-b00e-36be135fe37f


# ╔═╡ cbefaad9-0df1-4b1a-a3fb-adb5bc2e0f32
md"""
# Predictive Models
"""

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
# ╠═af5b14a9-9e5b-42d4-bc68-abf314fc3928
# ╟─5ca1e039-ee02-4467-95e0-8f44cb4995db
# ╟─9371dd2a-8673-4104-b04b-4d91a597427a
# ╟─43ee48fd-370d-412e-968f-3df0c2b7c422
# ╟─3426ad4c-a340-46c3-994a-0e30dccef086
# ╟─942ad3ee-eab0-465e-b6d8-f36112bc4bec
# ╠═8e86278f-bbec-4f50-8961-41a45bf2aa94
# ╟─745d02a7-6b60-47d3-973c-0f428013c265
# ╟─b078a14b-049e-4398-9c79-85ef7fde2843
# ╟─78c6cc72-5658-495d-9444-cfa4b4bc582a
# ╠═64dbadfb-21b1-4ecd-b157-efd6f3a96983
# ╠═82c5ac02-1bfe-459a-ad49-b5eb84aca954
# ╟─b62c7f0d-f9e8-41b6-9eb6-e77f55345806
# ╟─b23788a2-2890-4417-bb00-9ebd4565b449
# ╠═0b5304f0-3304-4f32-9070-3348c05797d3
# ╠═44044a33-0f6f-4159-ac89-2d8387ba781a
# ╠═889dc13a-0250-4051-8b15-157085c2936d
# ╠═afac8034-878f-4fa4-9346-549ae4f6417e
# ╠═547e2500-c5e8-4e5b-8b37-fe469b7e85fe
# ╠═8a5c0007-9b6a-4ad4-8bd2-ef6fa34bdba6
# ╟─5d26ccac-1d20-44f7-8696-440e91eeabf1
# ╠═5977499b-030f-49b6-a329-a6f746ff1b2b
# ╟─345182d1-e582-4498-ad3a-daed465891ec
# ╠═58e54b9a-1e54-4354-b8f7-005b58e7ff67
# ╟─409978e5-8f83-49f9-b46b-2f39fac7570f
# ╠═2d22cc12-a322-40de-991b-909889ee6125
# ╟─2a9579fc-6ec0-4741-859a-6c9105dc8f7d
# ╠═17d5f78e-7188-4496-b00e-36be135fe37f
# ╟─cbefaad9-0df1-4b1a-a3fb-adb5bc2e0f32
# ╟─e553e6b3-7b8e-4029-b823-eb98362c49a4
# ╠═c4f0042e-dac8-4caf-b983-8bc9678b4d1b
# ╟─c2324815-aa18-43a0-b9f3-cd7dbcde5647
# ╠═8bc9c3b1-0ffb-451b-bae9-bf3d209355b6
# ╟─2defcea8-94d2-4182-b313-7b242d185c08
