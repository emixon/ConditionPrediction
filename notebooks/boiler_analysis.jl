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
    using StatsBase
end

# ╔═╡ 8a745a37-637c-49e7-9b80-a00e35e2134d


# ╔═╡ f92de274-9c7f-4ea2-84a5-d18f97044498
md"""
### Load Data and Overview
This will load the dataset and filters the data of interest, saving to a dataframe called "boilers".

It also defines a helper function called getHistAll that plots 12 basic histograms of the data set.
"""

# ╔═╡ af5b14a9-9e5b-42d4-bc68-abf314fc3928
begin
    path = joinpath( @__DIR__, "..", "data", "boilers.csv")
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
            histogram(d.Month, bins=12, title="$MEC_Name Month of Assessment"),
            histogram(d.Year, title="$MEC_Name Year of Assessment"),
            histogram(d.Climate, title="$MEC_Name Climate"),
            histogram(d.AssetDesignLife, title="$MEC_Name Asset Design Life"),
            layout=(4, 4), size=(2400, 1800)
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
The :Recency and :PrevScore columns of the data will have "missing" data for cases where the component has never been assessed before. Additionaly, not all components have climate data. Since knowing the component has never been assessed could carry its own predictive value, we cannot simply replace "missing" with the mean values in this case. We must first create a new column of Type Bit {0, 1} that preserves the information of a component having previously been assessed or "never assessed".  In this case we will create a column named :PrevAssessed set to 0 for "missing" values and 1 for values that have been assessed in the past.

Once the new :PrevAssessed column is created, we will replace "missing" values with the mean of their respective columns, thus addressing the missing values without losing predictive information.

For the missing Climate data, we will simply replace "missing" values with the mean for available data prior to PCA.
"""

# ╔═╡ b078a14b-049e-4398-9c79-85ef7fde2843
begin
	boilers_raw_data = boilers
	boilers.PrevAssessed = Int.(.!ismissing.(boilers.PrevScore))
	
	mean_PrevScore = mean(skipmissing(boilers.PrevScore))
	mean_Recency = mean(skipmissing(boilers.Recency))
	mean_Climate = mean(skipmissing(boilers.Climate))
	
	boilers.PrevScore = coalesce.(boilers.PrevScore, mean_PrevScore)
	boilers.Recency = coalesce.(boilers.Recency, mean_Recency)
	boilers.Climate = coalesce.(boilers.Climate, mean_Climate)

	# New Boilers df
	boilers
end

# ╔═╡ 78c6cc72-5658-495d-9444-cfa4b4bc582a
md"""
### The \$1,000,000 boiler
The CRV column values notably cover a range several orders of magnitude more than the rest of our values. When doing a PCA analysis, this will cause CRV to dominate the first component due simply to its numerical scale. The typical recommended action to take in this case is to utilize a log scale for such a datatype when doing PCA. So we will add a LogCRV column to our boilers dataframe for this reason. As shown by the plot below, this also makes our CRV data more symmetric and lowers the influence of the drastically more expensive outliers.
"""

# ╔═╡ 64dbadfb-21b1-4ecd-b157-efd6f3a96983
begin
	crv = boilers[!, :CRV]
	scrv = sort(crv, rev=true)
	boilers.LogCRV = log1p.(boilers.CRV)
	log_crv = boilers[!, :LogCRV]
	log_scrv = sort(log_crv, rev=true)
	crv_plot = plot(scrv, xlabel="Index", ylabel="Cost", title="CRV", st=:scatter)
	log_crv_plot = plot(log_scrv, xlabel="Index", ylabel="Cost", title="Log CRV", st=:scatter)
	plot(crv_plot, log_crv_plot, layout=(1,2), size=(1200, 800))
end

# ╔═╡ b62c7f0d-f9e8-41b6-9eb6-e77f55345806
md"""
## Principal Component Analysis
In our goal to predict assessment scores for a given component, it is useful to perform a principal component analysis to identify which properties of the observation provide the majority of the variance. This allows us to de-noise our prediction models by omitting properties that have very little contribution to the information.
"""

# ╔═╡ b23788a2-2890-4417-bb00-9ebd4565b449
md"""
### Removing Identifier Columns and Standardization
The data wrangling process took care of most of the data cleanup and reshaping for the PCA, but we still have identifier columns that are not really quantitative measurements. The :Id, :Spec, and :Fac should be omitted from the PCA, though :Spec and :Fac can still be used later when developing predictive models. Since our goal ultimately is to predict the :Score of the component, we will also want to omit :Score from the initial PCA to avoid "data leakage".

Additionally, while we addressed CRV's order of magnitude, there are still a wide numerical range of units.  To ensure values with larger numerical scales (like :AssetAge) don't dominate the analysis, in addition to centering the data by subtracting the mean of each attribute, we also standardize by dividing each centered attribute by its standard deviation.
"""

# ╔═╡ 03d7f9cc-89ac-4521-8e9e-0ddcf6a8ba9c
begin
    # The Columns we want to use for the PCA
    pca_vars = [
        :NormalizedAge,
        :EffectiveAge,
        :RSL,
        :Recency,
        :PrevScore,
        :PrevAssessed,
        :PredictedCI,
        :Qty,
        :AssetAge,
        :AssetDesignLife,
        :LogCRV,
        :Year,
        :Month,
        :Climate
    ]

    X = Matrix(select(boilers, pca_vars))' # Select columns, convert to matrix, and transform
    
    X̄ = mean(X, dims=2) # Get the Mean
    σ = std(X, dims=2) # Get the Standard Deviation

    Z = (X .- X̄) ./ σ # Normalize AND Standardize

    F = svd(Z) # Get the SVD

    explained_variance = F.S .^ 2 ./ (size(Z, 2) - 1)
    explained_ratio = explained_variance ./ sum(explained_variance)
    cumulative_ratio = cumsum(explained_ratio)

    DataFrame(
        PC = 1:length(F.S),
        ExplainedVariance = explained_variance,
        ExplainedRatio = explained_ratio,
        CumulativeRatio = cumulative_ratio,
    )
end

# ╔═╡ a4346d81-ccfe-4376-9cc6-27973e80c095
begin
    modes = 11
    plot(
        1:modes,
        explained_ratio[1:modes] .* 100;
        marker = :circle,
        xlabel = "Principal Component",
        ylabel = "Explained Variance (%)",
        title = "Boiler PCA Plot",
        legend = false,
    )
end

# ╔═╡ e417d3b5-d72d-43d5-8662-ad3eefce447d
begin
    loadings = DataFrame(
        Variable = string.(pca_vars),
        PC1 = F.U[:, 1],
        PC2 = F.U[:, 2],
        PC3 = F.U[:, 3],
        PC4 = F.U[:, 4],
        PC5 = F.U[:, 5],
        PC6 = F.U[:, 6],
        PC7 = F.U[:, 7],
        PC8 = F.U[:, 8],
        PC9 = F.U[:, 9],
        PC10 = F.U[:, 10],
        PC11 = F.U[:, 11],
        PC12 = F.U[:, 12],
        PC13 = F.U[:, 13],
        PC14 = F.U[:, 14],
    )

    sort!(loadings, :PC1, by=abs, rev=true)

    loadings
end

# ╔═╡ 4daad98c-f8a2-4430-946e-8efd6189a008
md"""
### Repeat the analysis less redundant age measures
The initial analysis showed a high ratio of explained variance for PC1 (~34%), however, the variables dominating the PC were all lifecycle representations that are mathematically similar and therefore potentially redundant.

NormalizedAge: -.496

RSL: .491

PredictedCI: .487

EffectiveAge -.479

Below we repeat the analysis while keeping just one of the lifecycle parameters (in this case RSL). RSL is arbitrarily chosen in this case, mostly due to it being arguably the easiest for most people to understand from an interpretability standpoint. We could, however, just as easily repeat with any of the others.

We start by defining some helper functions to help with generalization of the process in case we want to repeat it from different perspectives. (This also just helps avoid Julia "redifined variable" issues)

The final result shows a much smoother decline in explained variance for our remaining PCs, with PC1-7 explaining ~93% of the variance.

Initial take-aways from this suggest that predictive models would potentially benefit from choosing only one of :NormalizedAge, :RSL, :PredictedCI, or :EffectiveAge to avoid duplicate encoded information. While not conclusively an expected improvement, it does suggest it to be a reasonable test case.
"""

# ╔═╡ ef2fb0da-9db3-48bc-a92d-a07e24a6c6a1
# In case we want to do this with more variations, we'll generalize with a function.
function CalcSVDAndExplainedVariance(pca_variables, df)
	X = Matrix(select(df, pca_variables))' # Select columns, convert to matrix, and transform
    
    X̄ = mean(X, dims=2) # Get the Mean
    σ = std(X, dims=2) # Get the Standard Deviation

    Z = (X .- X̄) ./ σ # Normalize AND Standardize

    F = svd(Z) # Get the SVD

    explained_variance = F.S .^ 2 ./ (size(Z, 2) - 1)
    explained_ratio = explained_variance ./ sum(explained_variance)
    cumulative_ratio = cumsum(explained_ratio)

    results = DataFrame(
        PC = 1:length(F.S),
        ExplainedVariance = explained_variance,
        ExplainedRatio = explained_ratio,
        CumulativeRatio = cumulative_ratio,
    )
    return F, results, explained_variance, explained_ratio
end

# ╔═╡ 0bf742cc-4661-4a33-bee0-4905860d2562
function GetPCAPlot(modes, explained_ratio)

	plot(
        1:modes,
        explained_ratio[1:modes] .* 100;
        marker = :circle,
        xlabel = "Principal Component",
        ylabel = "Explained Variance (%)",
        title = "Boiler PCA Plot",
        legend = false,
    )
end

# ╔═╡ e39f1b09-9bce-40d5-b5d9-03946bffe6fe
function GetLoadings(F, pca_vars; sort_by = :PC1)
    loadings = DataFrame(
        Variable = string.(pca_vars),
    )

    for k in axes(F.U, 2)
        loadings[!, Symbol("PC$k")] = F.U[:, k]
    end
    sort!(loadings, sort_by, by = abs, rev = true)
    return loadings
end

# ╔═╡ 653ae9d0-bca6-4778-a8d3-9d07dbf143c2
function GetLoadingsHeatMap(F, pca_variables; squared=false)
	k = size(F.U, 2)
    L = F.U[:, 1:k]

    limit = maximum(abs, L)

    heatmap(
        1:k,
        string.(pca_variables),
        (squared ? L .^ 2 : L);
        xlabel = "Principal Component",
        ylabel = "Variable",
        title = "PCA Loading Heatmap",
        colorbar_title = "Loading",
        clims = (squared ? (0, limit) : (-limit, limit)),
        c = (squared ? :viridis : :RdBu),
        yflip = true,
    )
end

# ╔═╡ af1af019-a784-4144-a632-e66fca8b1c3b
let
    # Repeat the PCA with only one condition/lifecycle representative (arbitrarily choosing RSL, we could also try others)
    pca_vars_single_lf = [
        :RSL,
        :Recency,
        :PrevScore,
        :PrevAssessed,
        :Qty,
        :AssetAge,
        :AssetDesignLife,
        :LogCRV,
        :Month,
        :Year,
        :Climate
    ]
    F, results, explained_variance, explained_ratio = CalcSVDAndExplainedVariance(pca_vars_single_lf, boilers)

    results, GetPCAPlot(8, explained_ratio), GetLoadings(F, pca_vars_single_lf), GetLoadingsHeatMap(F, pca_vars_single_lf, squared=true)
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
"""

# ╔═╡ 90916ca7-d937-4a85-b40d-a69c23e38562
md"""
### Pearson Correlation Analysis
"""

# ╔═╡ d832c11a-a8b3-4f92-9a30-caccc1d6b28e
begin

	# The Columns we want to use for the correlation analysis
    cor_vars = [
        :NormalizedAge,
        :EffectiveAge,
        :RSL,
        :Recency,
        :PrevScore,
        :PrevAssessed,
        :PredictedCI,
        :Score,
        :Qty,
        :AssetAge,
        :AssetDesignLife,
        :LogCRV,
        :Month,
        :Year,
        :Climate
    ]
    
	# create matrix for Pearson correlation
	boilers_cor = Matrix(select(boilers_raw_data, cor_vars))
	
    # calculate Spearman correlation coefficients
    pearson_matrix = cor(boilers_cor)
end

# ╔═╡ c971b211-e043-47b3-88b7-ad73b6872796
begin
	# CREATE HEATMAP

	# create data labels
	
	pearson_labels = [
		"NormalizedAge",
        "EffectiveAge",
        "RSL",
        "Recency",
        "PrevScore",
        "PrevAssessed",
        "PredictedCI",
        "Score",
        "Qty",
        "AssetAge",
        "AssetDesignLife",
        "LogCRV",
        "Month",
        "Year",
        "Climate"
	]

	# create mask to hide redundant values
	pearson_matrix_masked = copy(pearson_matrix)
	pearson_matrix_masked[triu!(trues(size(pearson_matrix_masked)), 1)] .= NaN
	
	# plot heatmap
	heatmap(pearson_labels, pearson_labels, pearson_matrix_masked, xrotation = 45, clim = (-1,1), c = :bwr, title = "Pearson Corrrelation Heatmap", yflip = true, aspect_ratio = :equal, right_margin = 20Plots.mm, left_margin = 20Plots.mm)
end

# ╔═╡ b9c194d6-87fe-459e-82f6-e551d4889747
md"""
### Spearman Correlation Analysis
"""

# ╔═╡ 5dc1a1ca-22ae-4cca-9d52-a83b71e2535c
begin
	# create matrix for correlation
	boilers_spearman = Matrix(select(boilers_raw_data, cor_vars))
	# calculate Spearman correlation coefficients
    spearman_matrix = corspearman(boilers_cor)
end

# ╔═╡ 17d5f78e-7188-4496-b00e-36be135fe37f
begin
	# CREATE SPEARMAN HEATMAP

	# create data labels
	
	spearman_labels = [
		"NormalizedAge",
        "EffectiveAge",
        "RSL",
        "Recency",
        "PrevScore",
        "PrevAssessed",
        "PredictedCI",
        "Score",
        "Qty",
        "AssetAge",
        "AssetDesignLife",
        "LogCRV",
        "Month",
        "Year",
        "Climate"
	]

	# create mask to hide redundant values
	spearman_matrix_masked = copy(spearman_matrix)
	spearman_matrix_masked[triu!(trues(size(spearman_matrix_masked)), 1)] .= NaN
	
	# plot heatmap
	heatmap(spearman_labels, spearman_labels, spearman_matrix_masked, xrotation = 45, clim = (-1,1), c = :bwr, title = "Spearman Correlation Heatmap", yflip = true, aspect_ratio = :equal, right_margin = 20Plots.mm, left_margin = 20Plots.mm)
end

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
# ╠═8a745a37-637c-49e7-9b80-a00e35e2134d
# ╟─f92de274-9c7f-4ea2-84a5-d18f97044498
# ╠═af5b14a9-9e5b-42d4-bc68-abf314fc3928
# ╟─5ca1e039-ee02-4467-95e0-8f44cb4995db
# ╠═9371dd2a-8673-4104-b04b-4d91a597427a
# ╟─43ee48fd-370d-412e-968f-3df0c2b7c422
# ╟─3426ad4c-a340-46c3-994a-0e30dccef086
# ╟─942ad3ee-eab0-465e-b6d8-f36112bc4bec
# ╠═8e86278f-bbec-4f50-8961-41a45bf2aa94
# ╟─745d02a7-6b60-47d3-973c-0f428013c265
# ╠═b078a14b-049e-4398-9c79-85ef7fde2843
# ╟─78c6cc72-5658-495d-9444-cfa4b4bc582a
# ╠═64dbadfb-21b1-4ecd-b157-efd6f3a96983
# ╟─b62c7f0d-f9e8-41b6-9eb6-e77f55345806
# ╟─b23788a2-2890-4417-bb00-9ebd4565b449
# ╠═03d7f9cc-89ac-4521-8e9e-0ddcf6a8ba9c
# ╟─a4346d81-ccfe-4376-9cc6-27973e80c095
# ╠═e417d3b5-d72d-43d5-8662-ad3eefce447d
# ╟─4daad98c-f8a2-4430-946e-8efd6189a008
# ╟─ef2fb0da-9db3-48bc-a92d-a07e24a6c6a1
# ╟─0bf742cc-4661-4a33-bee0-4905860d2562
# ╟─e39f1b09-9bce-40d5-b5d9-03946bffe6fe
# ╠═653ae9d0-bca6-4778-a8d3-9d07dbf143c2
# ╠═af1af019-a784-4144-a632-e66fca8b1c3b
# ╟─5d26ccac-1d20-44f7-8696-440e91eeabf1
# ╠═5977499b-030f-49b6-a329-a6f746ff1b2b
# ╟─345182d1-e582-4498-ad3a-daed465891ec
# ╠═58e54b9a-1e54-4354-b8f7-005b58e7ff67
# ╟─409978e5-8f83-49f9-b46b-2f39fac7570f
# ╠═2d22cc12-a322-40de-991b-909889ee6125
# ╟─2a9579fc-6ec0-4741-859a-6c9105dc8f7d
# ╟─90916ca7-d937-4a85-b40d-a69c23e38562
# ╠═d832c11a-a8b3-4f92-9a30-caccc1d6b28e
# ╠═c971b211-e043-47b3-88b7-ad73b6872796
# ╟─b9c194d6-87fe-459e-82f6-e551d4889747
# ╠═5dc1a1ca-22ae-4cca-9d52-a83b71e2535c
# ╠═17d5f78e-7188-4496-b00e-36be135fe37f
# ╟─cbefaad9-0df1-4b1a-a3fb-adb5bc2e0f32
# ╟─e553e6b3-7b8e-4029-b823-eb98362c49a4
# ╠═c4f0042e-dac8-4caf-b983-8bc9678b4d1b
# ╟─c2324815-aa18-43a0-b9f3-cd7dbcde5647
# ╠═8bc9c3b1-0ffb-451b-bae9-bf3d209355b6
# ╟─2defcea8-94d2-4182-b313-7b242d185c08
