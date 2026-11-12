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
    using HypothesisTests
    using StatsPlots.PlotMeasures: mm
    using PairPlots, Plots
    using GLM
    using Random
    using MLJ
    using MLJLinearModels
end

# ╔═╡ 8a745a37-637c-49e7-9b80-a00e35e2134d


# ╔═╡ f92de274-9c7f-4ea2-84a5-d18f97044498
md"""
### Load Data and Overview
This will load the dataset and filters the data of interest, saving to a dataframe called "boilers".

It also defines a helper function called getHistAll that plots 12 basic histograms of the data set.
"""

# ╔═╡ 5ca1e039-ee02-4467-95e0-8f44cb4995db
md"""
### Describe The Boilers Data Frame
Shows the initially available columns and types as well as basic stats.
"""

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

# ╔═╡ 745d02a7-6b60-47d3-973c-0f428013c265
md"""
### Addressing observations with missing data
The :Recency and :PrevScore columns of the data will have "missing" data for cases where the component has never been assessed before. Additionaly, not all components have climate data. Since knowing the component has never been assessed could carry its own predictive value, we cannot simply replace "missing" with the mean values in this case. We must first create a new column of Type Bit {0, 1} that preserves the information of a component having previously been assessed or "never assessed".  In this case we will create a column named :PrevAssessed set to 0 for "missing" values and 1 for values that have been assessed in the past.

Once the new :PrevAssessed column is created, we will replace "missing" values with the mean of their respective columns, thus addressing the missing values without losing predictive information.

For the missing Climate data, we will simply replace "missing" values with the mean for available data prior to PCA.
"""

# ╔═╡ 78c6cc72-5658-495d-9444-cfa4b4bc582a
md"""
### The \$1,000,000 boiler
The CRV column values notably cover a range several orders of magnitude more than the rest of our values. When doing a PCA analysis, this will cause CRV to dominate the first component due simply to its numerical scale. The typical recommended action to take in this case is to utilize a log scale for such a datatype when doing PCA. So we will add a LogCRV column to our boilers dataframe for this reason. As shown by the plot below, this also makes our CRV data more symmetric and lowers the influence of the drastically more expensive outliers.
"""

# ╔═╡ 9371dd2a-8673-4104-b04b-4d91a597427a
describe(boilers, :all)

# ╔═╡ 8e86278f-bbec-4f50-8961-41a45bf2aa94
select!(boilers, Not([:MEC, :System, :UOM]))

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

# ╔═╡ 2a9579fc-6ec0-4741-859a-6c9105dc8f7d
md"""
## Pairwise Correlation Analysis
"""

# ╔═╡ 90916ca7-d937-4a85-b40d-a69c23e38562
md"""
### Pearson Correlation Analysis
"""

# ╔═╡ acde71bb-e5e0-4ff2-b655-ffcb1d9c74fd
md"""
The Pearson correlation coefficient is the measure of the direction and strength of the linear relationship between two variables.
"""

# ╔═╡ b9c194d6-87fe-459e-82f6-e551d4889747
md"""
### Spearman Correlation Analysis
"""

# ╔═╡ 09771c81-3e9c-42ea-be1a-e389f0bcd906
md"""
In addition to the PCA analysis, a pairwise correlation analysis was completed to understand relationships among the variables. To this end, Spearman’s rank correlation coefficient (ρ), which is a nonparametric measure of the strength and direction of the monotonic relationship between two variables, was calculated for each combination of pairs among the following set of variables from the cleaned data: NormalizedAge, EffectiveAge, RSL, Recency, PrevScore, PrevAssessed, PredictedCI, Score, Qty, AssetAge, AssetDesignLife, LogCRV, Month, Year, and Climate. Figure 11 visualizes the results of this analysis as a heatmap of correlation coefficients. Positive correlation coefficients, shown as red squares in the heatmap, indicate a positive monotonic relationship, while negative correlation coefficients, shown as blue squares in the heatmap, represent a negative monotonic relationship. Asterisks are used to denote the statistical significance of the correlation between variable pairs, with * signifying p < 0.05, ** signifying p < 0.01, and *** signifying p < 0.001.

Figure 11 shows the strongest pairwise correlations between PredictedCI and NormalizedAge, EffectiveAge, and RSL, as well as between RSL and NormalizedAge. The calculated coefficients for these relationships were -0.88, -0.81, 0.91, and -0.90, respectively, all with statistical significance at p < 0.001. However, these relationships are not expected and are largely explained by the characteristics of the dataset. For example, NormalizedAge is an input to the equation used to calculate PredictedCI, and EffectiveAge is used to calculate RSL. 

In contrast, the relative strength of pairwise correlations with the Score variable, which represents an Expert assessor’s assigned condition rating of a given boiler, are lower. For example, the correlation coefficients with the largest magnitude are those with RSL, NormalizedAge, and PredictedCI, but their values are only 0.54, -0.53, and 0.50, respectively, each with statistical significance at p < 0.001. These results indicate that Score cannot be predicted by a single correlation with another variable, and that a multivariate approach is needed to understand the complex relationships influencing Score.

"""

# ╔═╡ eede658f-b6d3-47be-a5e3-c1a6873897c9


# ╔═╡ 49bec9d7-c299-473b-ac89-bb23d36355ac


# ╔═╡ a8df4899-9178-4b98-8183-056810119467


# ╔═╡ 50bbe087-de89-429f-8fae-77919773861e


# ╔═╡ 5d193b87-fc5c-479e-9793-652a27034297


# ╔═╡ cbefaad9-0df1-4b1a-a3fb-adb5bc2e0f32
md"""
# Predictive Models
"""

# ╔═╡ f7981403-8dbe-40fd-a8f0-b766be8e6c0d
md"""
## Linear Regression
"""

# ╔═╡ 544b0f7b-a9bb-4577-8c01-0e184b3570bc
md"""
### Ordinary Least-Squares Regression
"""

# ╔═╡ 8d1b932b-bfdc-4534-9d15-0f7a9c7b88fd
md"""
Tutorial source: https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-3/
"""

# ╔═╡ 70dbe789-5de5-40af-a68e-bad94f1ae81f
begin   
    ind_vars = [
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
    X = mapcols(col -> Float64.(col), select(boilers, ind_vars))
    y = Float64.(boilers.Score)
    
end

# ╔═╡ 2648a32b-0064-4e3b-8ac7-178c86c4b22d
begin
    # use partition to create train and test datasets
    train_rows, test_rows = partition(
        eachindex(y), # just says to use all rows
        0.80; # 80/20 split between train and test
        stratify = y, # keeps a region proportions similar between train and test
        rng = 1234, # makes this repeatable
    )
end

# ╔═╡ 610dbf1a-cc43-41dd-a1df-8d0aea4b8e02


# ╔═╡ 6f3b0e0c-4487-471e-b696-ae0873b7ebb8


# ╔═╡ 5a8756c2-723e-4326-aa05-0bf6893ca7ea


# ╔═╡ 8a6286e9-7b53-43f6-8e6e-0496ac24eea4
begin
    # normalize data to common range
    scalers = Dict(col => StatsBase.fit(UnitRangeTransform, df_train[!, col]) for col in reg_vars)
    for col in reg_vars
        df_train[!, col] = StatsBase.transform(scalers[col], df_train[!, col])
        df_test[!, col]  = StatsBase.transform(scalers[col], df_test[!, col])
    end
end

# ╔═╡ c0bfae21-3738-4b40-8e2e-6d44498b96b8
begin
    X_train = select(df_train, Not(:Score))
    y_train = df_train.Score
    
    X_test = select(df_test, Not(:Score))
    y_test = df_test.Score
end

# ╔═╡ 528714e7-ebc2-4f9f-8c3e-a62a888793a6
begin
    inputFeatures = length(feature_cols)

    ols = LinearRegressor(
        fit_intercept = true,
    )

    pipeline_ols = Standardizer() |> ols

    mach_ols = machine(pipeline_ols, X_train, y_train)

    fit!(mach_ols)

    predictions = predict(mach_ols, X_test)

end

# ╔═╡ 17cfa049-0fee-49a0-b836-c8891bfd8a3d
ols_model = LinearRegressor()

# ╔═╡ 342297e1-fa1e-47bc-aa66-bbba4323d30c
begin
    mach = machine(ols_model, X_train, y_train)
    fit!(mach)

    fp = fitted_params(mach)
    coefs = fp.coefs
    intercept = fp.intercept
    for (name, val) in coefs
        println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
    end
    println("Intercept: $(round(intercept, sigdigits=3))")
end

# ╔═╡ 80537988-01be-47f5-9557-9a926a426821
begin
	ŷ = MLJ.predict(mach, X_test)
	round(rms(ŷ, y_test), sigdigits=4)
end

# ╔═╡ 85daf586-1d93-4bb6-b263-be3e8bc14592
round(rsq(ŷ, y_test), sigdigits=4)

# ╔═╡ 1a1f1d7b-ff90-4509-9f95-85902ac283f5
begin
	res = ŷ .- y_test
	plot(res, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800,600)))
	hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
end

# ╔═╡ bb91371c-18d9-4891-8fe5-7284d8d889f0
histogram(res, normalize=true, size=(800,600), label="residual")

# ╔═╡ 4e752445-177e-486a-b6f1-d0c392b0c8f4
md"""
### Ridge Regression
"""

# ╔═╡ 6413baf0-2c77-4946-9873-c066e3c007c7
ridge_model = RidgeRegressor(lambda = 0.001)

# ╔═╡ a064fc54-7b97-4135-9b61-2fcee64362b6
begin
	mach_ridge = machine(ridge_model, X_train, y_train)
	fit!(mach_ridge)

    fp_ridge = fitted_params(mach_ridge)
    coefs_ridge = fp_ridge.coefs
    intercept_ridge = fp_ridge.intercept
    for (name, val) in coefs_ridge
        println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
    end
    println("Intercept: $(round(intercept_ridge, sigdigits=3))")
end

# ╔═╡ 34c5e30a-d710-4ebf-9c22-2fcddfd8b37d
begin
	ŷ_ridge = MLJ.predict(mach_ridge, X_test)
	round(rms(ŷ_ridge, y_test), sigdigits=4)
end

# ╔═╡ e265624e-cb5c-4308-b577-9b4f53de1f96
round(rsq(ŷ_ridge, y_test), sigdigits=4)

# ╔═╡ 0386552b-e143-4e4b-9f4e-33d351ea7823
begin
	res_ridge = ŷ_ridge .- y_test
	plot(res_ridge, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800,600)))
	hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
end

# ╔═╡ 4745c960-7a1d-40fc-884e-26dc745e606a
histogram(res_ridge, normalize=true, size=(800,600), label="residual")

# ╔═╡ 4b3f6850-db77-4904-880b-edafae19bc0c
md"""
### Lasso Regression
"""

# ╔═╡ ee0a8fa5-ae54-442f-af71-aa04fbcd4858
lasso_model = LassoRegressor(lambda = 0.001)

# ╔═╡ 78ab1354-a828-4efe-ba59-49565b4c4750
begin
	mach_lasso = machine(lasso_model, X_train, y_train)
	fit!(mach_lasso)

    fp_lasso = fitted_params(mach_lasso)
    coefs_lasso = fp_lasso.coefs
    intercept_lasso = fp_lasso.intercept
    for (name, val) in coefs_lasso
        println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
    end
    println("Intercept: $(round(intercept_lasso, sigdigits=3))")
end

# ╔═╡ 1989773e-2efd-4f41-b0c3-cfa763a4e424
begin
	ŷ_lasso = MLJ.predict(mach_lasso, X_test)
	round(rms(ŷ_lasso, y_test), sigdigits=4)
end

# ╔═╡ a3b6ab7a-bcb9-4325-b3ca-f1c08a3807e5
round(rsq(ŷ_lasso, y_test), sigdigits=4)

# ╔═╡ f9a07554-4e03-475e-9f8e-9bad4e5bd962
begin
	res_lasso = ŷ_lasso .- y_test
	plot(res_lasso, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800,600)))
	hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
end

# ╔═╡ 439da894-e094-498e-b1b2-0dc31f42bef8
histogram(res_ridge, normalize=true, size=(800,600), label="residual")

# ╔═╡ 04520bb6-c623-4e75-acd9-b765c213c144
y_cat = categorical(y_train .> 85)

# ╔═╡ 8c11abd9-8fff-4aa3-b1bb-e0b983fd0c97
begin
	# logistic classifier 
	
	# Load the LogisticClassifier model
	LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
	
	# Example data: X (features table or matrix), y (categorical/binary vector)
	model_logistic = LogisticClassifier()
	mach_logistic = machine(model_logistic, X_train, y_cat)
	fit!(mach_logistic)
	
end

# ╔═╡ 830d1d3a-c8b6-4ff5-9783-6d3902a9645d


# ╔═╡ c2324815-aa18-43a0-b9f3-cd7dbcde5647
md"""
## Decision Tree (XGBoost) Predictive Model
"""

# ╔═╡ 8bc9c3b1-0ffb-451b-bae9-bf3d209355b6


# ╔═╡ 2defcea8-94d2-4182-b313-7b242d185c08
md"""
## Quantile Distribution Model
"""

# ╔═╡ ee264170-dcc6-4d27-af75-144fecda5cdc
begin
	path = joinpath( @__DIR__, "..", "data", "boilers.csv")
    MEC_Name = "Boilers"
    MEC = 1
    df = CSV.read(path, DataFrame)
    boilers = df[df. MEC .== MEC, :]
	select!(boilers, Not([:MEC, :System, :UOM]))
	wrangled_boilers = copy(boilers)
	wrangled_boilers.PrevAssessed = Int.(.!ismissing.(wrangled_boilers.PrevScore))
	
	mean_PrevScore = mean(skipmissing(wrangled_boilers.PrevScore))
	mean_Recency = mean(skipmissing(wrangled_boilers.Recency))
	mean_Climate = mean(skipmissing(wrangled_boilers.Climate))
	
	wrangled_boilers.PrevScore = coalesce.(wrangled_boilers.PrevScore, mean_PrevScore)
	wrangled_boilers.Recency = coalesce.(wrangled_boilers.Recency, mean_Recency)
	wrangled_boilers.Climate = coalesce.(wrangled_boilers.Climate, mean_Climate)
	wrangled_boilers.NormalizedAssetAge = wrangled_boilers.AssetAge ./ wrangled_boilers.AssetDesignLife

	month_angle = 2π .* (wrangled_boilers.Month .- 1) ./ 12

	# Addressses cyclical nature of months (keeping december near january)
    wrangled_boilers.MonthSin = sin.(month_angle)
    wrangled_boilers.MonthCos = cos.(month_angle)

	transform!(
	    groupby(wrangled_boilers, :Fac), 
	    :Score => (x -> 100 * sum(x .>= 88) / length(x)) => :PercentGreen
	)

	# One-hot encode nominal identifier fields
    function add_indicator_columns!(df, source_col; prefix = String(source_col))
        levels = sort(collect(skipmissing(unique(df[!, source_col]))))
        new_columns = Symbol[]

        for level in levels
            new_col = Symbol("$(prefix)_$(level)")
            df[!, new_col] = Int8.(coalesce.(df[!, source_col] .== level, false))
            push!(new_columns, new_col)
        end

        return new_columns
    end

    spec_indicator_cols = add_indicator_columns!(
        wrangled_boilers,
        :Spec;
        prefix = "Spec",
    )

    fac_indicator_cols = add_indicator_columns!(
        wrangled_boilers,
        :Fac;
        prefix = "Fac",
    )

	wrangled_boilers.LogCRV = log1p.(wrangled_boilers.CRV)
	# New Boilers df
	wrangled_boilers
end

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

# ╔═╡ af5b14a9-9e5b-42d4-bc68-abf314fc3928
begin
    path = joinpath( @__DIR__, "..", "data", "boilers.csv")
    MEC_Name = "Boilers"
    MEC = 1
    df = load_data(path)
    boilers = df[df. MEC .== MEC, :]
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
# ╠═ee264170-dcc6-4d27-af75-144fecda5cdc
# ╟─b62c7f0d-f9e8-41b6-9eb6-e77f55345806
# ╟─b23788a2-2890-4417-bb00-9ebd4565b449
# ╟─4daad98c-f8a2-4430-946e-8efd6189a008
# ╟─ef2fb0da-9db3-48bc-a92d-a07e24a6c6a1
# ╟─0bf742cc-4661-4a33-bee0-4905860d2562
# ╟─e39f1b09-9bce-40d5-b5d9-03946bffe6fe
# ╟─2a9579fc-6ec0-4741-859a-6c9105dc8f7d
# ╟─90916ca7-d937-4a85-b40d-a69c23e38562
# ╟─acde71bb-e5e0-4ff2-b655-ffcb1d9c74fd
# ╟─b9c194d6-87fe-459e-82f6-e551d4889747
# ╟─09771c81-3e9c-42ea-be1a-e389f0bcd906
# ╠═eede658f-b6d3-47be-a5e3-c1a6873897c9
# ╠═49bec9d7-c299-473b-ac89-bb23d36355ac
# ╠═a8df4899-9178-4b98-8183-056810119467
# ╠═50bbe087-de89-429f-8fae-77919773861e
# ╠═5d193b87-fc5c-479e-9793-652a27034297
# ╟─cbefaad9-0df1-4b1a-a3fb-adb5bc2e0f32
# ╟─f7981403-8dbe-40fd-a8f0-b766be8e6c0d
# ╟─544b0f7b-a9bb-4577-8c01-0e184b3570bc
# ╟─8d1b932b-bfdc-4534-9d15-0f7a9c7b88fd
# ╠═70dbe789-5de5-40af-a68e-bad94f1ae81f
# ╠═2648a32b-0064-4e3b-8ac7-178c86c4b22d
# ╠═528714e7-ebc2-4f9f-8c3e-a62a888793a6
# ╠═610dbf1a-cc43-41dd-a1df-8d0aea4b8e02
# ╠═6f3b0e0c-4487-471e-b696-ae0873b7ebb8
# ╠═5a8756c2-723e-4326-aa05-0bf6893ca7ea
# ╠═8a6286e9-7b53-43f6-8e6e-0496ac24eea4
# ╠═c0bfae21-3738-4b40-8e2e-6d44498b96b8
# ╠═17cfa049-0fee-49a0-b836-c8891bfd8a3d
# ╠═342297e1-fa1e-47bc-aa66-bbba4323d30c
# ╠═80537988-01be-47f5-9557-9a926a426821
# ╠═85daf586-1d93-4bb6-b263-be3e8bc14592
# ╠═1a1f1d7b-ff90-4509-9f95-85902ac283f5
# ╠═bb91371c-18d9-4891-8fe5-7284d8d889f0
# ╟─4e752445-177e-486a-b6f1-d0c392b0c8f4
# ╠═6413baf0-2c77-4946-9873-c066e3c007c7
# ╠═a064fc54-7b97-4135-9b61-2fcee64362b6
# ╠═34c5e30a-d710-4ebf-9c22-2fcddfd8b37d
# ╠═e265624e-cb5c-4308-b577-9b4f53de1f96
# ╠═0386552b-e143-4e4b-9f4e-33d351ea7823
# ╠═4745c960-7a1d-40fc-884e-26dc745e606a
# ╟─4b3f6850-db77-4904-880b-edafae19bc0c
# ╠═ee0a8fa5-ae54-442f-af71-aa04fbcd4858
# ╠═78ab1354-a828-4efe-ba59-49565b4c4750
# ╠═1989773e-2efd-4f41-b0c3-cfa763a4e424
# ╠═a3b6ab7a-bcb9-4325-b3ca-f1c08a3807e5
# ╠═f9a07554-4e03-475e-9f8e-9bad4e5bd962
# ╠═439da894-e094-498e-b1b2-0dc31f42bef8
# ╠═04520bb6-c623-4e75-acd9-b765c213c144
# ╠═8c11abd9-8fff-4aa3-b1bb-e0b983fd0c97
# ╠═830d1d3a-c8b6-4ff5-9783-6d3902a9645d
# ╟─c2324815-aa18-43a0-b9f3-cd7dbcde5647
# ╠═8bc9c3b1-0ffb-451b-bae9-bf3d209355b6
# ╟─2defcea8-94d2-4182-b313-7b242d185c08
