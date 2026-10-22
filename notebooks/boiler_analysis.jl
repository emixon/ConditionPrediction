### A Pluto.jl notebook ###
# v1.0.4

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

# ╔═╡ ea18b937-bd81-4517-a04a-ca4987d37319
begin
	Pkg.add(["HypothesisTests"])
	
	using HypothesisTests
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

# ╔═╡ 7606e704-083e-4a5f-8da5-bcdf3bfa3216
md"""
## Two Comparison Groups
"""

# ╔═╡ 12b33241-044c-4631-a9b9-e9c44e24847a
begin
    # Load separate data
    comparison_data = dropmissing(load_data(path), :Score)

    # Separate observations by assessment score.
    green_boilers = comparison_data[comparison_data.Score .>= 88, :]
    notgreen_boilers = comparison_data[comparison_data.Score .< 88, :]

    # Count observations
    group_summary = DataFrame(
        Group = ["Green (Score >= 88)", "Not green (Score < 88)"],
        N = [nrow(green_boilers), nrow(notgreen_boilers)],
    )
    # Calculate the percentage
    group_summary.Percent = round.(
        100 .* group_summary.N ./ sum(group_summary.N);
        digits =2,
    )
    group_summary
end

# ╔═╡ 0263eb8b-4e72-4ebc-988b-e524302c2d6a
md"""
## Two Comparison Groups

Assessment records are divided into green (`Score ≥ 88`) and not-green (`Score < 88`) groups based on the current assessment score. Records without a current score are excluded because their condition status cannot be assigned. The summary table reports the number and percentage of observations in each group, establishing the distribution of condition status before examining lifecycle characteristics, assessment history, asset age, and climate. Since unequal group sizes can make comparisons based on raw counts misleading, the subsequent figures use proportions to compare distributions and condition percentages. Sample-size labels indicate the number of observations supporting each comparison. These counts represent assessment records rather than necessarily distinct boilers.
"""

# ╔═╡ 5d26ccac-1d20-44f7-8696-440e91eeabf1
md"""
## Interesting Plot 1
"""

# ╔═╡ a684491d-5717-498b-b3a7-1c864afb9a31


# ╔═╡ 611d8e7c-fdfe-43e8-a12d-647272d2099e
md"""
Remaining service life (RSL) and previous assessment scores provide two perspectives on the relationship between asset history and current condition. Green assessments have a substantially higher average RSL (18.61) than not-green assessments (8.76), while their average previous scores are 88.56 and 79.39, respectively. The differences between these distributions indicate that both lifecycle characteristics and previous condition may contain useful information for predicting assessment scores. Although the distributions overlap, the general separation between the groups supports further consideration of these variables in predictive modeling.
"""

# ╔═╡ 4ae06b54-432d-4434-9bc8-47c8a91240e7
style = (
    fontfamily = "sans-serif",
    titlefontsize = 13,
    guidefontsize = 11,
    tickfontsize = 10,
    legendfontsize = 9,
    background_color = :white,
    framestyle = :axes,
    grid = false,
    margin = 6Plots.mm,
    left_margin = 12Plots.mm,
    bottom_margin = 14Plots.mm,
)

# ╔═╡ e7fae12f-3e0b-4e9f-a874-4cb0db599494
interesting_plot1 = let
    function compare_histogram(variable, heading)
        green = filter(isfinite, collect(skipmissing(green_boilers[!, variable])))
        notgreen = filter(isfinite, collect(skipmissing(notgreen_boilers[!, variable])))

        if isempty(green) || isempty(notgreen)
            error("Both groups need recorded values for $variable.")
        end

        low, high = extrema(vcat(green, notgreen))
        if low == high
            low -= 0.5
            high += 0.5
        end
        edges = range(low, nextfloat(Float64(high)); length=31)

        histogram([green, notgreen];
            bins=edges, normalize=:probability, alpha=0.5,
            label=["Green (n=$(length(green)))" "Not green (n=$(length(notgreen)))"],
            title=heading, xlabel=heading,
            ylabel="Proportion within group"
        )
    end

    plot(
        compare_histogram(:RSL, "Remaining Service Life"),
        compare_histogram(:PrevScore, "Previous Assessment Score");
        style...,
        layout=(1, 2),
        size=(1200, 800),
        legend=:outertop,
        legend_column=2
    )
end

# ╔═╡ 345182d1-e582-4498-ad3a-daed465891ec
md"""
## Interesting Plot 2
"""

# ╔═╡ b41a40ab-1eee-4f01-a8f3-a32a249cf6c7
md"""
Asset age is another important characteristic when examining differences in component condition. The proportion of not-green assessments generally increases with age, rising from approximately 10.9% for assets younger than 10 to 30.3% for those between 30 and 40, before decreasing to 24.1% for assets aged 40 or older. Although the relationship is not strictly increasing, the results indicate that asset age may help explain variations in assessed condition. The lower proportion among the oldest assets also suggests that age alone is insufficient to describe condition, highlighting the potential importance of other asset characteristics and assessment history.
"""

# ╔═╡ 9c2bd780-dca9-4778-8ae3-58003769aefb
interesting_plot2 = let
    d = dropmissing(comparison_data, :AssetAge)
    d = d[isfinite.(d.Score) .& isfinite.(d.AssetAge), :]

    edges = [0, 10, 20, 30, 40, Inf]
    ages = ["0–<10", "10–<20", "20–<30", "30–<40", "40+"]

    groups = [d.Score[(d.AssetAge .>= edges[i]) .&
                      (d.AssetAge .< edges[i+1])] for i in 1:5]

    green = [isempty(g) ? NaN : 100 * mean(g .>= 88) for g in groups]
    notgreen = 100 .- green
    labels = ["$(ages[i])\n(n=$(length(groups[i])))" for i in 1:5]

    plot(1:5, [green notgreen];
        style...,
        label=["Green" "Not green"],
        legend=:outertop,
        legend_column=2,
        markershape=[:circle :diamond],
        linestyle=[:solid :dash],
        linewidth=3,
        xticks=(1:5, labels),
        xlims=(0.5, 5.5),
        ylims=(-3, 103),
        yticks=0:20:100,
        title="Green vs Not Green Across Asset-Age Groups",
        xlabel="Asset-Age Group",
        ylabel="Assessments within age group (%)",
        size=(1200, 800)
    )
end

# ╔═╡ d9299f4a-f0f2-4bc4-aec6-35f6a9ac652a
md"""
## Interesting Plot 3
"""

# ╔═╡ e23f9964-af24-45e1-bca5-987438d340b1
md"""
Differences in assessed condition are also apparent across climate categories. Climate Codes 4 and 8 have relatively high proportions of not-green assessments, approximately 50.0% and 46.1%, respectively, compared with only 12.4% for Climate Code 7. This variation indicates that climate classification may provide additional information when evaluating component condition. However, nearly half of the observations have missing climate information, and the differences between categories may also reflect variations in asset age, maintenance history, or other characteristics. Further analysis is therefore necessary to determine whether climate contributes independently to condition prediction.
"""

# ╔═╡ 4757dd14-b688-49cc-8e73-3878eceff35e

interesting_plot3 = let
    d = comparison_data[isfinite.(comparison_data.Score), :]
    climate = string.(coalesce.(d.Climate, "Unknown"))
    codes = sort(unique(climate))

    groups = [d.Score[climate .== c] for c in codes]
    green = [isempty(g) ? NaN : 100 * mean(g .>= 88) for g in groups]
    notgreen = 100 .- green
    labels = ["$(codes[i])\n(n=$(length(groups[i])))" for i in eachindex(codes)]

    p = groupedbar(1:length(codes), [green notgreen];
        style...,
        bar_position=:stack,
        bar_width=0.65,
        fillalpha=0.6,
        linewidth=0,
        label=["Green" "Not green"],
        legend=:outertop,
        legend_column=2,
        xticks=(1:length(codes), labels),
        xlims=(0.5, length(codes)+0.5),
        ylims=(0, 110),
        yticks=0:20:100,
        ywiden=false,
        title="Green vs Not Green by Climate",
        xlabel="Climate Code",
        ylabel="Assessments (%)",
        size=(max(1200, 140*length(codes)), 800),
        bottom_margin=16Plots.mm
    )

    for i in eachindex(green)
        if isnan(green[i])
            annotate!(p, i, 50, text("No data", 10))
        else
            for (value, y, group) in (
                (notgreen[i], notgreen[i]/2, "Not green"),
                (green[i], notgreen[i]+green[i]/2, "Green")
            )
                label = value < 8 ? "$group: $(round(value; digits=1))%" :
                                    "$(round(value; digits=1))%"
                annotate!(p, i, value < 8 ? 105 : y, text(label, 10))
            end
        end
    end

    p
end

# ╔═╡ d0b92a37-92ed-4ce4-85cb-beba9eaab65a
md"""
## Interesting Plot 4
"""

# ╔═╡ 50a8cd94-bf14-44b8-b170-319fe9b97b16
md"""
The climate comparison is extended by grouping assessments into winter, spring, summer, and fall to examine potential seasonal patterns in recorded condition. For Climate Code 5, the proportion of not-green assessments increases from approximately 10.8% during summer to 24.0% during fall, indicating differences in condition distributions between assessment periods. Similar comparisons across the remaining climate categories provide a broader perspective on how condition status varies throughout the year. Groups containing fewer than 50 observations are identified separately to avoid overinterpreting limited data. These seasonal differences may also reflect assessment timing and asset characteristics rather than direct environmental effects.
"""

# ╔═╡ 8eeb40e8-93cd-410b-b6d7-30fa362dca6e
interesting_plot4 = let
    d = dropmissing(comparison_data, [:Month, :Climate])
    d = d[isfinite.(d.Score) .& isfinite.(d.Month) .&
          isfinite.(d.Climate) .& (d.Month .>= 1) .&
          (d.Month .<= 12) .& (d.Month .== round.(d.Month)), :]

    nrow(d) > 0 || error("No valid seasonal observations.")

    season_map = ["Winter", "Winter", "Spring", "Spring", "Spring",
                  "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"]

    d.Season = season_map[Int.(d.Month)]
    codes = sort(unique(d.Climate))

    function season_plot(season)
        groups = [d.Score[(d.Climate .== c) .& (d.Season .== season)]
                  for c in codes]

        counts = length.(groups)
        rates = [counts[i] < 50 ? NaN : 100 * mean(groups[i] .< 88)
                 for i in eachindex(codes)]

        labels = ["$(Int(codes[i]))\n(n=$(counts[i]))"
                  for i in eachindex(codes)]

        p = bar(1:length(codes), rates;
            label="",
            color=:steelblue,
            bar_width=0.65,
            xticks=(1:length(codes), labels),
            xlims=(0.5, length(codes)+0.5),
            ylims=(0, 110),
            yticks=0:20:100,
            ywiden=false,
            xlabel="Climate Code",
            ylabel="Not Green (%)",
            title=season,
            grid=false
        )

        for i in eachindex(codes)
            label = counts[i] < 50 ? "Low n" :
                    "$(round(rates[i]; digits=1))%"
            y = counts[i] < 50 ? 8 : rates[i] + 4
            annotate!(p, i, y, text(label, 9))
        end

        p
    end

    plots = [season_plot(s) for s in
             ["Winter", "Spring", "Summer", "Fall"]]

    plot(plots...;
        style...,
        layout=(2, 2),
        size=(1300, 950)
    )
end

# ╔═╡ aa888638-4879-40f8-b7be-9b7d961d0f21
md"""
## Interesting Plot 5
"""

# ╔═╡ 59fb2405-2cbe-4979-beb7-cbf24b907687
md"""
The seasonal and climate comparisons are further summarized using a heatmap, allowing differences in not-green assessment percentages to be examined across both variables simultaneously. The visualization highlights combinations of climate category and assessment season with relatively high or low proportions of not-green observations, providing a more consolidated view of the patterns examined in the preceding analysis. Differences across these combinations suggest that condition status may reflect several interacting characteristics rather than a single factor. However, groups with limited observations require cautious interpretation, particularly where relatively large percentages are based on small sample sizes.
"""

# ╔═╡ 8cb1f1b2-e752-4110-8ed9-17c5b0eefdf2
interesting_plot5 = let
    d = dropmissing(comparison_data, [:Month, :Climate])
    d = d[isfinite.(d.Score) .& isfinite.(d.Month) .&
          (d.Month .>= 1) .& (d.Month .<= 12) .&
          (d.Month .== round.(d.Month)), :]

    nrow(d) > 0 || error("No valid seasonal observations.")

    season_map = ["Winter", "Winter", "Spring", "Spring", "Spring",
                  "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"]

    seasons = ["Winter", "Spring", "Summer", "Fall"]
    d.Season = season_map[Int.(d.Month)]
    climate = string.(d.Climate)
    codes = sort(unique(climate))

    groups = [d.Score[(climate .== codes[i]) .&
                      (d.Season .== seasons[j])]
              for i in eachindex(codes), j in 1:4]

    counts = length.(groups)
    rates = [isempty(g) ? NaN : 100 * mean(g .< 88) for g in groups]

    p = heatmap(1:4, 1:length(codes), rates;
        style...,
        xticks=(1:4, seasons),
        yticks=(1:length(codes), codes),
        xlabel="Assessment Season",
        ylabel="Climate Code",
        title="Not-Green Assessments by Season and Climate",
        color=cgrad([:royalblue, :lightskyblue, :moccasin, :darkorange]),
        clims=(0, 100),
        colorbar_title="Not green (%)",
        yflip=true,
        size=(1200, max(800, 65*length(codes)))
    )

    for i in eachindex(codes), j in 1:4
        n = counts[i, j]
        label = n == 0 ? "No data" :
                "$(round(rates[i, j]; digits=1))%\n(n=$n)"

        text_color = n > 0 && (rates[i, j] < 30 || rates[i, j] > 75) ?
                     :white : :black

        annotate!(p, j, i, text(label, 9, text_color))
    end

    p
end

# ╔═╡ aa974399-dc5f-4858-97e1-d15dc3f44ce8
md"""
## Interesting Plot 6
"""

# ╔═╡ 986775ce-5341-4597-9c3d-20fe0f785b58
md"""
Assessment recency provides another perspective on the relationship between assessment history and current boiler condition. The proportion of not-green assessments is compared across different time intervals since the previous assessment, including boilers with no recorded assessment history. Differences in these proportions may indicate that the time elapsed between assessments contains useful information for understanding variations in boiler condition. Comparing each group with the overall not-green percentage also helps identify whether certain reassessment intervals are associated with relatively poorer condition.
"""

# ╔═╡ 00156390-9033-438e-ab74-b9393ef9b1cf
interesting_plot6 = let
    d = comparison_data[isfinite.(comparison_data.Score), :]
    valid = [ismissing(r) || (isfinite(r) && r >= 0) for r in d.Recency]
    d = d[valid, :]

    function recency_group(r)
        if ismissing(r)
            return 1
        elseif r <= 1
            return 2
        elseif r <= 3
            return 3
        elseif r <= 5
            return 4
        else
            return 5
        end
    end

    categories = recency_group.(d.Recency)

    groups = [d.Score[categories .== i] for i in 1:5]
    counts = length.(groups)

    rates = [isempty(g) ? NaN : 100 * mean(g .< 88) for g in groups]

    labels = [
        "No previous\nassessment",
        "0–1 years",
        "2–3 years",
        "4–5 years",
        "6+ years"
    ]

    labels = ["$(labels[i])\n(n=$(counts[i]))" for i in 1:5]

    overall = 100 * mean(d.Score .< 88)

    p = bar(1:5, rates;
        style...,
        color=:steelblue,
        bar_width=0.65,
        label="",
        xticks=(1:5, labels),
        xlims=(0.5, 5.5),
        ylims=(0, 112),
        yticks=0:20:100,
        ywiden=false,
        xlabel="Years Since Previous Assessment",
        ylabel="Not-Green Assessments (%)",
        size=(1200, 800),
        bottom_margin=18Plots.mm
    )

    hline!(p, [overall];
        color=:gray,
        linestyle=:dash,
        linewidth=2,
        label="Overall ($(round(overall; digits=1))%)"
    )

    for i in 1:5
        if counts[i] == 0
            annotate!(p, i, 5, text("No data", 10))
        else
            annotate!(p, i, min(rates[i] + 4, 107),
                      text("$(round(rates[i]; digits=1))%", 10))
        end
    end

    plot!(p; legend=:outertop)
    p
end

# ╔═╡ 023a5858-1710-4ae0-8b81-939a679b4014
begin
    savefig(interesting_plot1, "interesting_plot1.png")
    savefig(interesting_plot2, "interesting_plot2.png")
    savefig(interesting_plot3, "interesting_plot3.png")
    savefig(interesting_plot4, "interesting_plot4.png")
    savefig(interesting_plot5, "interesting_plot5.png")
    savefig(interesting_plot6, "interesting_plot6.png")
end

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

# ╔═╡ 09771c81-3e9c-42ea-be1a-e389f0bcd906
md"""
To quantify the strength of monotonic relationships between variables in the dataset, a pairwise correlation analysis was conducted. Spearman’s rank correlation coefficient (ρ) was calculated for each combination of variable pairs. The results indicate that Score, an Expert assessor’s assigned condition rating of a given boiler, is not explained by a single correlation with another variable. Therefore, a multivariate approach is needed.
"""

# ╔═╡ 5dc1a1ca-22ae-4cca-9d52-a83b71e2535c
begin
	# create matrix for correlation
	boilers_spearman = Matrix(select(boilers_raw_data, cor_vars))
	# calculate Spearman correlation coefficients
    spearman_matrix = corspearman(boilers_cor)
end

# ╔═╡ 6dffd504-3993-443b-a5e8-b4ba47b3b107
begin
	# statistical significance
	x = [1, 2, 3, 4, 5]
	y = [5, 6, 7, 8, 7]
	test_result = CorrelationTest(x, y_vec, method=:spearman)
	println(test_result)
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
# ╟─7606e704-083e-4a5f-8da5-bcdf3bfa3216
# ╠═12b33241-044c-4631-a9b9-e9c44e24847a
# ╟─0263eb8b-4e72-4ebc-988b-e524302c2d6a
# ╟─5d26ccac-1d20-44f7-8696-440e91eeabf1
# ╟─a684491d-5717-498b-b3a7-1c864afb9a31
# ╟─611d8e7c-fdfe-43e8-a12d-647272d2099e
# ╟─4ae06b54-432d-4434-9bc8-47c8a91240e7
# ╟─e7fae12f-3e0b-4e9f-a874-4cb0db599494
# ╟─345182d1-e582-4498-ad3a-daed465891ec
# ╟─b41a40ab-1eee-4f01-a8f3-a32a249cf6c7
# ╟─9c2bd780-dca9-4778-8ae3-58003769aefb
# ╟─d9299f4a-f0f2-4bc4-aec6-35f6a9ac652a
# ╟─e23f9964-af24-45e1-bca5-987438d340b1
# ╟─4757dd14-b688-49cc-8e73-3878eceff35e
# ╟─d0b92a37-92ed-4ce4-85cb-beba9eaab65a
# ╟─50a8cd94-bf14-44b8-b170-319fe9b97b16
# ╟─8eeb40e8-93cd-410b-b6d7-30fa362dca6e
# ╟─aa888638-4879-40f8-b7be-9b7d961d0f21
# ╟─59fb2405-2cbe-4979-beb7-cbf24b907687
# ╟─8cb1f1b2-e752-4110-8ed9-17c5b0eefdf2
# ╟─aa974399-dc5f-4858-97e1-d15dc3f44ce8
# ╟─986775ce-5341-4597-9c3d-20fe0f785b58
# ╠═00156390-9033-438e-ab74-b9393ef9b1cf
# ╠═023a5858-1710-4ae0-8b81-939a679b4014
# ╟─2a9579fc-6ec0-4741-859a-6c9105dc8f7d
# ╟─90916ca7-d937-4a85-b40d-a69c23e38562
# ╠═d832c11a-a8b3-4f92-9a30-caccc1d6b28e
# ╠═c971b211-e043-47b3-88b7-ad73b6872796
# ╟─b9c194d6-87fe-459e-82f6-e551d4889747
# ╟─09771c81-3e9c-42ea-be1a-e389f0bcd906
# ╠═5dc1a1ca-22ae-4cca-9d52-a83b71e2535c
# ╠═ea18b937-bd81-4517-a04a-ca4987d37319
# ╠═6dffd504-3993-443b-a5e8-b4ba47b3b107
# ╠═17d5f78e-7188-4496-b00e-36be135fe37f
# ╟─cbefaad9-0df1-4b1a-a3fb-adb5bc2e0f32
# ╟─e553e6b3-7b8e-4029-b823-eb98362c49a4
# ╠═c4f0042e-dac8-4caf-b983-8bc9678b4d1b
# ╟─c2324815-aa18-43a0-b9f3-cd7dbcde5647
# ╠═8bc9c3b1-0ffb-451b-bae9-bf3d209355b6
# ╟─2defcea8-94d2-4182-b313-7b242d185c08
