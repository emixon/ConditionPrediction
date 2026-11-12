### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
begin
    import Pkg

    #Pkg.activate(Base.current_project())
    #Pkg.instantiate()
    
    #using ConditionPrediction
    using CSV
    using DataFrames
    using StatsPlots
    using StatsPlots.PlotMeasures: mm
    using Statistics
    using LinearAlgebra
    using StatsBase
    using MLJ
    using MLJFlux
    using MLJLinearModels
    using Flux
    using StatisticalMeasures
    import Optimisers
    import Random
end

# ╔═╡ d779c071-1218-45cc-ae66-16e01e12e1e3
using MLJMultivariateStatsInterface

# ╔═╡ e670ba1c-19aa-4a93-93b4-f99562de37b4


# ╔═╡ f92de274-9c7f-4ea2-84a5-d18f97044498
md"""
## Load Data and Wrangle
"""

# ╔═╡ 3426ad4c-a340-46c3-994a-0e30dccef086
md"""
## Data Wrangling
"""

# ╔═╡ b078a14b-049e-4398-9c79-85ef7fde2843
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
    observation_modes = Matrix(F.Vt')

    return F, explained_variance, explained_ratio, observation_modes
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
        size = (1600, 1000),
        dpi = 200,
        titlefontsize = 40,
        guidefontsize = 30,
        tickfontsize = 24,
    
        left_margin = 18mm,
        bottom_margin = 20mm,
        top_margin = 10mm,
        right_margin = 10mm,
    )
end

# ╔═╡ 653ae9d0-bca6-4778-a8d3-9d07dbf143c2
function GetLoadingsHeatMap(F, pca_variables; squared = false)
    k = min(size(F.U, 2), length(pca_variables))
    L = F.U[:, 1:k]  # variables × principal components

    limit = squared ? maximum(L .^ 2) : maximum(abs, L)

    heatmap(
        1:k,
        string.(pca_variables),
        squared ? L .^ 2 : L;
        xlabel = "Principal Component",
        ylabel = "Variable",
        title = "PCA Loading Heatmap",
        colorbar_title = squared ? "Squared Loading" : "Loading",
        clims = squared ? (0, limit) : (-limit, limit),
        c = squared ? :viridis : :RdBu,
        yflip = true,
        size = (1600, 1000),
        dpi = 200,
        titlefontsize = 40,
        guidefontsize = 30,
        tickfontsize = 24,
        left_margin = 18mm,
        bottom_margin = 20mm,
        top_margin = 10mm,
        right_margin = 10mm,
    )
end

# ╔═╡ e346bbf0-066e-4de6-bcdd-6c2732756d17
function GetPCScoreHistogram(
    observation_modes,
    assessment_scores,
    pc;
    threshold = 88,
    nbins = 100,
)
    pc_values = observation_modes[:, pc]

    green = assessment_scores .>= threshold
    not_green = .!green

    xmin, xmax = extrema(pc_values)
    bins = range(xmin, xmax; length = nbins + 1)

    p = histogram(
        pc_values[not_green];
        bins = bins,
        normalize = :pdf,
        color = :orange,
        alpha = 0.55,
        label = "Not Green (Score < $threshold)",
        xlabel = "PC$pc Coordinate",
        ylabel = "Density",
        title = "PC$pc Distribution by Assessment Score",
        size = (1200, 800),
        dpi = 200,
        titlefontsize = 24,
        guidefontsize = 18,
        tickfontsize = 14,
        legendfontsize = 14,
        left_margin = 18mm,
        bottom_margin = 20mm,
        top_margin = 10mm,
        right_margin = 10mm,
    )

    histogram!(
        p,
        pc_values[green];
        bins = bins,
        normalize = :pdf,
        color = :green,
        alpha = 0.45,
        label = "Green (Score ≥ $threshold)",
    )

    return p
end

# ╔═╡ 5a3a0107-bed6-440a-9805-5a2b9b483b72
function GetScorePercentGreen3D(
    observation_modes,
    df;
    threshold = 88,
)
    @assert size(observation_modes, 1) == nrow(df)

    green = df.Score .>= threshold
    not_green = .!green

    p = scatter(
        observation_modes[not_green, 1],
        observation_modes[not_green, 2],
        observation_modes[not_green, 3],
        color = :darkorange,
        alpha = 0.45,
        markersize = 3,
        markerstrokewidth = 0,
        label = "Not Green (Score < $threshold)",
        xlabel = "PC1 Observation Coordinate",
        ylabel = "PC2 Observation Coordinate",
        zlabel = "PC3 Observation Coordinate",
        title = "Boiler PCA Coordinates and Facility Green Rate",
        #zlims = (0, 100),
        camera = (25, 60),
        size = (1800, 1200),
        dpi = 200,
        titlefontsize = 28,
        guidefontsize = 20,
        tickfontsize = 16,
        legendfontsize = 16,
        left_margin = 14mm,
        bottom_margin = 14mm,
        top_margin = 10mm,
        right_margin = 18mm,
    )

    scatter!(
        p,
        observation_modes[green, 1],
        observation_modes[green, 2],
        observation_modes[green, 3],
        color = :forestgreen,
        alpha = 0.45,
        markersize = 3,
        markerstrokewidth = 0,
        label = "Green (Score ≥ $threshold)",
    )

    return p
end

# ╔═╡ af1af019-a784-4144-a632-e66fca8b1c3b
let
    pca_vars_single_lf = [#vcat([
        :RSL,
        #:NormalizedAge,
        #:EffectiveAge,
        :PredictedCI,  # Might be useful paired with RSL for some reason
        #:Recency,
        #:PrevScore,
        :PrevAssessed,  # Best performing between Recency/PrevScore/PrevAssessed
        #:Qty,
        :NormalizedAssetAge,
        #:AssetAge,
        #:AssetDesignLife,
        :MonthSin,
        :MonthCos,
        :LogCRV,  # Currently divided by Qty to normalize individual boiler cost
        :Climate,
        :PercentGreen
    ]#, 
    #spec_indicator_cols, # Too many columns
    #fac_indicator_cols)  # Too many columns
    modes = 1

    F, explained_variance, explained_ratio, observation_modes =
        CalcSVDAndExplainedVariance(pca_vars_single_lf, wrangled_boilers)

    pca_plot = GetPCAPlot(length(pca_vars_single_lf), explained_ratio)

    loadings_heatmap = GetLoadingsHeatMap(
        F,
        pca_vars_single_lf;
        squared = true,
    )

    histogram_plot = GetPCScoreHistogram(
            observation_modes,
            wrangled_boilers.Score,
            1;
            threshold = 85,
        )

    scatter_plot = GetScorePercentGreen3D(
        observation_modes,
        wrangled_boilers;
        threshold = 85,
    )

    pca_plot, loadings_heatmap, histogram_plot, scatter_plot
    #savefig(score_histograms, "RSL_CI_no_PGreen.png")
end

# ╔═╡ 6f6e6ca0-f018-4ddd-b8e9-61dd05f00560
md"""
### Assessment outcome probability model

The predictive model is intended to support condition-assessment planning rather than
only estimate a single condition score. For a component being considered for assessment
at a candidate date, the model estimates the probability that a newly collected field
assessment would fall within each maintenance decision region.

The predictor variables represent the component state and context available before a
new assessment, including lifecycle characteristics, modeled deterioration state,
historical assessment information, and relevant facility or environmental attributes.
The response variable is the field-assessed condition score recorded during the new
assessment.

For the preliminary analysis, the score distribution is simplified into two decision
regions:

- **Do Nothing:** Score greater than or equal to 85
- **Needs Action:** Score less than 85

The ANN estimates the conditional probabilities:

\[
P(\mathrm{Score} \geq 85 \mid X)
\]

and

\[
P(\mathrm{Score} < 85 \mid X)
\]

where \(X\) represents the component state and contextual attributes available at the
candidate assessment date.
"""

# ╔═╡ 31d5dc1e-5f96-4841-9ce1-1877ca215454
md"""
### Feature set and binary response variable

The preliminary ANN uses component lifecycle, modeled condition, assessment-history,
seasonal, cost, and environmental variables as predictors. The target variable is a
binary maintenance decision region derived from the field-assessed condition score.

Observations with Score greater than or equal to 85 were classified as `DoNothing`,
while observations with Score below 85 were classified as `NeedsAction`. This
transformation allows the model to estimate the probability that a candidate assessment
would reveal a condition requiring action rather than only producing a single point
estimate of condition.
"""

# ╔═╡ 2a218ae9-897e-4f89-8895-87826b38d47a
begin
    # Feature Selection
    feature_cols = [
        :RSL,
        :PredictedCI,
        :PrevAssessed,
        :NormalizedAssetAge,
        :MonthSin,
        :MonthCos,
        :LogCRV,
        :Climate,
        #:PercentGreen,
    ]

    # Select the desired features
    model_df = dropmissing(
        select(wrangled_boilers, vcat(feature_cols, [:Score]))
    )

    # Convert to Float32 for Flux
    X = Float32.(select(model_df, feature_cols))

    # Setup the output categories (simple split on 85 threshold for now)
    # Create this as a new column so numerical Score is still available
    model_df.ActionCategory = categorical(
        ifelse.(
            model_df.Score .>= 85,
            "DoNothing",
            "NeedsAction",
        )
    )
    y = model_df.Score
end

# ╔═╡ 741c414a-b1e9-4e26-aad0-359e49588d8b
md"""
## Train/Test Partitioning
"""

# ╔═╡ 139075ff-92d9-480a-b53b-4bca487b100f
begin
    # use partition to create train and test datasets
    train_rows, test_rows = partition(
        eachindex(y), # just says to use all rows
        0.80; # 80/20 split between train and test
        stratify = y, # keeps a region proportions similar between train and test
        rng = 1234, # makes this repeatable
    )
end

# ╔═╡ b268474e-4bcb-4ae8-ae6b-cd1eeb8210da
md"""
## Helper Functions
"""

# ╔═╡ 645d1d8a-f76e-4857-a0e7-66f75fcf4e23
# define function for plotting residuals as a histogram
function residuals_histogram(y_test, y_train, ŷ_test, ŷ_train)
    # calculate residuals
    residuals_train = y_train .- ŷ_train
    residuals_test  = y_test .- ŷ_test
    
    histogram(
        residuals_train;
        bins = 30,
        alpha = 0.5,
        label = "Train Residuals",
        xlabel = "Residual",
        ylabel = "Count",
        
        guidefontsize = 30,
        tickfontsize = 24,
        legendfontsize = 24,
        legend = :topleft,

        size = (1250, 1050),
        dpi = 200,

        left_margin = 28mm,
        bottom_margin = 20mm,
        top_margin = 14mm,
        right_margin = 20mm,
    )

    histogram!(
        residuals_test;
        bins = 30,
        alpha = 0.5,
        label = "Test Residuals",
    )

end

# ╔═╡ fdc35e29-2bf6-4348-a03f-de9eaf0769f6
function reg_summary_df(y_train, ŷ_train, y_test, ŷ_test)

    df = DataFrame(
        Dataset = ["Train", "Test"],
        R2 = [
            1 - sum((y_train .- ŷ_train).^2) / sum((y_train .- mean(y_train)).^2),
            1 - sum((y_test .- ŷ_test).^2) / sum((y_test .- mean(y_test)).^2)
        ],
        MAE = [
            mean(abs.(y_train .- ŷ_train)),
            mean(abs.(y_test .- ŷ_test))
        ],
        RMSE = [
            sqrt(mean((y_train .- ŷ_train).^2)),
            sqrt(mean((y_test .- ŷ_test).^2))
        ]
    )
    return df
end

# ╔═╡ f0b871d9-5d18-43d5-8d27-741ac9a06363
function plot_performance(y_train, y_test, ŷ_train, ŷ_test)
    # ideal y=x line
      perfect_line = [minimum(vcat(y_train, y_test, ŷ_train, ŷ_test)), maximum(vcat(y_train, y_test, ŷ_train, ŷ_test))]

    plot(
        perfect_line,
        perfect_line;
        label = "Perfect Fit",
        color = :black,
        linewidth = 4,
        linestyle = :dot,
    )

    # plot training data
    scatter!(
        y_train,
        ŷ_train;
        label = "Train",
        alpha = 0.6,
        xlabel = "Actual",
        ylabel = "Predicted",
        guidefontsize = 30,
        tickfontsize = 24,
        legendfontsize = 20,
        legend = :bottomright,
        markersize = 10,
        color = :dodgerblue,

        size = (1250, 1050),
        dpi = 200,

        left_margin = 28mm,
        bottom_margin = 20mm,
        top_margin = 14mm,
        right_margin = 20mm,
    )
    
    # plot testing data
    
    scatter!(
        y_test,
        ŷ_test;
        label = "Test",
        alpha = 0.6,
        markersize = 10,
        color = :orange,
    )
end

# ╔═╡ d05b79c4-0235-45a1-933d-c8147e7b0e0b
md"""
## Ordinary Least-Squares Regression
"""

# ╔═╡ 3b280832-9ad9-437c-becd-e79eb3a51e91
begin

    # calculate number of independent variables if needed later
    inputFeatures = length(feature_cols)

    # ordinary least squares linear regression method
    ols = MLJLinearModels.LinearRegressor()

    # create OLS model pipeline that includes data standardization
    # Standardizer() scales features to have mean of 0 and st dev of 1
    ols_pipeline = Standardizer() |> ols

     # Cross Validation (this trains the model on the selected training data)
    # Resampling is doing 5 fold validation on training data
    
    y_continuous = coerce(y, Continuous) # specify y as continuous
    
    cv_results_ols = evaluate(
        ols_pipeline,
        X[train_rows, :],
        y_continuous[train_rows];
        resampling = CV(nfolds = 5, rng = 1234),
        measure = [rmse, mae, rsq],
        verbosity = 0
    )

    # cross validation results
    @show cv_results_ols.measurement
    @show cv_results_ols.per_fold

    per_fold_df = DataFrame(
    Fold = 1:5,
    RMSE = cv_results_ols.per_fold[1],
    MAE  = cv_results_ols.per_fold[2],
    R2   = cv_results_ols.per_fold[3]
    )
    
    full_results_df = vcat(
        per_fold_df,
        DataFrame(
            Fold = ["Overall"],
            RMSE = [cv_results_ols.measurement[1]],
            MAE  = [cv_results_ols.measurement[2]],
            R2   = [cv_results_ols.measurement[3]]
        )
    )


end

# ╔═╡ a12d3efa-e011-4939-bc03-ee75cf72d5d0
begin
    # Testing the model against held out data

    # Fit the model with all training data
    mach_ols = machine(ols_pipeline, X[train_rows, :], y_continuous[train_rows])
    fit!(mach_ols, verbosity = 0)

    # Predicted Scores
    ŷ_train_ols = MLJ.predict(mach_ols, X[train_rows, :])
    ŷ_test_ols  = MLJ.predict(mach_ols, X[test_rows, :])

    # Actual Scores
    y_train_ols = y_continuous[train_rows]
    y_test_ols  = y_continuous[test_rows]

    # Metrics
    summary_df_ols = reg_summary_df(y_train_ols, ŷ_train_ols, y_test_ols, ŷ_test_ols)
    summary_df_ols
end

# ╔═╡ 3b281328-7809-46a7-8293-02499727e4ab
# define function for calculating accuracy
function model_accuracy(y_test, y_train, ŷ_test, ŷ_train)
	test_accuracy = mean((y_test_ols .> 85) .== (ŷ_test_ols .> 85)) * 100
	println("Test Accuracy: $(test_accuracy)%")
	train_accuracy = mean((y_train_ols .> 85) .== (ŷ_train_ols .> 85)) * 100
	println("Train Accuracy: $(train_accuracy)%")
	return test_accuracy, train_accuracy
end

# ╔═╡ 5016e7fd-380b-4a38-8e7a-7e63cbb1518a
begin
    ols_coefs = fitted_params(mach_ols).linear_regressor.coefs

    ols_coef_df = DataFrame(
        Feature = feature_cols,
        Coefficient = ols_coefs
    )

    ols_coef_df
end

# ╔═╡ 618282f9-e415-4a17-b8f0-92e92e6a7668
begin
	plot_performance(y_train_ols, y_test_ols, ŷ_train_ols, ŷ_test_ols)
	savefig("ols_performance.png")
end

# ╔═╡ 39d6f28c-405a-4eb7-95dd-dd01c8de7f25
begin
	residuals_histogram(y_test_ols, y_train_ols, ŷ_test_ols, ŷ_train_ols)
	savefig("ols_residuals_hist.png")
end

# ╔═╡ f92f7120-79fd-4c0b-8632-e29cb161788a
model_accuracy(y_test_ols, y_train_ols, ŷ_test_ols, ŷ_train_ols)

# ╔═╡ 8c92dffd-5cb4-46f5-b07d-74ec27265384
fitted_params(mach_ols)

# ╔═╡ 7b69149a-4890-41c0-94ed-2fb9a6a571f3
md"""
## Ridge Regression
"""

# ╔═╡ 30d268c8-5e91-41f0-a9f0-3b58ed5ce3fe
begin

    # Ridge regression method
    ridge = MLJLinearModels.RidgeRegressor()

    # 5-Fold Cross Validation to Select Lambda (Regularization coefficient)
    ridge_tuned = TunedModel(
        model = ridge,
        tuning = Grid(goal = 20),
        resampling = CV(nfolds = 5, rng = 1234),
        measure = rmse,
        range = range(ridge, :lambda,
                      lower = 1e-4,
                      upper = 1000,
                      scale = :log)
    )

    ridge_pipeline = Standardizer() |> ridge_tuned
end

# ╔═╡ d6699ec3-50db-47f7-be0b-303b0f9c147b
begin
    # Testing the model against held out data

    # Fit the model with all training data
    mach_ridge = machine(ridge_pipeline, X[train_rows, :], y_continuous[train_rows])
    fit!(mach_ridge, verbosity = 0)

    # Predicted Scores
    ŷ_train_ridge = MLJ.predict(mach_ridge, X[train_rows, :])
    ŷ_test_ridge  = MLJ.predict(mach_ridge, X[test_rows, :])

    # Actual Scores
    y_train_ridge = y_continuous[train_rows]
    y_test_ridge  = y_continuous[test_rows]

    # Metrics
    summary_df_ridge = reg_summary_df(y_train_ridge, ŷ_train_ridge, y_test_ridge, ŷ_test_ridge)
    summary_df_ridge
end

# ╔═╡ 56b92223-6559-4fb3-aee4-f5623c29bad2
begin
	plot_performance(y_train_ridge, y_test_ridge, ŷ_train_ridge, ŷ_test_ridge)
	savefig("ridge_performance.png")
end

# ╔═╡ 6a054746-79b0-4f1f-98a0-05da22b2b6cc
begin
	residuals_histogram(y_test_ridge, y_train_ridge, ŷ_test_ridge, ŷ_train_ridge)
	savefig("ridge_residuals_hist.png")
end

# ╔═╡ 62a5eea0-e180-43ed-9f5b-1a190a625e48
model_accuracy(y_test_ridge, y_train_ridge, ŷ_test_ridge, ŷ_train_ridge)

# ╔═╡ 7bba9815-2ec7-477f-88b9-30e8d785ad62
fitted_params(mach_ridge)

# ╔═╡ d7c76249-16a5-4116-9f00-7b2c8a5ed474
begin
    fp_ridge = fitted_params(mach_ridge)

    ridge_coefs = fp_ridge.deterministic_tuned_model.best_fitted_params.coefs

    ridge_lambda =
        report(mach_ridge).deterministic_tuned_model.best_model.lambda

    ridge_coef_df = DataFrame(
        Feature = feature_cols,
        Coefficient = ridge_coefs
    )

    ridge_coef_df, ridge_lambda
end


# ╔═╡ f8d5ee66-ddf4-4c5a-ad92-aea4da10fcce
md"""
## Lasso Regression
"""

# ╔═╡ 59bf7ef6-c3bf-4cf5-96ef-36335f333327
md"""
Because the ANN was formulated as a probabilistic binary classifier rather than a
continuous score-regression model, performance was evaluated using log loss instead
of RMSE. Log loss rewards models that assign high probability to the observed condition
region and penalizes overconfident incorrect predictions. This emphasis on probability
quality is appropriate because the predicted decision-region probabilities are intended
to support later Value of Information calculations.
"""

# ╔═╡ af2f63ab-3172-4e1f-950e-8449197da08e
begin

    # Lasso regression method
    lasso = MLJLinearModels.LassoRegressor()

    # 5-Fold Cross Validation to Select Lambda (Regularization coefficient)
    lasso_tuned = TunedModel(
        model = lasso,
        tuning = Grid(goal = 20),
        resampling = CV(nfolds = 5, rng = 1234),
        measure = rmse,
        range = range(ridge, :lambda,
                      lower = 1e-4,
                      upper = 1000,
                      scale = :log)
    )

    lasso_pipeline = Standardizer() |> lasso_tuned
end

# ╔═╡ 3989ad12-80ba-4244-9e13-adecbb72c956
begin
    # Testing the model against held out data

    # Fit the model with all training data
    mach_lasso = machine(lasso_pipeline, X[train_rows, :], y_continuous[train_rows])
    fit!(mach_lasso, verbosity = 0)

    # Predicted Scores
    ŷ_train_lasso = MLJ.predict(mach_lasso, X[train_rows, :])
    ŷ_test_lasso  = MLJ.predict(mach_lasso, X[test_rows, :])

    # Actual Scores
    y_train_lasso = y_continuous[train_rows]
    y_test_lasso  = y_continuous[test_rows]

    # Metrics
    summary_df_lasso = reg_summary_df(y_train_lasso, ŷ_train_lasso, y_test_lasso, ŷ_test_lasso)
    summary_df_lasso
end

# ╔═╡ 0bd9b3c8-6747-46e2-8981-d60a49f64262
begin
	plot_performance(y_train_lasso, y_test_lasso, ŷ_train_lasso, ŷ_test_lasso)
	savefig("lasso_performace.png")	
end

# ╔═╡ ae81bb04-0644-478d-80ff-7cf7ed4691db
begin
	residuals_histogram(y_test_lasso, y_train_lasso, ŷ_test_lasso, ŷ_train_lasso)
	savefig("lasso_residuals_hist.png")	
end

# ╔═╡ e9f6b1c5-35d6-4470-9360-f0e546465009
model_accuracy(y_test_lasso, y_train_lasso, ŷ_test_lasso, ŷ_train_lasso)

# ╔═╡ edd27b43-8204-444a-82a8-43ec5170b5f4
fitted_params(mach_lasso)

# ╔═╡ fe3e29de-2765-45f6-9c6f-d67d90756d4d
begin
    fp_lasso = fitted_params(mach_lasso)

    lasso_coefs = fp_lasso.deterministic_tuned_model.best_fitted_params.coefs

    lasso_lambda =
        report(mach_lasso).deterministic_tuned_model.best_model.lambda

    lasso_coef_df = DataFrame(
        Feature = feature_cols,
        Coefficient = lasso_coefs
    )

    lasso_coef_df, lasso_lambda
end

# ╔═╡ 6fb87d58-690b-489a-bde9-8e86f717c19a
begin
    coefficient_df = DataFrame(
        Parameter = vcat(feature_cols, ["Lambda"]),
        OLS = vcat(ols_coefs, [missing]),
        Ridge = vcat(ridge_coefs, [ridge_lambda]),
        Lasso = vcat(lasso_coefs, [lasso_lambda])
    )

    coefficient_df
end

# ╔═╡ 86b015ac-f3cd-45e6-bbe0-6614522928f2
report(mach_ridge).deterministic_tuned_model.best_history_entry.model.lambda

# ╔═╡ 8c8eefa7-f702-4185-bf11-02722c577fed
md"""
## Logistic Regression
"""

# ╔═╡ 1a443f32-e338-45e6-b6ce-39bcccb3a8dd
begin
    logistic = MLJLinearModels.LogisticClassifier()

    logistic_tuned = TunedModel(
        model = logistic,
        tuning = Grid(goal = 20),
        resampling = StratifiedCV(nfolds = 5, rng = 1234),
        measure = log_loss,
        range = range(
            logistic,
            :lambda,
            lower = 1e-4,
            upper = 100,
            scale = :log
        )
    )

    logistic_pipeline = Standardizer() |> logistic_tuned
end

# ╔═╡ 424c1659-f241-4eab-93fa-02be380bbb2c
# categorical data for logistic regression
y_cat = model_df.ActionCategory

# ╔═╡ e32ee31d-8c4b-416e-a168-9d93fd1e3bd4
begin

    # Testing the model against held out data

    # Fit the model with all training data
    mach_logistic = machine(logistic_pipeline, X[train_rows, :], y_cat[train_rows])
    fit!(mach_logistic, verbosity = 0)

    # Probabilities
    y_prob_train = MLJ.predict(mach_logistic, X[train_rows, :])
    y_prob_test  = MLJ.predict(mach_logistic, X[test_rows, :])

    # Actual Scores
    y_train_logistic = y_cat[train_rows]
    y_test_logistic  = y_cat[test_rows]

    # class predictions ("Do Nothing" vs "Needs Action")
    y_pred_train = mode.(y_prob_train)
    y_pred_test  = mode.(y_prob_test)
    
    # Metrics
    summary_df = DataFrame(
        Dataset  = ["Train", "Test"],
        LogLoss  = [
            log_loss(y_prob_train, y_train_logistic),
            log_loss(y_prob_test, y_test_logistic)
        ],
        Accuracy = [
            accuracy(y_pred_train, y_train_logistic),
            accuracy(y_pred_test, y_test_logistic)
        ]
    )
end

# ╔═╡ f341be50-37f9-45b2-9597-0db46997e6a7
begin
    cm = confusion_matrix(y_test_logistic, y_pred_test)

    labels = string.(levels(y_test_logistic))

    p = heatmap(
        labels,
        labels,
        cm.mat;
        xlabel = "Predicted",
        ylabel = "Actual",
        cbar = false,
        aspect_ratio = :equal,
        size = (1200, 920),
        guidefontsize = 30,
        tickfontsize = 20,
        dpi = 200,
        left_margin = 28mm,
        bottom_margin = 20mm,
        top_margin = 14mm,
    )

    for i in axes(cm.mat, 1), j in axes(cm.mat, 2)
        annotate!(
            p,
            j,
            i,
            text(
                string(cm.mat[i, j]),
                28,
                :white,
                :center
            )
        )
    end

    p
end

# ╔═╡ 316d58ec-0d6d-4e89-ad52-bf90693437fb
   cm2 = confusion_matrix(y_test_logistic, y_pred_test)

# ╔═╡ e47deada-667c-42d6-b7f6-3667bb33fd00
begin
    begin
        y_prob = MLJ.predict(mach_logistic, X[test_rows, :])
    
        positive_class = levels(y_cat)[2]
    
        probs = pdf.(y_prob, positive_class)
    
        actual_positive = y_cat[test_rows] .== positive_class
        actual_negative = .!actual_positive
    
        histogram(
            probs[actual_negative];
            alpha = 0.55,
            bins = 20,
            normalize = :pdf,
            label = "Observed DoNothing",
            color = :steelblue
        )
    
        histogram!(
            probs[actual_positive];
            alpha = 0.55,
            bins = 20,
            normalize = :pdf,
            label = "Observed NeedsAction",
            xlabel = "Predicted P(Needs Action)",
            ylabel = "Proportion Within Observed Class",
            color = :darkorange,
            titlefontsize = 40,
            guidefontsize = 30,
            tickfontsize = 24,
            legendfontsize = 20,
    
            size = (1200, 1000),
            dpi = 200,
    
            left_margin = 28mm,
            bottom_margin = 20mm,
            top_margin = 14mm,
            right_margin = 18mm,
        )
    end
savefig("log_prob_sep_hist.png")
end

# ╔═╡ b7f85f88-1f37-4e8a-b65f-c42ffd2f8a1d
begin
    fp = fitted_params(mach_logistic)

    coefs = fp.probabilistic_tuned_model.best_fitted_params.coefs

    intercept = fp.probabilistic_tuned_model.best_fitted_params.intercept

    lambda =
        report(mach_logistic).probabilistic_tuned_model.best_history_entry.model.lambda

    logistic_df = DataFrame(
        Parameter = vcat(["λ", "Intercept"], feature_cols),
        Logistic = vcat([lambda], [intercept], coefs)
    )

    logistic_df
end

# ╔═╡ 3cde7233-438d-4cb4-aeac-ef6cadb7e22d
# ╠═╡ disabled = true
#=╠═╡
begin
    # Current best feature set is included as experiment 1:
    current_best_cols = [
        :RSL,
        :NormalizedAge,
        :EffectiveAge,
        :PredictedCI,
        :Recency,
        :PrevScore,
        :PrevAssessed,
        :NormalizedAssetAge,
        :MonthSin,
        :MonthCos,
        :LogCRV,
        :Climate,
    ]

    # Candidate variables for random feature-set generation:
    candidate_features = [
        :RSL,
        :NormalizedAge,
        :EffectiveAge,
        :PredictedCI,
        :Recency,
        :PrevScore,
        :PrevAssessed,
        :Qty,
        :NormalizedAssetAge,
        :AssetAge,
        :AssetDesignLife,
        :MonthSin,
        :MonthCos,
        :LogCRV,
        :Climate,
        :PercentGreen,
    ]

    number_of_feature_sets = 100
    rng = Random.MersenneTwister(1234)

    # Build unique feature combinations:
    feature_sets = Vector{Vector{Symbol}}()
    seen_feature_sets = Set{Tuple}()

    push!(feature_sets, current_best_cols)
    push!(seen_feature_sets, Tuple(current_best_cols))

    while length(feature_sets) < number_of_feature_sets
        n_selected = Random.rand(rng, 2:length(candidate_features))

        selected_indices = sort(
            Random.randperm(rng, length(candidate_features))[1:n_selected]
        )

        selected_cols = candidate_features[selected_indices]
        feature_key = Tuple(selected_cols)

        if !(feature_key in seen_feature_sets)
            push!(feature_sets, selected_cols)
            push!(seen_feature_sets, feature_key)
        end
    end

    # Run the ANN workflow for every feature set:
    feature_sweep_summaries = NamedTuple[]

    for (i, selected_cols) in enumerate(feature_sets)
        @info "Testing feature set $i / $number_of_feature_sets ($(length(selected_cols)) features)"

        summary = evaluate_feature_set(selected_cols)

        push!(
            feature_sweep_summaries,
            merge(summary, (experiment = i,)),
        )
    end

    feature_sweep_results = DataFrame(feature_sweep_summaries)

    sort!(feature_sweep_results, :cv_log_loss)
end
  ╠═╡ =#

# ╔═╡ 49ba444f-c820-469b-98e3-8d11f61b5229
# ╠═╡ disabled = true
#=╠═╡
begin
	pcacols = [
        :RSL,
        :NormalizedAge,
        :EffectiveAge,
        :PredictedCI,  # Might be useful paired with RSL for some reason
        :Recency,
        :PrevScore,
        :PrevAssessed,  # Best performing between Recency/PrevScore/PrevAssessed
        #:Qty,
        :NormalizedAssetAge,
        #:AssetAge,
        #:AssetDesignLife,
        :MonthSin,
        :MonthCos,
        :LogCRV,  # Currently divided by Qty to normalize individual boiler cost
        :Climate,
        #:PercentGreen
    ]

    npcs = 6
    evaluate_pca_feature_set(pcacols, npcs)
end
  ╠═╡ =#

# ╔═╡ d3b4eed4-e5f1-4cdf-b8ca-88c66c9edb43
# ╠═╡ disabled = true
#=╠═╡
begin
    architectures = Tuple[
        # 2 total layers: 1 hidden layer
        (8,),
        (16,),
        (24,),
        (32,),
        (48,),
        (64,),
        (80,),
        (96,),
        (112,),
        (124,),
    
        # 3 total layers: 2 hidden layers
        (8, 8),
        (16, 8),
        (16, 16),
        (32, 16),
        (32, 32),
        (48, 24),
        (64, 32),
        (64, 64),
        (96, 48),
        (124, 64),
    
        # 4 total layers: 3 hidden layers
        (8, 8, 8),
        (16, 8, 8),
        (16, 16, 8),
        (32, 16, 8),
        (32, 32, 16),
        (48, 24, 12),
        (64, 32, 16),
        (64, 64, 32),
        (96, 48, 24),
        (124, 64, 32),
    
        # 5 total layers: 4 hidden layers
        (8, 8, 8, 8),
        (16, 8, 8, 8),
        (16, 16, 8, 8),
        (32, 16, 8, 8),
        (32, 32, 16, 8),
        (48, 24, 12, 8),
        (64, 32, 16, 8),
        (64, 64, 32, 16),
        (96, 48, 24, 12),
        (124, 64, 32, 16),
    ]
    
    architecture_results = DataFrame(
        hidden = string.(architectures),
        hidden_layers = length.(architectures),
        total_layers = length.(architectures) .+ 1,
        cv_log_loss = fill(NaN, length(architectures)),
    )

    for (i, hidden_layers) in enumerate(architectures)
        @info "Testing architecture $i / $(length(architectures)): $hidden_layers"
    
        architecture_results.cv_log_loss[i] =
            get_cv_log_loss(hidden_layers)
    end
    
    sort!(architecture_results, :cv_log_loss)
end
  ╠═╡ =#

# ╔═╡ 920b4200-3b99-48e9-906e-d637bfdacb08
# ╠═╡ disabled = true
#=╠═╡
begin
    # Include some deliberately very small and very large learning rates:
    learning_rates = [
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
    ]

    learning_rate_results = DataFrame(
        learning_rate = learning_rates,
        cv_log_loss = fill(Inf, length(learning_rates)),
        status = fill("Not run", length(learning_rates)),
    )

    for (i, learning_rate) in enumerate(learning_rates)
        @info "Testing learning rate $i / $(length(learning_rates)): $learning_rate"

        try
            loss = get_cv_log_loss(
                hidden_layers;
                learning_rate = learning_rate,
                epochs = epochs,
                lambda = lambda,
                batch_size = batch_size,
            )

            if isfinite(loss)
                learning_rate_results.cv_log_loss[i] = loss
                learning_rate_results.status[i] = "OK"
            else
                learning_rate_results.status[i] = "Non-finite loss"
            end

        catch err
            learning_rate_results.status[i] = sprint(showerror, err)
        end
    end

    sort!(learning_rate_results, :cv_log_loss)
end
  ╠═╡ =#

# ╔═╡ 31d24c17-120a-4996-9d97-df960ef02a04
#=╠═╡
sort!(learning_rate_results, :cv_log_loss)
  ╠═╡ =#

# ╔═╡ e03990da-c44e-4bc5-bb1c-a183fe838ac4
# ╠═╡ disabled = true
#=╠═╡
begin
    # Sweep regularization strength:
    lambdas = [
        0.0,
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ]

    regularization_results = DataFrame(
        lambda = lambdas,
        cv_log_loss = fill(Inf, length(lambdas)),
        status = fill("Not run", length(lambdas)),
    )

    for (i, lambda) in enumerate(lambdas)
        @info "Testing lambda $i / $(length(lambdas)): $lambda"

        try
            loss = get_cv_log_loss(
                hidden_layers;
                learning_rate = learning_rate,
                epochs = epochs,
                lambda = lambda,
                batch_size = batch_size,
            )

            if isfinite(loss)
                regularization_results.cv_log_loss[i] = loss
                regularization_results.status[i] = "OK"
            else
                regularization_results.status[i] = "Non-finite loss"
            end

        catch err
            regularization_results.status[i] = sprint(showerror, err)
        end
    end

    sort!(regularization_results, :cv_log_loss)
end
  ╠═╡ =#

# ╔═╡ 1b022b5f-635e-446a-9fb5-a7fdac9f1a7e
# ╠═╡ disabled = true
#=╠═╡
begin
    optimizer_names = [
        "Adam(0.001)",
        "RAdam(0.001)",
        "NAdam(0.001)",
        "AMSGrad(0.001)",
        "AdaMax(0.001)",
        "RMSProp(0.001)",
        "Momentum(0.01, 0.9)",
        "Nesterov(0.01, 0.9)",
        "Descent(0.01)",
        "AdaGrad(0.1)",
    ]

    optimizers = Any[
        Optimisers.Adam(0.001),
        Optimisers.RAdam(0.001),
        Optimisers.NAdam(0.001),
        Optimisers.AMSGrad(0.001),
        Optimisers.AdaMax(0.001),
        Optimisers.RMSProp(0.001),
        Optimisers.Momentum(0.01, 0.9),
        Optimisers.Nesterov(0.01, 0.9),
        Optimisers.Descent(0.01),
        Optimisers.AdaGrad(0.1),
    ]

    optimizer_results = DataFrame(
        optimizer = optimizer_names,
        cv_log_loss = fill(Inf, length(optimizers)),
        status = fill("Not run", length(optimizers)),
    )

    for i in eachindex(optimizers)
        @info "Testing optimizer $i / $(length(optimizers)): $(optimizer_names[i])"

        try
            loss = get_cv_log_loss(
                hidden;
                optimiser = optimizers[i],
                epochs = epochs,
                lambda = lambda,
                batch_size = batch_size,
            )

            if isfinite(loss)
                optimizer_results.cv_log_loss[i] = loss
                optimizer_results.status[i] = "OK"
            else
                optimizer_results.status[i] = "Non-finite loss"
            end

        catch err
            optimizer_results.status[i] = sprint(showerror, err)
        end
    end

    sort!(optimizer_results, :cv_log_loss)
end
  ╠═╡ =#

# ╔═╡ d978e03b-5188-41a3-ba0e-1586f049659f
# ╠═╡ disabled = true
#=╠═╡
begin
    epoch_values = [
        10,
        25,
        50,
        100,
        150,
        250,
        400,
        600,
    ]

    epoch_results = DataFrame(
        epochs = epoch_values,
        cv_log_loss = fill(Inf, length(epoch_values)),
        status = fill("Not run", length(epoch_values)),
    )

    for (i, epochs) in enumerate(epoch_values)
        @info "Testing epochs $i / $(length(epoch_values)): $epochs"

        try
            loss = get_cv_log_loss(
                hidden;
                optimiser = optimiser,
                epochs = epochs,
                lambda = lambda,
                batch_size = batch_size,
            )

            if isfinite(loss)
                epoch_results.cv_log_loss[i] = loss
                epoch_results.status[i] = "OK"
            else
                epoch_results.status[i] = "Non-finite loss"
            end

        catch err
            epoch_results.status[i] = sprint(showerror, err)
        end
    end

    sort!(epoch_results, :cv_log_loss)
end
  ╠═╡ =#

# ╔═╡ 96546945-c144-4d79-85dc-780eb6aac97f
# ╠═╡ disabled = true
#=╠═╡
begin
    batch_sizes = [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        4096,
    ]

    batch_results = DataFrame(
        batch_size = batch_sizes,
        cv_log_loss = fill(Inf, length(batch_sizes)),
        status = fill("Not run", length(batch_sizes)),
    )

    for (i, batch_size) in enumerate(batch_sizes)
        @info "Testing batch size $i / $(length(batch_sizes)): $batch_size"

        try
            loss = get_cv_log_loss(
                hidden;
                optimiser = optimiser,
                epochs = epochs,
                lambda = lambda,
                batch_size = batch_size,
            )

            if isfinite(loss)
                batch_results.cv_log_loss[i] = loss
                batch_results.status[i] = "OK"
            else
                batch_results.status[i] = "Non-finite loss"
            end

        catch err
            batch_results.status[i] = sprint(showerror, err)
        end
    end

    sort!(batch_results, :cv_log_loss)

    batch_results
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJFlux = "094fc8d1-fd35-5302-93ea-dabda2abf845"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
MLJMultivariateStatsInterface = "1b6a4a23-ba22-4f51-9698-8599985d3728"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatisticalMeasures = "a19d573c-0a75-4610-95b3-7071388c7541"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.17"
DataFrames = "~1.8.2"
Flux = "~0.16.11"
MLJ = "~0.23.3"
MLJFlux = "~0.6.8"
MLJLinearModels = "~0.10.4"
MLJMultivariateStatsInterface = "~0.5.4"
Optimisers = "~0.4.9"
StatisticalMeasures = "~0.3.7"
StatsBase = "~0.34.13"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.0"
manifest_format = "2.1"
project_hash = "849ce136598fd737edc1feda75ea030ffdb01c79"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "OrderedCollections", "Parsers", "Tables"]
git-tree-sha1 = "829b16815b5594d8d66b3fddb4eca193a2c3db84"
registries = "General"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.6.2"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
registries = "General"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "7f54761502ff149a9d492e4acefe9805898e29b3"
registries = "General"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.2+0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
registries = "General"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "1bbcf97714051f302b3beafab65bcdeb6044c4df"
registries = "General"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.2.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e386db8b4753b42caac75ac81d0a4fe161a68a97"
registries = "General"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.1"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
registries = "General"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitBasis]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "89dc08420d4f593ff30f02611d136b475a5eb43d"
registries = "General"
uuid = "50ba71b6-fa0f-514d-ae9a-0916efc90dcf"
version = "0.9.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "abed1e735dd4152f48c90cf0767e1790e25f332f"
registries = "General"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.17"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9cb23bbb1127eefb022b022481466c0f1127d430"
registries = "General"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.2"

[[deps.CategoricalArrays]]
deps = ["Compat", "DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "20ff1463035a170b25eba2ef9823bb9ad51635e4"
registries = "General"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "1.1.1"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysArrowExt = "Arrow"
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStatsBaseExt = "StatsBase"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "Reexport", "ScientificTypes"]
git-tree-sha1 = "8d8a4e27199f2e961a85ea6fe768c050fb60d6d9"
registries = "General"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.2.2"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
registries = "General"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "3496e2b359c27793d12bec54ec94be7983a6ceb2"
registries = "General"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.2"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "d4101e848e8d3f585d61d244c2fe0c80a70e6b3b"
registries = "General"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.1.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
registries = "General"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
registries = "General"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
registries = "General"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "af523e59de94de05b9b367a2ca4d63233aacd222"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.5"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
registries = "General"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
registries = "General"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "4b6690f3770daf596a0a092203a2e6130d34c53b"
registries = "General"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.22"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
registries = "General"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "478d3450b1cc56521d1d99ad923acf43f8813c6d"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.134"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
registries = "General"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "971d7831cc85f43bc9f51d615a3f7f21270c2f1d"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.21"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FeatureSelection]]
deps = ["MLJModelInterface", "ScientificTypesBase", "Tables"]
git-tree-sha1 = "cf591d055fc9919f2400a290ac19174f58cbdab1"
registries = "General"
uuid = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6"
version = "0.2.6"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
registries = "General"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
registries = "General"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "5031f23e040bf17082e5b52422d77b5e844eefb1"
registries = "General"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.33.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Flux]]
deps = ["ADTypes", "Adapt", "BFloat16s", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "GPUArrays", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "c816ae0963abf4bc2f74f968e4c1f6dbb7902753"
registries = "General"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.11"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxEnzymeExt = "Enzyme"
    FluxFiniteDifferencesExt = "FiniteDifferences"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]
    FluxMooncakeExt = "Mooncake"
    FluxReactantExt = "Reactant"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "1b86cca764a61dcac4fef4c5e16e378e5ed6953c"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.5"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "1ac2813982db52b974c9343124ca61adbf297316"
registries = "General"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "0811627284eba0d19bf64d0d247c63fa1e5c26b6"
registries = "General"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.14"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "EnumX", "PrecompileTools", "Random", "Reseau", "SHA", "URIs", "UUIDs", "Zlib_jll"]
git-tree-sha1 = "b0ff6627f1308bc1691528aac3b70ac3abd1d598"
registries = "General"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "2.6.7"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "88d07a6b68b8fffb13cacd49e23ea73c571859b2"
registries = "General"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.20"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "e663925ebc3d93c1150a7570d114f9ea2f664726"
registries = "General"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.4"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
registries = "General"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "877edc1d2f51adcef0bfacd19464a19e7cfddddb"
registries = "General"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.6"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "a5b87110fa95d711355af44832497745aa93fb52"
registries = "General"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.42"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d4bfee24427f4f441bd9212a107e375c39663aab"
registries = "General"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.13.1"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d77aea19c9a71059a021acd99b0a4343e9661d94"
registries = "General"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.47+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b7970cef8ae1c990ba0c09cd8bdc1145e006632f"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "22.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
registries = "General"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LearnAPI]]
deps = ["Preferences"]
git-tree-sha1 = "a205f0181e25a22089a62a56b9e537b889540dfb"
registries = "General"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "2.0.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.103+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "a09a85e94e681def2446752cddaf9401a4d82b6d"
registries = "General"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.8.2"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7f6be2e4cdaaf558623d93113d6ddade7b916209"
registries = "General"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.4"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "c4ab44fe709638fda6f2c0cbfea2c114932d6c2f"
registries = "General"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.1.0"

    [deps.MLCore.extensions]
    MLCorePythonCallExt = "PythonCall"

    [deps.MLCore.weakdeps]
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "29b00f22be6fd821a214760f0224329f21998a05"
registries = "General"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.10"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "DataAPI", "Distributed", "Distributions", "FeatureSelection", "LinearAlgebra", "MLJBalancing", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJTransforms", "MLJTuning", "MLJWrappers", "OpenML", "Pkg", "ProgressMeter", "Random", "Reexport", "ScientificTypes", "StatisticalMeasures", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "67d1c6c5aeaddb8f90eb2ce32b5949b6d60516db"
registries = "General"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.23.3"

[[deps.MLJBalancing]]
deps = ["MLJBase", "MLJModelInterface", "MLUtils", "OrderedCollections", "Random", "StatsBase"]
git-tree-sha1 = "3886c172c42219166c1137a9f4dc39b863f28823"
registries = "General"
uuid = "45f359ea-796d-4f51-95a5-deb1a414c586"
version = "0.1.6"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "DataAPI", "Dates", "DelimitedFiles", "Distributed", "Distributions", "FillArrays", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Measurements", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "31695dfdd200cda7d06ac552e83f8d65dc55fc2d"
registries = "General"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.14.2"
weakdeps = ["StatisticalMeasures"]

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatisticalMeasuresBase", "StatsBase"]
git-tree-sha1 = "549f8ae7ae51e7c38f84202477508b0371ed16a9"
registries = "General"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.4.5"

[[deps.MLJFlux]]
deps = ["CategoricalArrays", "ColorTypes", "ComputationalResources", "Flux", "MLJModelInterface", "Metalhead", "Optimisers", "ProgressMeter", "Random", "Statistics", "Tables"]
git-tree-sha1 = "f059a2d788652cb9fd714c5be1617e2baab1c573"
registries = "General"
uuid = "094fc8d1-fd35-5302-93ea-dabda2abf845"
version = "0.6.8"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "21506d401828d09a9fefc55a35d3da5c27d73f57"
registries = "General"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.6.5"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "NLSolversBase", "Optim", "Parameters"]
git-tree-sha1 = "e454d1705d1d90cb9077c8a7bc25204da7ad3713"
registries = "General"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.10.4"

[[deps.MLJModelInterface]]
deps = ["InteractiveUtils", "REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c275fae2e693206b4527dd9d2382aa15359ef3ed"
registries = "General"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.12.1"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "fe8ed28e3a52b453a94e317465be591f4fda5622"
registries = "General"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.18.10"

[[deps.MLJMultivariateStatsInterface]]
deps = ["CategoricalDistributions", "Distances", "LinearAlgebra", "MLJModelInterface", "MultivariateStats", "StatsBase"]
git-tree-sha1 = "4a0b6e80590b3c6a95d71d5b8e32cca13ad02c5a"
registries = "General"
uuid = "1b6a4a23-ba22-4f51-9698-8599985d3728"
version = "0.5.4"

[[deps.MLJTransforms]]
deps = ["BitBasis", "CategoricalArrays", "Combinatorics", "Dates", "Distributions", "LinearAlgebra", "MLJModelInterface", "OrderedCollections", "Parameters", "ScientificTypes", "ScientificTypesBase", "Statistics", "StatsBase", "TableOperations", "Tables"]
git-tree-sha1 = "398618135f634bf28d0fcc8a02d9ec3906675a6f"
registries = "General"
uuid = "23777cdb-d90c-4eb0-a694-7c2b83d5c1d6"
version = "0.1.7"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase", "StatisticalMeasuresBase"]
git-tree-sha1 = "f36dc8658fd7c4b69d35bbc826837d07141feacb"
registries = "General"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.8.9"

[[deps.MLJWrappers]]
deps = ["MLJModelInterface"]
git-tree-sha1 = "19371a0b1348766c71d3c9934e0471fe36978e72"
registries = "General"
uuid = "b5d0f7f3-9870-4c70-ba08-cb780c37e63f"
version = "0.1.3"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "CodeTracking", "Compat", "DataAPI", "DelimitedFiles", "Distributed", "InteractiveUtils", "MLCore", "Mmap", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "0a589dc0ada20d30b7e9ad13752cf25361875bf2"
registries = "General"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.13"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf"]
git-tree-sha1 = "cb47f69a1cab9dcec7ff4a5d6e163410d6905866"
registries = "General"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.14.1"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsMakieExt = "Makie"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Metalhead]]
deps = ["Artifacts", "BSON", "ChainRulesCore", "Flux", "Functors", "JLD2", "LazyArtifacts", "MLUtils", "NNlib", "PartialFunctions", "Random", "Statistics"]
git-tree-sha1 = "c022cd34e7bd63374361bbc033cc5d845d94abc5"
registries = "General"
uuid = "dbeba491-748d-5e0e-a39e-b530a07fa0cc"
version = "0.9.6"

    [deps.Metalhead.extensions]
    MetalheadCUDAExt = "CUDA"

    [deps.Metalhead.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "8bcdc90d5c361a73635ad1c19d9c39a52bf3e0a3"
registries = "General"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.5"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "f96d38936d92d610318dec4d3b5ef37b0373c20f"
registries = "General"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "d450844d195714a2d29b38c231193c0297cf90e8"
registries = "General"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.45"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreCUDNNExt = ["EnzymeCore", "CUDA", "cuDNN"]
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibMooncakeCUDAExt = ["Mooncake", "CUDA"]
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "576eb4656529c12e77a46b17c23103dfba9fa570"
registries = "General"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.29"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "9510d7008275fc5b33fc72a73f8fddef0b5430c6"
registries = "General"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.11"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "Downloads", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "d7ce3ebc5df95fc62c060b9f00c4fc534c5af4ac"
registries = "General"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.4"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "81f338ad984b25bc82471db4d235e7b8bf21b8d6"
registries = "General"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.3.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Compat", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "b6a586b581eccc60a181145ffd4382099e32e2df"
registries = "General"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.9"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"
    OptimisersReactantMLDataDevicesExt = ["Reactant", "MLDataDevices"]

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "0ed04c372da78ff5b98e35e3bd4f4ca9931299cc"
registries = "General"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.13.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "ba0dc8a8a67cacac4842631f960c046e4e563675"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.8"

[[deps.PartialFunctions]]
deps = ["MacroTools"]
git-tree-sha1 = "ba0ea009d9f1e38162d016ca54627314b6d8aac8"
registries = "General"
uuid = "570af359-4316-4cb7-8c74-252c00c2016b"
version = "1.2.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
registries = "General"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyPrinting]]
git-tree-sha1 = "0b7f4ad437e31c51cf5b91fb103579b04025170a"
registries = "General"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "eeac4c9337f876c3c32b8da732141953567f57c9"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.1"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
registries = "General"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
registries = "General"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Reseau]]
deps = ["NetworkOptions", "OpenSSL_jll", "PrecompileTools", "Random", "SHA"]
git-tree-sha1 = "aaee74fab545e9af4b21d10c2203c93c636c4ae0"
registries = "General"
uuid = "802f3686-a58f-41ce-bb0c-3c43c75bba36"
version = "1.4.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "c5e8a0a52c0d5d043214de04f11f4d9c2c6ccdb8"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.13"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "696fbda25b0a9dda3e57225d0b457e6fe1565caa"
registries = "General"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.4.0"

[[deps.ScientificTypesBase]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "e785eaa35a0f5518a388f9010e66fda64ea95ede"
registries = "General"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.1.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
registries = "General"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "eec446511ab8c3293dd846c61c15128392fceed5"
registries = "General"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.3"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "88ef0d98204515bc53dd4aaea5f7788882d1b66c"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
registries = "General"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "REPL", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "dc12e34f7d4b298c6349e629b818d93cb9bdec43"
registries = "General"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.3.7"

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

    [deps.StatisticalMeasures.weakdeps]
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "b833adc23afcdaf47029eb2dc7cde8e9d7aceedd"
registries = "General"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.4"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "d3ddfea0085bf97e01fbe8dadf83df7d60a0aa43"
registries = "General"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.6.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
registries = "General"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"
weakdeps = ["Measurements", "StaticArraysCore", "Tables"]

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
registries = "General"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
registries = "General"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "21b39bfb1fab6156b61fbcba4c86c57b6216d2c3"
registries = "General"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.2"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
registries = "General"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "cab29ed70d9355a28b95b1798007ff3a00eefc91"
registries = "General"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.8"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
registries = "General"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a1ec45a8a0adee4d581a37f7e1a0fb401c1cb113"
registries = "General"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.13"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteCUDAExt = "CUDA"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "c6a86c133861234450ab260dee01b42abd604095"
registries = "General"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.8"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.2+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╠═e4bed79a-ac9a-11f1-85c1-f7561a0ee5b3
# ╠═d779c071-1218-45cc-ae66-16e01e12e1e3
# ╠═e670ba1c-19aa-4a93-93b4-f99562de37b4
# ╟─f92de274-9c7f-4ea2-84a5-d18f97044498
# ╟─3426ad4c-a340-46c3-994a-0e30dccef086
# ╠═b078a14b-049e-4398-9c79-85ef7fde2843
# ╠═ef2fb0da-9db3-48bc-a92d-a07e24a6c6a1
# ╠═0bf742cc-4661-4a33-bee0-4905860d2562
# ╠═653ae9d0-bca6-4778-a8d3-9d07dbf143c2
# ╟─e346bbf0-066e-4de6-bcdd-6c2732756d17
# ╟─5a3a0107-bed6-440a-9805-5a2b9b483b72
# ╠═af1af019-a784-4144-a632-e66fca8b1c3b
# ╟─6f6e6ca0-f018-4ddd-b8e9-61dd05f00560
# ╟─31d5dc1e-5f96-4841-9ce1-1877ca215454
# ╠═2a218ae9-897e-4f89-8895-87826b38d47a
# ╠═741c414a-b1e9-4e26-aad0-359e49588d8b
# ╠═139075ff-92d9-480a-b53b-4bca487b100f
# ╟─b268474e-4bcb-4ae8-ae6b-cd1eeb8210da
# ╠═3b281328-7809-46a7-8293-02499727e4ab
# ╠═645d1d8a-f76e-4857-a0e7-66f75fcf4e23
# ╠═fdc35e29-2bf6-4348-a03f-de9eaf0769f6
# ╠═f0b871d9-5d18-43d5-8d27-741ac9a06363
# ╟─d05b79c4-0235-45a1-933d-c8147e7b0e0b
# ╠═3b280832-9ad9-437c-becd-e79eb3a51e91
# ╠═a12d3efa-e011-4939-bc03-ee75cf72d5d0
# ╠═5016e7fd-380b-4a38-8e7a-7e63cbb1518a
# ╠═618282f9-e415-4a17-b8f0-92e92e6a7668
# ╠═39d6f28c-405a-4eb7-95dd-dd01c8de7f25
# ╠═f92f7120-79fd-4c0b-8632-e29cb161788a
# ╠═8c92dffd-5cb4-46f5-b07d-74ec27265384
# ╟─7b69149a-4890-41c0-94ed-2fb9a6a571f3
# ╠═30d268c8-5e91-41f0-a9f0-3b58ed5ce3fe
# ╠═d6699ec3-50db-47f7-be0b-303b0f9c147b
# ╠═56b92223-6559-4fb3-aee4-f5623c29bad2
# ╠═6a054746-79b0-4f1f-98a0-05da22b2b6cc
# ╠═62a5eea0-e180-43ed-9f5b-1a190a625e48
# ╠═7bba9815-2ec7-477f-88b9-30e8d785ad62
# ╠═d7c76249-16a5-4116-9f00-7b2c8a5ed474
# ╟─f8d5ee66-ddf4-4c5a-ad92-aea4da10fcce
# ╟─59bf7ef6-c3bf-4cf5-96ef-36335f333327
# ╠═af2f63ab-3172-4e1f-950e-8449197da08e
# ╠═3989ad12-80ba-4244-9e13-adecbb72c956
# ╠═0bd9b3c8-6747-46e2-8981-d60a49f64262
# ╠═ae81bb04-0644-478d-80ff-7cf7ed4691db
# ╠═e9f6b1c5-35d6-4470-9360-f0e546465009
# ╠═edd27b43-8204-444a-82a8-43ec5170b5f4
# ╠═fe3e29de-2765-45f6-9c6f-d67d90756d4d
# ╠═6fb87d58-690b-489a-bde9-8e86f717c19a
# ╠═86b015ac-f3cd-45e6-bbe0-6614522928f2
# ╟─8c8eefa7-f702-4185-bf11-02722c577fed
# ╠═1a443f32-e338-45e6-b6ce-39bcccb3a8dd
# ╠═424c1659-f241-4eab-93fa-02be380bbb2c
# ╠═e32ee31d-8c4b-416e-a168-9d93fd1e3bd4
# ╠═f341be50-37f9-45b2-9597-0db46997e6a7
# ╠═316d58ec-0d6d-4e89-ad52-bf90693437fb
# ╠═e47deada-667c-42d6-b7f6-3667bb33fd00
# ╠═b7f85f88-1f37-4e8a-b65f-c42ffd2f8a1d
# ╟─3cde7233-438d-4cb4-aeac-ef6cadb7e22d
# ╟─49ba444f-c820-469b-98e3-8d11f61b5229
# ╟─d3b4eed4-e5f1-4cdf-b8ca-88c66c9edb43
# ╟─920b4200-3b99-48e9-906e-d637bfdacb08
# ╟─31d24c17-120a-4996-9d97-df960ef02a04
# ╟─e03990da-c44e-4bc5-bb1c-a183fe838ac4
# ╟─1b022b5f-635e-446a-9fb5-a7fdac9f1a7e
# ╟─d978e03b-5188-41a3-ba0e-1586f049659f
# ╟─96546945-c144-4d79-85dc-780eb6aac97f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
