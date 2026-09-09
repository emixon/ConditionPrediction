using ConditionPrediction
using StatsPlots
using Statistics
path = joinpath( @__DIR__, "..", "data", "observations.csv")

df = load_data(path)

age_hist = histogram(df.NormalizedAge, bins=500, title="Normalized Age")

ci_hist = histogram(df.PredictedCI, bins=500, title="Predicted CI")

score_hist = histogram(df.ActualScore, bins=500, title="Assessed Score")

crv_hist = histogram(df.CRV, bins=500, title="Replacement Value")

combined = plot( age_hist, ci_hist, score_hist, crv_hist; layout=(2, 2), size=(2000, 1600))

display(combined)