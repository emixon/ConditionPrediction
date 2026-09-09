function load_data(
    path::AbstractString = joinpath(
        @__DIR__, "..", "data", "sample_data.csv"
    )
)
    return CSV.read(path, DataFrame)
end