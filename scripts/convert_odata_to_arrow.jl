# Convert the three raw tabular CSV subsets in odata/ to a single Arrow file.
#
# Usage:
#   julia --project=clustering clustering/scripts/convert_odata_to_arrow.jl
#
# Output: odata/real_data.arrow  (gitignored alongside the source CSVs)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, DataFrames, Arrow

const ODATA_DIR  = joinpath(@__DIR__, "..", "..", "odata")
const OUTPUT     = joinpath(ODATA_DIR, "real_data.arrow")

const CSV_FILES = [
    "cluster_eventlevel_subset1_082725(in).csv",
    "cluster_eventlevel_subset2_082725(in).csv",
    "cluster_eventlevel_subset3_082725(in).csv",
]

function main()
    parts = DataFrame[]

    for fname in CSV_FILES
        path = joinpath(ODATA_DIR, fname)
        println("Reading $fname …")
        df = CSV.read(path, DataFrame;
            missingstring = "",
            normalizenames = true)   # strips BOM, replaces spaces/special chars

        println("  $(nrow(df)) rows, $(ncol(df)) columns")
        push!(parts, df)
    end

    println("\nConcatenating …")
    combined = vcat(parts...; cols = :union)
    println("  Total rows: $(nrow(combined))")
    println("  Columns:    $(join(names(combined), ", "))")

    println("\nWriting Arrow to $(relpath(OUTPUT)) …")
    Arrow.write(OUTPUT, combined)
    println("Done — $(round(filesize(OUTPUT) / 1024^2, digits=1)) MB")
end

main()
