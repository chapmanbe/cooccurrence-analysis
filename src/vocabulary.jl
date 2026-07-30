# ──────────────────────────────────────────────────────────────────────────────
# Presentation vocabulary
#
# The math and the data model are domain-neutral, but figure titles and axis
# labels are read by humans in a specific domain. A cancer-epidemiology figure
# labelled "Item Prevalence (% of records with each item)" is not wrong so much
# as useless.
#
# This is the one place where an application's nouns enter the package, and it
# affects **labels only** — never a computation, a column name, or a return
# value. Callers set it once at startup:
#
#     set_vocabulary!(item="Cancer Site", items="cancer sites",
#                     record="Patient",   records="patients",
#                     group="Sex")
#
# Deliberately global rather than a keyword on every plot function: there is one
# vocabulary per application, and threading it through fourteen signatures would
# put the burden on every call site to get every figure right.
# ──────────────────────────────────────────────────────────────────────────────

"""
    Vocabulary

Human-readable nouns used in figure titles and axis labels.

- `item` / `items` — singular Title Case, plural lower case ("Item" / "items")
- `record` / `records` — singular Title Case, plural lower case ("Record" / "records")
- `group` — singular Title Case name of the stratification variable ("Group")
"""
mutable struct Vocabulary
    item::String
    items::String
    record::String
    records::String
    group::String
end

const DEFAULT_VOCABULARY = (item="Item", items="items",
                            record="Record", records="records", group="Group")

"The active presentation vocabulary. Set it with [`set_vocabulary!`](@ref)."
const VOCAB = Vocabulary(DEFAULT_VOCABULARY...)

"""
    set_vocabulary!(; item, items, record, records, group) -> Vocabulary

Set the nouns used in figure titles and axis labels. Every keyword is optional
and defaults to its current value, so partial updates work. Affects labels only.

```julia
set_vocabulary!(item="Cancer Site", items="cancer sites",
                record="Patient", records="patients", group="Sex")
```
"""
function set_vocabulary!(; item::AbstractString=VOCAB.item,
                           items::AbstractString=VOCAB.items,
                           record::AbstractString=VOCAB.record,
                           records::AbstractString=VOCAB.records,
                           group::AbstractString=VOCAB.group)
    VOCAB.item, VOCAB.items = item, items
    VOCAB.record, VOCAB.records = record, records
    VOCAB.group = group
    return VOCAB
end

"""
    reset_vocabulary!() -> Vocabulary

Restore the domain-neutral defaults ("Item" / "Record" / "Group").
"""
reset_vocabulary!() = set_vocabulary!(; DEFAULT_VOCABULARY...)

"Lower-case singular item noun, for mid-sentence use (\"P(cancer site | class)\")."
item_lc() = lowercase(VOCAB.item)

"Lower-case singular record noun, for mid-sentence use."
record_lc() = lowercase(VOCAB.record)
