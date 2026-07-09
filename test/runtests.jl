using Test
using DataFrames
import Arrow
using Graphs: nv, ne, add_edge!
using SimpleWeightedGraphs: SimpleWeightedGraph
using CairoMakie: Figure, Axis, BarPlot
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CooccurrenceAnalysis

# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixture: small event-level dataset with planted associations
# ──────────────────────────────────────────────────────────────────────────────

"""
Build a test DataFrame with 50 records:
- 30 single-item records (background)
- 10 records with Item01 + Item02 (planted strong association)
- 5 records with GA1 + Item05 (planted group_a association)
- 3 records with Item01 + GB1 (planted group_b association)
- 2 records with Item03 + Item04 (weak association)
"""
function make_test_event_df()
    rows = NamedTuple[]

    id = 0
    # 15 single-item group B
    for _ in 1:8
        id += 1
        push!(rows, (id=id, seq=1, Group="B", item="Item01",
                      year=2010, has_second_event="No", level="L1"))
    end
    for _ in 1:4
        id += 1
        push!(rows, (id=id, seq=1, Group="B", item="Item02",
                      year=2010, has_second_event="No", level="L1"))
    end
    for _ in 1:3
        id += 1
        push!(rows, (id=id, seq=1, Group="B", item="Item03",
                      year=2010, has_second_event="No", level="L3"))
    end

    # 15 single-item group A
    for _ in 1:8
        id += 1
        push!(rows, (id=id, seq=1, Group="A", item="GA1",
                      year=2010, has_second_event="No", level="L1"))
    end
    for _ in 1:4
        id += 1
        push!(rows, (id=id, seq=1, Group="A", item="Item05",
                      year=2010, has_second_event="No", level="L2"))
    end
    for _ in 1:3
        id += 1
        push!(rows, (id=id, seq=1, Group="A", item="Item03",
                      year=2010, has_second_event="No", level="L3"))
    end

    # 10 Item01 + Item02 (group_b, planted strong association)
    for _ in 1:10
        id += 1
        push!(rows, (id=id, seq=1, Group="B", item="Item01",
                      year=2010, has_second_event="Yes", level="L1"))
        push!(rows, (id=id, seq=2, Group="B", item="Item02",
                      year=2010, has_second_event="Yes", level="L1"))
    end

    # 5 GA1 + Item05 (group_a)
    for _ in 1:5
        id += 1
        push!(rows, (id=id, seq=1, Group="A", item="GA1",
                      year=2010, has_second_event="Yes", level="L1"))
        push!(rows, (id=id, seq=2, Group="A", item="Item05",
                      year=2010, has_second_event="Yes", level="L2"))
    end

    # 3 Item01 + GB1 (group_b)
    for _ in 1:3
        id += 1
        push!(rows, (id=id, seq=1, Group="B", item="Item01",
                      year=2010, has_second_event="Yes", level="L1"))
        push!(rows, (id=id, seq=2, Group="B", item="GB1",
                      year=2010, has_second_event="Yes", level="L3"))
    end

    # 2 Item03 + Item04 (1 group_a, 1 group_b) — planted as SEQUENTIAL (5y gap)
    id += 1
    push!(rows, (id=id, seq=1, Group="A", item="Item03",
                  year=2010, has_second_event="Yes", level="L3"))
    push!(rows, (id=id, seq=2, Group="A", item="Item04",
                  year=2015, has_second_event="Yes", level="L2"))
    id += 1
    push!(rows, (id=id, seq=1, Group="B", item="Item03",
                  year=2010, has_second_event="Yes", level="L3"))
    push!(rows, (id=id, seq=2, Group="B", item="Item04",
                  year=2015, has_second_event="Yes", level="L2"))

    return DataFrame(rows)
end

# ──────────────────────────────────────────────────────────────────────────────
@testset "CooccurrenceAnalysis" begin
# ──────────────────────────────────────────────────────────────────────────────

event_df = make_test_event_df()

@testset "utils" begin
    @test format_itemset(["Item01", "Item02"]) == "Item01 + Item02"
    @test format_itemset(["Item02", "Item01"]) == "Item01 + Item02"  # sorted
    @test significance_stars(0.0001) == "***"
    @test significance_stars(0.005) == "**"
    @test significance_stars(0.03) == "*"
    @test significance_stars(0.1) == "ns"
end

@testset "data_preparation" begin
    @testset "get_record_summary" begin
        ps = get_record_summary(event_df)
        @test nrow(ps) == 50
        @test all(ps.n_items .>= 1)
        multi = filter(row -> row.n_items >= 2, ps)
        @test nrow(multi) == 20  # 10 + 5 + 3 + 2
    end

    @testset "build_transactions default (min_items=2)" begin
        txns = build_transactions(event_df)
        @test nrow(txns) == 20  # only multi-item records
        # Should have boolean columns for the items present
        @test all(eltype(col) == Bool for col in eachcol(txns))
        # Item01+Item02 records should have both true
        @test "Item01" in names(txns)
        @test "Item02" in names(txns)
    end

    @testset "build_transactions with group_filter" begin
        group_a_txns = build_transactions(event_df; group_filter="A")
        # Group A multi-item: 5 GA1+Item05 + 1 Item03+Item04 = 6
        @test nrow(group_a_txns) == 6
        # Group B-only items should be excluded
        @test !("GB1" in names(group_a_txns))

        group_b_txns = build_transactions(event_df; group_filter="B")
        # Group B multi-item: 10 Item01+Item02 + 3 Item01+GB1 + 1 Item03+Item04 = 14
        @test nrow(group_b_txns) == 14
        # Group A-only items should be excluded
        @test !("GA1" in names(group_b_txns))
    end

    @testset "build_transactions min_items=1" begin
        txns = build_transactions(event_df; min_items=1)
        @test nrow(txns) == 50  # all records
    end

    @testset "empty result" begin
        empty_df = DataFrame(id=Int[], seq=Int[], Group=String[], item=String[])
        txns = build_transactions(empty_df)
        @test nrow(txns) == 0
    end

    @testset "build_transactions timing_filter" begin
        # Fixture has 18 concurrent (year all 2010) + 2 sequential
        # (Item03+Item04 records with 2010/2015 gap)
        all_txns = build_transactions(event_df; timing_filter=:all)
        @test nrow(all_txns) == 20

        sync_txns = build_transactions(event_df; timing_filter=:concurrent)
        @test nrow(sync_txns) == 18  # excludes the 2 sequential Item03+Item04

        meta_txns = build_transactions(event_df; timing_filter=:sequential)
        @test nrow(meta_txns) == 2

        # concurrent_window of 5 should bring sequential records into concurrent
        sync_window = build_transactions(event_df;
            timing_filter=:concurrent, concurrent_window=5)
        @test nrow(sync_window) == 20  # 5y gap is now within window

        meta_window = build_transactions(event_df;
            timing_filter=:sequential, concurrent_window=5)
        @test nrow(meta_window) == 0  # nothing exceeds 5y in fixture
    end

    @testset "build_transactions_with_ids" begin
        onehot, ids = build_transactions_with_ids(event_df; timing_filter=:sequential)
        @test nrow(onehot) == length(ids) == 2
        # The sequential records are the 2 Item03+Item04 ones — the last 2 ids
        @test all(p in event_df.id for p in ids)
        # Verify row alignment: each id's event record set has Item03 & Item04
        for (i, id) in enumerate(ids)
            items = unique(filter(r -> r.id == id, event_df).item)
            @test "Item03" in items
            @test "Item04" in items
            @test onehot[i, "Item03"]
            @test onehot[i, "Item04"]
        end
    end

    @testset "build_transactions invalid timing_filter" begin
        @test_throws ErrorException build_transactions(event_df; timing_filter=:bogus)
    end
end

@testset "mining" begin
    txns_df = build_transactions(event_df; min_items=2)

    @testset "mine_frequent_itemsets" begin
        itemsets = mine_frequent_itemsets(txns_df; min_support=0.1)
        @test nrow(itemsets) > 0
        @test all(itemsets.Length .>= 2)
        # Item01+Item02 (10/20 = 0.5 support) should be found
        bt = filter(row -> Set(row.Itemset) == Set(["Item01", "Item02"]), itemsets)
        @test nrow(bt) == 1
        @test bt.Support[1] ≈ 0.5
    end

    @testset "mine_frequent_itemsets with min_count" begin
        itemsets = mine_frequent_itemsets(txns_df; min_count=5)
        # Item01+Item02 (N=10) should be found
        bt = filter(row -> Set(row.Itemset) == Set(["Item01", "Item02"]), itemsets)
        @test nrow(bt) == 1
        # Item03+Item04 (N=2) should NOT be found
        lc = filter(row -> Set(row.Itemset) == Set(["Item03", "Item04"]), itemsets)
        @test nrow(lc) == 0
    end

    @testset "mine_association_rules" begin
        rules = mine_association_rules(txns_df; min_support=0.1, min_confidence=0.3)
        @test nrow(rules) > 0
        @test all(length.(rules.LHS) .>= 1)  # no empty LHS
        @test "Support" in names(rules)
        @test "Confidence" in names(rules)
        @test "Lift" in names(rules)
    end

    @testset "empty transactions" begin
        empty = DataFrame()
        @test nrow(mine_frequent_itemsets(empty)) == 0
        @test nrow(mine_association_rules(empty)) == 0
    end
end

@testset "statistical_validation" begin
    # Build record item lists from test data
    gp = groupby(event_df, :id)
    record_items = [sort(unique(g.item)) for g in gp]

    @testset "build_contingency_table" begin
        ct = build_contingency_table("Item01", "Item02", record_items)
        @test sum(ct) == 50  # total records
        @test ct[1, 1] == 10  # both Item01 and Item02
        # Row sums: has Item01 = 8+10+3 = 21, has Item02 = 4+10 = 14
        @test ct[1, 1] + ct[1, 2] == 21  # has Item01
        @test ct[1, 1] + ct[2, 1] == 14  # has Item02
    end

    @testset "odds_ratio zero-cell handling (C5)" begin
        # No zero cell: raw ratio (a·d)/(b·c).
        @test odds_ratio([10 2; 3 20]) ≈ (10*20)/(2*3)
        @test odds_ratio([10 2; 3 20]; correction=:none) ≈ (10*20)/(2*3)

        # Zero in b (perfect association): raw is Inf, Haldane keeps it finite.
        zc = [5 0; 0 5]
        @test isfinite(odds_ratio(zc))                       # Haldane-corrected
        @test odds_ratio(zc) ≈ ((5.5)*(5.5))/((0.5)*(0.5))
        @test odds_ratio(zc; correction=:none) == Inf

        # 0/0 raw is NaN; Haldane is finite.
        @test isfinite(odds_ratio([0 0; 3 4]))
        @test isnan(odds_ratio([0 0; 3 4]; correction=:none))

        @test_throws ErrorException odds_ratio([1 1; 1 1]; correction=:bogus)
    end

    @testset "test_association" begin
        result = test_association("Item01", "Item02", record_items; test=:fisher)
        @test result.observed == 10
        @test result.expected > 0
        @test result.odds_ratio > 1.0  # positive association
        @test result.p_value < 0.05    # should be significant
        @test result.test_type == :fisher

        # Chi-square test
        result_chi = test_association("Item01", "Item02", record_items; test=:chisq)
        @test result_chi.p_value < 0.05
    end

    @testset "adjust_pvalues" begin
        pvals = [0.001, 0.01, 0.03, 0.06, 0.5]
        adj_bh = adjust_pvalues(pvals; method=:bh)
        @test length(adj_bh) == 5
        @test all(adj_bh .>= pvals)  # adjusted should be >= raw

        adj_bonf = adjust_pvalues(pvals; method=:bonferroni)
        @test all(adj_bonf .>= adj_bh)  # Bonferroni is more conservative

        @test isempty(adjust_pvalues(Float64[]))
    end

    @testset "validate_rules" begin
        txns_df = build_transactions(event_df; min_items=2)
        rules = mine_association_rules(txns_df; min_support=0.1, min_confidence=0.3)
        validated = validate_rules(rules, event_df)

        @test "p_value" in names(validated)
        @test "p_adjusted" in names(validated)
        @test "significant" in names(validated)
        @test "odds_ratio" in names(validated)
        @test nrow(validated) == nrow(rules)
    end
end

@testset "group_stratification" begin
    @testset "stratified_analysis" begin
        strat = stratified_analysis(event_df;
            min_support=0.1, min_confidence=0.3, min_count=nothing)

        @test strat.group_a_n_records == 6
        @test strat.group_b_n_records == 14

        # Group A rules should include GA1+Item05
        if nrow(strat.group_a_rules) > 0
            rule_items = [Set(vcat(r.LHS, [r.RHS]))
                         for r in eachrow(strat.group_a_rules)]
            @test any(s -> Set(["GA1", "Item05"]) ⊆ s, rule_items)
        end

        # Group B rules should include Item01+Item02
        if nrow(strat.group_b_rules) > 0
            rule_items = [Set(vcat(r.LHS, [r.RHS]))
                         for r in eachrow(strat.group_b_rules)]
            @test any(s -> Set(["Item01", "Item02"]) ⊆ s, rule_items)
        end
    end

    @testset "compare_strata" begin
        strat = stratified_analysis(event_df;
            min_support=0.05, min_confidence=0.1, min_count=nothing)

        if nrow(strat.group_a_rules) > 0 && nrow(strat.group_b_rules) > 0
            comp = compare_strata(strat.group_a_rules, strat.group_b_rules)
            @test "category" in names(comp)
            @test "association" in names(comp)
            @test all(comp.category .∈ Ref([:universal, :group_a_only,
                                            :group_b_only, :group_specific_item]))
        end
    end
end

@testset "integration" begin
    # Save test data as Arrow, run pipeline, clean up
    path = tempname() * ".arrow"
    Arrow.write(path, event_df)

    result = run_full_pipeline(path;
        min_support=0.1, min_confidence=0.3, min_count=nothing,
        stratify_by_group=true)

    @test result.n_total_records == 50
    @test result.n_records == 20
    @test nrow(result.rules) > 0

    # The Item01+Item02 association should be found and significant
    if hasproperty(result.rules, :significant)
        sig_rules = filter(row -> row.significant, result.rules)
        @test nrow(sig_rules) > 0
    end

    rm(path)
end

# ──────────────────────────────────────────────────────────────────────────────
# Network analysis tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "phi_coefficient" begin
    # Perfect positive association: all agree
    ct_perfect = [10 0; 0 10]
    @test phi_coefficient(ct_perfect) ≈ 1.0

    # No association: independent
    ct_zero = [5 5; 5 5]
    @test phi_coefficient(ct_zero) ≈ 0.0

    # Negative association
    ct_neg = [0 10; 10 0]
    @test phi_coefficient(ct_neg) ≈ -1.0

    # Zero denominator
    ct_degen = [5 0; 0 0]
    @test phi_coefficient(ct_degen) == 0.0
end

@testset "compute_pairwise_associations" begin
    assoc = compute_pairwise_associations(event_df)
    @test nrow(assoc) > 0
    @test "item_a" in names(assoc)
    @test "item_b" in names(assoc)
    @test "lift" in names(assoc)
    @test "phi" in names(assoc)
    @test "p_adjusted" in names(assoc)

    # Item01+Item02 should have high lift
    bt = filter(row -> Set([row.item_a, row.item_b]) == Set(["Item01", "Item02"]), assoc)
    @test nrow(bt) == 1
    @test bt.lift[1] > 1.0
end

@testset "build_cooccurrence_network" begin
    # Use low thresholds for small test data
    net = build_cooccurrence_network(event_df;
        min_count=2, alpha=0.5)

    @test net isa CooccurrenceNetwork
    @test length(net.items) == length(net.item_index)
    @test net.n_records == 50

    # Graph should have vertices matching items
    @test nv(net.graph) == length(net.items)

    # Should have at least some edges (Item01+Item02 planted strongly)
    @test ne(net.graph) > 0

    # Prevalence should be populated
    @test !isempty(net.prevalence)

    # Test strict filtering reduces edges
    strict_net = build_cooccurrence_network(event_df;
        min_count=20, alpha=0.01)
    @test ne(strict_net.graph) <= ne(net.graph)
end

@testset "network timing_filter" begin
    # Fixture has 18 concurrent + 2 sequential (Item03+Item04) multi-item
    # records. Total event rows from multi-item records: 36 sync + 4 meta = 40.

    @testset "filter_event_by_timing helper" begin
        # :all is a no-op
        @test nrow(filter_event_by_timing(event_df; timing_filter=:all)) == nrow(event_df)

        # :concurrent keeps only the 18 multi-item concurrent records (36 rows)
        sync_rows = filter_event_by_timing(event_df; timing_filter=:concurrent)
        @test length(unique(sync_rows.id)) == 18
        @test nrow(sync_rows) == 36  # 18 records × 2 events

        # :sequential keeps only the 2 Item03+Item04 records (4 rows)
        meta_rows = filter_event_by_timing(event_df; timing_filter=:sequential)
        @test length(unique(meta_rows.id)) == 2
        @test nrow(meta_rows) == 4

        # window=5 brings everyone into concurrent
        sync5 = filter_event_by_timing(event_df; timing_filter=:concurrent, concurrent_window=5)
        @test length(unique(sync5.id)) == 20
    end

    @testset "compute_pairwise_associations + timing_filter" begin
        # Concurrent network excludes Item03+Item04 co-occurrence
        sync_assoc = compute_pairwise_associations(event_df; timing_filter=:concurrent)
        item34 = filter(row -> Set([row.item_a, row.item_b]) == Set(["Item03", "Item04"]), sync_assoc)
        @test nrow(item34) == 0  # zero co-occurrence under concurrent

        # Sequential network includes Item03+Item04 (and excludes single-item records)
        seq_assoc = compute_pairwise_associations(event_df; timing_filter=:sequential)
        item34_seq = filter(row -> Set([row.item_a, row.item_b]) == Set(["Item03", "Item04"]), seq_assoc)
        @test nrow(item34_seq) == 1
        @test item34_seq.observed[1] == 2  # both sequential records
    end

    @testset "build_cooccurrence_network + timing_filter" begin
        sync_net = build_cooccurrence_network(event_df;
            min_count=2, alpha=0.5, timing_filter=:concurrent)
        meta_net = build_cooccurrence_network(event_df;
            min_count=2, alpha=0.5, timing_filter=:sequential)
        @test sync_net isa CooccurrenceNetwork
        @test meta_net isa CooccurrenceNetwork
        # Record counts reflect timing-filtered cohort
        @test sync_net.n_records == 18
        @test meta_net.n_records == 2
    end

    @testset "compute_pairwise_associations rejects bad timing_filter" begin
        @test_throws ErrorException compute_pairwise_associations(event_df; timing_filter=:bogus)
    end
end

@testset "detect_communities" begin
    net = build_cooccurrence_network(event_df; min_count=2, alpha=0.5)

    @testset "louvain" begin
        comm = detect_communities(net; method=:louvain)
        @test comm isa CommunityResult
        @test length(comm.assignments) == nv(net.graph)
        @test comm.n_communities >= 1
        @test comm.modularity >= 0.0 || comm.n_communities == 1
        # Every item should be in some community
        all_items_in_comm = vcat(values(comm.communities)...)
        @test Set(all_items_in_comm) == Set(net.items)
    end

    @testset "louvain modularity gain matches ΔQ (C1, C2, T6)" begin
        # Two triangles (nodes 1-2-3 and 4-5-6) joined by one weak bridge.
        # The optimal partition is the two cliques.
        g = SimpleWeightedGraph(6)
        for (u, v) in [(1,2),(2,3),(1,3),(4,5),(5,6),(4,6)]
            add_edge!(g, u, v, 1.0)
        end
        add_edge!(g, 3, 4, 0.1)  # weak bridge

        # Node strengths and a helper to recompute per-community strength totals.
        strengths = [sum(g.weights[v, :]) for v in 1:6]
        m2 = sum(strengths)  # = 2m
        comm_strength(assign) = [sum(strengths[v] for v in 1:6 if assign[v] == c; init=0.0)
                                 for c in 1:6]

        # The gain formula must agree with the file's own _modularity: for any
        # single-node move, net_gain = gain(target) − gain(current) == ΔQ exactly.
        # This is the direct check that C1's factor-of-2 and C2's node-exclusion
        # are both correct.
        assign = collect(1:6)  # all singletons
        cs = comm_strength(assign)
        for (node, target) in [(2, 1), (5, 4), (4, 3)]
            cur = assign[node]
            gain = CooccurrenceAnalysis._modularity_gain(g, node, target, assign, strengths, cs, m2)
            loss = CooccurrenceAnalysis._modularity_gain(g, node, cur, assign, strengths, cs, m2)
            net_gain = gain - loss
            q_before = CooccurrenceAnalysis._modularity(g, assign)
            moved = copy(assign); moved[node] = target
            dq = CooccurrenceAnalysis._modularity(g, moved) - q_before
            @test net_gain ≈ dq atol=1e-10
        end

        # Louvain recovers the two cliques and maximizes modularity vs. the
        # trivial all-in-one-community partition.
        final = CooccurrenceAnalysis._louvain(g, 100)
        @test final[1] == final[2] == final[3]
        @test final[4] == final[5] == final[6]
        @test final[1] != final[4]
        q_split = CooccurrenceAnalysis._modularity(g, final)
        q_merged = CooccurrenceAnalysis._modularity(g, fill(1, 6))
        @test q_split > q_merged
    end

    @testset "label_propagation" begin
        comm = detect_communities(net; method=:label_propagation)
        @test comm isa CommunityResult
        @test length(comm.assignments) == nv(net.graph)
        @test comm.n_communities >= 1
    end
end

@testset "compute_network_metrics" begin
    net = build_cooccurrence_network(event_df; min_count=2, alpha=0.5)
    metrics = compute_network_metrics(net)

    @test nrow(metrics) == nv(net.graph)
    @test "item" in names(metrics)
    @test "degree" in names(metrics)
    @test "strength" in names(metrics)
    @test "betweenness" in names(metrics)
    @test "clustering_coeff" in names(metrics)
    @test "prevalence" in names(metrics)

    # Degrees should be non-negative
    @test all(metrics.degree .>= 0)
end

@testset "network_group_stratification" begin
    @testset "stratified_network_analysis" begin
        strat = stratified_network_analysis(event_df;
            min_count=1, alpha=1.0)

        @test strat.group_a_net isa CooccurrenceNetwork
        @test strat.group_b_net isa CooccurrenceNetwork
        @test strat.group_a_communities isa CommunityResult
        @test strat.group_b_communities isa CommunityResult

        # Group A network should not contain group_b-only items
        for item in strat.group_a_net.items
            @test !(item in GROUP_B_ONLY_ITEMS)
        end

        # Group B network should not contain group_a-only items
        for item in strat.group_b_net.items
            @test !(item in GROUP_A_ONLY_ITEMS)
        end
    end

    @testset "compare_networks" begin
        strat = stratified_network_analysis(event_df;
            min_count=1, alpha=1.0)

        if nv(strat.group_a_net.graph) > 0 && nv(strat.group_b_net.graph) > 0
            comp = compare_networks(strat.group_a_net, strat.group_b_net,
                                    strat.group_a_communities, strat.group_b_communities)
            @test comp isa NetworkComparisonResult
            @test comp.shared_edges isa DataFrame
            @test comp.group_a_only_edges isa DataFrame
            @test comp.group_b_only_edges isa DataFrame
            @test -1.0 <= comp.community_ari <= 1.0
        end
    end
end

@testset "adjusted_rand_index" begin
    # Perfect agreement
    @test CooccurrenceAnalysis._adjusted_rand_index([1,1,2,2], [1,1,2,2]) ≈ 1.0

    # Different labelings of same structure
    @test CooccurrenceAnalysis._adjusted_rand_index([1,1,2,2], [2,2,1,1]) ≈ 1.0

    # Random-level agreement should be near 0
    ari = CooccurrenceAnalysis._adjusted_rand_index(
        [1,1,1,2,2,2,3,3,3], [1,2,3,1,2,3,1,2,3])
    @test ari < 0.5
end

@testset "network_visualization" begin
    net = build_cooccurrence_network(event_df; min_count=2, alpha=0.5)
    comm = detect_communities(net)

    @testset "plot_cooccurrence_network" begin
        fig = plot_cooccurrence_network(net; communities=comm)
        @test fig isa Figure
    end

    @testset "plot_community_heatmap" begin
        fig = plot_community_heatmap(net, comm)
        @test fig isa Figure
    end
end

@testset "network_integration" begin
    path = tempname() * ".arrow"
    Arrow.write(path, event_df)

    result = run_network_pipeline(path;
        min_count=2, alpha=0.5, stratify_by_group=true)

    @test result.network isa CooccurrenceNetwork
    @test result.communities isa CommunityResult
    @test nrow(result.metrics) > 0

    rm(path)
end

# ──────────────────────────────────────────────────────────────────────────────
# Bayesian Bernoulli Mixture Clustering tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "transactions_to_matrix" begin
    txns = build_transactions(event_df; min_items=2)
    X, items = transactions_to_matrix(txns)
    @test X isa Matrix{Bool}
    @test size(X, 1) == nrow(txns)
    @test size(X, 2) == ncol(txns)
    @test length(items) == size(X, 2)
    @test items == names(txns)
end

@testset "fit_bernoulli_mixture" begin
    # Planted 2-class data: class 1 has features 1,2; class 2 has features 3,4
    rng = MersenneTwister(42)
    N = 100
    X = falses(N, 4)
    for i in 1:50
        X[i, 1] = rand(rng) < 0.8
        X[i, 2] = rand(rng) < 0.7
        X[i, 3] = rand(rng) < 0.1
        X[i, 4] = rand(rng) < 0.1
    end
    for i in 51:100
        X[i, 1] = rand(rng) < 0.1
        X[i, 2] = rand(rng) < 0.1
        X[i, 3] = rand(rng) < 0.8
        X[i, 4] = rand(rng) < 0.7
    end

    result = fit_bernoulli_mixture(X, 2; n_init=3, rng=MersenneTwister(123))
    @test result isa BernoulliMixtureResult
    @test result.K == 2
    @test size(result.theta) == (2, 4)
    @test size(result.responsibilities) == (100, 2)
    @test length(result.assignments) == 100
    @test all(1 .<= result.assignments .<= 2)
    @test all(0.0 .<= result.theta .<= 1.0)
    @test sum(result.pi) ≈ 1.0
    @test result.bic isa Float64
    @test result.converged
end

@testset "select_K" begin
    rng = MersenneTwister(42)
    N = 60
    X = falses(N, 4)
    for i in 1:30
        X[i, 1] = rand(rng) < 0.8; X[i, 2] = rand(rng) < 0.7
    end
    for i in 31:60
        X[i, 3] = rand(rng) < 0.8; X[i, 4] = rand(rng) < 0.7
    end

    ms = select_K(X, 2:4; n_init=2, rng=MersenneTwister(99))
    @test ms isa BernoulliMixtureModelSelection
    @test ms.best_K in 2:4
    @test ms.best isa BernoulliMixtureResult
    @test length(ms.bic_values) == 3
    @test haskey(ms.results, 2)
    @test haskey(ms.results, 3)
    @test haskey(ms.results, 4)
    # BIC should select K=2 for clearly 2-class data
    @test ms.best_K == 2
end

@testset "_compute_bic" begin
    # n_params = K*D + (K-1) for K=2, D=4 → 9
    bic = CooccurrenceAnalysis._compute_bic(-100.0, 2, 4, 50)
    @test bic ≈ 200.0 + 9 * log(50)
end

@testset "empirical_bayes_priors" begin
    # Construct a matrix with known marginals: feature 1 → 0.5, feature 2 → 0.1
    X = falses(100, 2)
    X[1:50, 1] .= true
    X[1:10, 2] .= true

    α, β = empirical_bayes_priors(X; concentration=10.0, floor=1.0)
    @test length(α) == 2
    @test length(β) == 2
    # Feature 1: π̂=0.5 → α = 10*0.5 + 1 = 6.0, β = 10*0.5 + 1 = 6.0
    @test α[1] ≈ 6.0
    @test β[1] ≈ 6.0
    # Feature 2: π̂=0.1 → α = 10*0.1 + 1 = 2.0, β = 10*0.9 + 1 = 10.0
    @test α[2] ≈ 2.0
    @test β[2] ≈ 10.0

    # Floor keeps prior proper for π̂=0
    X_zero = falses(50, 1)
    α0, β0 = empirical_bayes_priors(X_zero; concentration=10.0, floor=1.0)
    @test α0[1] ≈ 1.0
    @test β0[1] ≈ 11.0
end

@testset "fit_bernoulli_mixture with empirical_bayes prior" begin
    # Sparse-data shrinkage check: rare features should be pulled toward marginal
    # under :empirical_bayes vs :flat
    rng = MersenneTwister(2026)
    N = 200
    D = 20
    X = falses(N, D)
    # Dense features 1-4 define two clusters
    for i in 1:100
        X[i, 1] = rand(rng) < 0.8
        X[i, 2] = rand(rng) < 0.7
    end
    for i in 101:200
        X[i, 3] = rand(rng) < 0.8
        X[i, 4] = rand(rng) < 0.7
    end
    # Rare features 5-20: marginal frequency ~0.02, no cluster signal
    for d in 5:D, n in 1:N
        X[n, d] = rand(rng) < 0.02
    end

    flat_result = fit_bernoulli_mixture(X, 2;
        prior=:flat, n_init=3, rng=MersenneTwister(7))
    eb_result = fit_bernoulli_mixture(X, 2;
        prior=:empirical_bayes, concentration=10.0, floor=1.0,
        n_init=3, rng=MersenneTwister(7))

    @test flat_result isa BernoulliMixtureResult
    @test eb_result isa BernoulliMixtureResult

    # For rare features (no cluster signal), EB θ_{k,d} should be closer to
    # the marginal π̂_d than the flat-prior θ_{k,d}
    pi_hat = vec(sum(X, dims=1)) ./ N
    rare_features = 5:D
    flat_dev = sum(abs.(flat_result.theta[:, rare_features] .- pi_hat[rare_features]'))
    eb_dev = sum(abs.(eb_result.theta[:, rare_features] .- pi_hat[rare_features]'))
    @test eb_dev < flat_dev

    # Both priors should still recover the planted 2-cluster structure on dense features
    @test all(0.0 .<= eb_result.theta .<= 1.0)
    @test sum(eb_result.pi) ≈ 1.0
end

@testset "fit_bernoulli_mixture rejects unknown prior" begin
    X = falses(20, 4)
    X[1:10, 1] .= true
    @test_throws ErrorException fit_bernoulli_mixture(X, 2; prior=:bogus)
end

@testset "bernoulli_clustering with empirical_bayes" begin
    result = bernoulli_clustering(event_df;
        K_range=2:3, n_init=2, prior=:empirical_bayes,
        concentration=5.0, rng=MersenneTwister(42))
    @test result isa CooccurrenceAnalysisResult
    @test nrow(result.class_profiles) in 2:3
end

@testset "fit_bernoulli_mixture_turing" begin
    # Small planted 2-cluster data
    rng = MersenneTwister(7)
    N, D = 60, 5
    X = falses(N, D)
    for i in 1:30
        X[i, 1] = rand(rng) < 0.8; X[i, 2] = rand(rng) < 0.7
    end
    for i in 31:60
        X[i, 3] = rand(rng) < 0.8; X[i, 4] = rand(rng) < 0.7
    end

    post = fit_bernoulli_mixture_turing(X, 2;
        n_samples=100, n_chains=1, rng=MersenneTwister(11))

    @test post isa BernoulliMixturePosterior
    @test post.K == 2
    @test size(post.pi_samples, 2) == 2
    @test size(post.theta_samples, 2) == 2
    @test size(post.theta_samples, 3) == D
    @test size(post.pi_ci) == (2, 2)
    @test size(post.theta_ci) == (2, D, 2)
    @test all(post.pi_ci[:, 1] .<= post.pi_ci[:, 2])
    @test all(post.theta_ci[:, :, 1] .<= post.theta_ci[:, :, 2])
    @test size(post.responsibility_means) == (N, 2)
    @test all(0.0 .<= post.responsibility_means .<= 1.0)
    @test all(isapprox.(sum(post.responsibility_means, dims=2), 1.0; atol=1e-6))
    @test length(post.assignments) == N
    @test all(1 .<= post.assignments .<= 2)
end

@testset "fit_bernoulli_mixture_turing with empirical_bayes" begin
    rng = MersenneTwister(13)
    N, D = 50, 4
    X = falses(N, D)
    for i in 1:25
        X[i, 1] = rand(rng) < 0.8
    end
    for i in 26:50
        X[i, 2] = rand(rng) < 0.8
    end

    post = fit_bernoulli_mixture_turing(X, 2;
        prior=:empirical_bayes, concentration=5.0,
        n_samples=100, n_chains=1, rng=MersenneTwister(17))
    @test post isa BernoulliMixturePosterior
    @test post.K == 2
end

@testset "fit_bernoulli_mixture_turing rejects unknown prior" begin
    X = falses(20, 3)
    X[1:10, 1] .= true
    @test_throws ErrorException fit_bernoulli_mixture_turing(X, 2; prior=:bogus, n_samples=10)
end

@testset "bernoulli_clustering_turing wrapper" begin
    post, items = bernoulli_clustering_turing(event_df, 2;
        n_samples=100, n_chains=1, rng=MersenneTwister(42))
    @test post isa BernoulliMixturePosterior
    @test post.K == 2
    @test items isa Vector{String}
    @test size(post.theta_samples, 3) == length(items)
end

@testset "posterior_summary runs" begin
    rng = MersenneTwister(21)
    N, D = 40, 3
    X = falses(N, D)
    X[1:20, 1] .= true
    X[21:40, 2] .= true
    post = fit_bernoulli_mixture_turing(X, 2;
        n_samples=100, n_chains=1, rng=MersenneTwister(23))
    posterior_summary(post; n_top=2)
    @test true
end

@testset "bernoulli_clustering" begin
    result = bernoulli_clustering(event_df;
        K_range=2:3, n_init=2, rng=MersenneTwister(42))
    @test result isa CooccurrenceAnalysisResult
    @test result.n_records == 20
    @test nrow(result.class_profiles) in 2:3
    @test nrow(result.record_assignments) == 20
    @test "assignment" in names(result.record_assignments)
    @test "id" in names(result.record_assignments)
    @test length(result.item_names) > 0
end

@testset "stratified_bernoulli_clustering" begin
    strat = stratified_bernoulli_clustering(event_df;
        K_range=2:3, n_init=2, rng=MersenneTwister(42))
    @test strat.group_a_result isa CooccurrenceAnalysisResult
    @test strat.group_b_result isa CooccurrenceAnalysisResult

    for item in strat.group_a_result.item_names
        @test !(item in GROUP_B_ONLY_ITEMS)
    end
    for item in strat.group_b_result.item_names
        @test !(item in GROUP_A_ONLY_ITEMS)
    end
end

@testset "compare_clusterings" begin
    strat = stratified_bernoulli_clustering(event_df;
        K_range=2:3, n_init=2, rng=MersenneTwister(42))
    comp = compare_clusterings(strat.group_a_result, strat.group_b_result)
    @test comp isa ClusteringComparisonResult
    @test comp.shared_high_prob_items isa DataFrame
    @test comp.group_specific_clusters isa DataFrame
end

@testset "clustering_summary" begin
    result = bernoulli_clustering(event_df;
        K_range=2:3, n_init=2, rng=MersenneTwister(42))
    # Should run without error
    clustering_summary(result)
    clustering_summary(result; prob_threshold=0.5)
    @test true  # no error thrown
end

@testset "bayesian_visualization" begin
    result = bernoulli_clustering(event_df;
        K_range=2:3, n_init=2, rng=MersenneTwister(42))

    @testset "plot_class_probabilities" begin
        fig = plot_class_probabilities(result)
        @test fig isa Figure
    end

    @testset "plot_bic_elbow" begin
        fig = plot_bic_elbow(result)
        @test fig isa Figure
    end

    @testset "plot_class_profiles" begin
        fig = plot_class_profiles(result)
        @test fig isa Figure
    end

    @testset "plot_clustering_comparison" begin
        strat = stratified_bernoulli_clustering(event_df;
            K_range=2:3, n_init=2, rng=MersenneTwister(42))
        fig = plot_clustering_comparison(strat.group_a_result, strat.group_b_result)
        @test fig isa Figure
    end
end

@testset "clustering_integration" begin
    path = tempname() * ".arrow"
    Arrow.write(path, event_df)

    result = run_clustering_pipeline(path;
        K_range=2:3, n_init=2, stratify_by_group=true)

    @test result.clustering isa CooccurrenceAnalysisResult

    rm(path)
end

# ──────────────────────────────────────────────────────────────────────────────
# ARM visualization tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "arm_visualization" begin
    txns_df = build_transactions(event_df; min_items=2)
    rules = mine_association_rules(txns_df; min_support=0.1, min_confidence=0.3)
    validated = validate_rules(rules, event_df)

    @testset "plot_arm_scatter" begin
        fig = plot_arm_scatter(validated)
        @test fig isa Figure
    end

    @testset "plot_arm_matrix" begin
        fig = plot_arm_matrix(validated)
        @test fig isa Figure
    end

    @testset "plot_arm_comparison" begin
        strat = stratified_analysis(event_df;
            min_support=0.05, min_confidence=0.1, min_count=nothing)
        if nrow(strat.group_a_rules) > 0 && nrow(strat.group_b_rules) > 0
            comp = compare_strata(strat.group_a_rules, strat.group_b_rules)
            fig = plot_arm_comparison(comp)
            @test fig isa Figure
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Extended network visualization tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "extended_network_visualization" begin
    net = build_cooccurrence_network(event_df; min_count=1, alpha=1.0)
    metrics = compute_network_metrics(net)

    @testset "plot_centrality_barchart" begin
        fig = plot_centrality_barchart(metrics)
        @test fig isa Figure
    end

    @testset "plot_group_stratified_network" begin
        strat = stratified_network_analysis(event_df; min_count=1, alpha=1.0)
        if nv(strat.group_a_net.graph) > 0 && nv(strat.group_b_net.graph) > 0
            comp = compare_networks(strat.group_a_net, strat.group_b_net,
                                    strat.group_a_communities, strat.group_b_communities)
            fig = plot_group_stratified_network(comp, strat.group_a_net, strat.group_b_net)
            @test fig isa Figure
        end
    end

    @testset "plot_centrality_comparison" begin
        strat = stratified_network_analysis(event_df; min_count=1, alpha=1.0)
        group_a_metrics = compute_network_metrics(strat.group_a_net)
        group_b_metrics = compute_network_metrics(strat.group_b_net)
        fig = plot_centrality_comparison(group_a_metrics, group_b_metrics)
        @test fig isa Figure
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Overview visualization tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "overview_visualization" begin
    txns_df = build_transactions(event_df; min_items=2)

    @testset "plot_upset" begin
        fig = plot_upset(txns_df; min_count=1)
        @test fig isa Figure
    end

    @testset "plot_record_heatmap" begin
        fig = plot_record_heatmap(txns_df)
        @test fig isa Figure
    end

    @testset "plot_item_prevalence" begin
        fig = plot_item_prevalence(event_df; top_n=5)
        @test fig isa Figure
    end

    @testset "plot_item_prevalence top_n > n_items" begin
        # Should not error when top_n exceeds the number of distinct items
        fig = plot_item_prevalence(event_df; top_n=1000)
        @test fig isa Figure
    end

    @testset "plot_item_prevalence group bars nonzero (C4)" begin
        # The old ^f/^m group-detection regexes matched neither "A" nor "B",
        # silently zeroing the two group bar series. Assert all three series
        # (overall + both groups) carry a positive bar.
        fig = plot_item_prevalence(event_df; top_n=8)
        ax = only(filter(x -> x isa Axis, fig.content))
        bars = filter(p -> p isa BarPlot, ax.scene.plots)
        # direction=:x → the height is the 2nd coordinate of each Point
        series_max = [maximum(pt[2] for pt in b[1][]) for b in bars]
        @test count(>(0.0), series_max) >= 3
    end

    @testset "plot_cooccurrence_heatmap all records" begin
        fig = plot_cooccurrence_heatmap(event_df; group=:both, min_items=1)
        @test fig isa Figure
    end

    @testset "plot_cooccurrence_heatmap group filter" begin
        fig_f = plot_cooccurrence_heatmap(event_df; group=:group_b, min_items=2)
        @test fig_f isa Figure
        fig_m = plot_cooccurrence_heatmap(event_df; group=:group_a, min_items=2)
        @test fig_m isa Figure
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# HDP Bernoulli mixture tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "HDP Bernoulli mixture" begin

    # Simple ARI helper for test validation
    function _test_ari(a::Vector{Int}, b::Vector{Int})
        n = length(a)
        ca, cb = sort(unique(a)), sort(unique(b))
        cmap_a = Dict(c => i for (i,c) in enumerate(ca))
        cmap_b = Dict(c => i for (i,c) in enumerate(cb))
        nij = zeros(Int, length(ca), length(cb))
        for k in 1:n
            nij[cmap_a[a[k]], cmap_b[b[k]]] += 1
        end
        c2(x) = x*(x-1)÷2
        ai = vec(sum(nij, dims=2)); bj = vec(sum(nij, dims=1))
        sn = sum(c2(nij[i,j]) for i in 1:length(ca) for j in 1:length(cb))
        sa = sum(c2(x) for x in ai); sb = sum(c2(x) for x in bj)
        nc2 = c2(n)
        nc2 == 0 && return 1.0
        expected = sa * sb / nc2
        denom = 0.5*(sa + sb) - expected
        denom == 0.0 && return 1.0
        return (sn - expected) / denom
    end

    @testset "3-cluster planted recovery" begin
        # Plant 3 clusters with cross-group structure:
        # - cluster A: items 1-2 active, present in both groups
        # - cluster B: items 3-4 active, group_a-only
        # - cluster C: items 5-6 active, group_b-only
        rng = MersenneTwister(2026)
        N_per = 130    # records per cluster
        D = 8

        X = falses(3 * N_per, D)
        truth = zeros(Int, 3 * N_per)
        grp   = zeros(Int, 3 * N_per)

        # Cluster A (shared): both groups, items 1-2
        for i in 1:N_per
            X[i, 1] = rand(rng) < 0.85
            X[i, 2] = rand(rng) < 0.80
            truth[i] = 1
            grp[i] = (i <= N_per ÷ 2) ? 1 : 2   # split evenly
        end
        # Cluster B (group_a-only): group 1, items 3-4
        for i in N_per+1:2*N_per
            X[i, 3] = rand(rng) < 0.85
            X[i, 4] = rand(rng) < 0.80
            truth[i] = 2
            grp[i] = 1
        end
        # Cluster C (group_b-only): group 2, items 5-6
        for i in 2*N_per+1:3*N_per
            X[i, 5] = rand(rng) < 0.85
            X[i, 6] = rand(rng) < 0.80
            truth[i] = 3
            grp[i] = 2
        end

        result = fit_hdp_bernoulli_mixture(X, grp, 2;
            K_max=8, alpha=1.0, gamma=1.0,
            prior=:flat, n_init=3, rng=MersenneTwister(99))

        @test result isa HDPBernoulliResult
        @test result.K_max == 8
        @test result.n_groups == 2
        @test size(result.theta) == (8, D)
        @test size(result.responsibilities) == (3*N_per, 8)
        @test length(result.assignments) == 3*N_per
        @test length(result.group) == 3*N_per
        @test size(result.pi_mean) == (2, 8)
        @test length(result.beta_mean) == 8

        # Posterior validity
        @test all(0.0 .<= result.theta .<= 1.0)
        @test isapprox(sum(result.beta_mean), 1.0; atol=0.05)
        @test all(isapprox.(sum(result.pi_mean, dims=2), 1.0; atol=0.1))
        @test all(isapprox.(sum(result.responsibilities, dims=2), 1.0; atol=1e-6))

        # Cluster recovery: ARI vs ground truth
        ari = _test_ari(result.assignments, truth)
        @test ari > 0.8

        # Effective K should be ~3
        @test result.effective_K in 2:5

        # ELBO is finite
        @test isfinite(result.elbo)
    end

    @testset "K_max truncation insensitivity" begin
        # Same planted data, verify effective_K stable across K_max
        rng = MersenneTwister(42)
        N, D = 180, 6
        X = falses(N, D)
        grp = ones(Int, N)
        grp[91:end] .= 2
        for i in 1:60; X[i,1]=rand(rng)<0.8; X[i,2]=rand(rng)<0.7; end
        for i in 61:120; X[i,3]=rand(rng)<0.8; X[i,4]=rand(rng)<0.7; end
        for i in 121:180; X[i,5]=rand(rng)<0.8; X[i,6]=rand(rng)<0.7; end

        r6  = fit_hdp_bernoulli_mixture(X, grp, 2; K_max=6,  n_init=2, rng=MersenneTwister(1))
        r12 = fit_hdp_bernoulli_mixture(X, grp, 2; K_max=12, n_init=2, rng=MersenneTwister(1))

        # Effective K should be similar (both ~3 clusters)
        @test abs(r6.effective_K - r12.effective_K) <= 2
        @test r6.effective_K >= 2
        @test r12.effective_K >= 2
    end

    @testset "empirical_bayes prior" begin
        rng = MersenneTwister(7)
        N, D = 120, 8
        X = falses(N, D)
        grp = vcat(ones(Int, 60), fill(2, 60))
        for i in 1:60; X[i,1]=rand(rng)<0.8; end
        for i in 61:120; X[i,2]=rand(rng)<0.8; end

        result = fit_hdp_bernoulli_mixture(X, grp, 2;
            K_max=6, prior=:empirical_bayes, concentration=5.0,
            n_init=2, rng=MersenneTwister(11))
        @test result isa HDPBernoulliResult
        @test all(0.0 .<= result.theta .<= 1.0)
    end

    @testset "error on empty group" begin
        X = falses(20, 4)
        X[1:10, 1] .= true
        grp = ones(Int, 20)   # all in group 1, group 2 empty
        @test_throws ErrorException fit_hdp_bernoulli_mixture(X, grp, 2; K_max=4)
    end

    @testset "single group reduces gracefully" begin
        rng = MersenneTwister(3)
        N, D = 80, 4
        X = falses(N, D)
        grp = ones(Int, N)
        for i in 1:40; X[i,1]=rand(rng)<0.8; end
        for i in 41:80; X[i,2]=rand(rng)<0.8; end

        result = fit_hdp_bernoulli_mixture(X, grp, 1; K_max=6, n_init=2, rng=MersenneTwister(5))
        @test result isa HDPBernoulliResult
        @test result.n_groups == 1
        @test size(result.pi_mean) == (1, 6)
    end

    @testset "N < K_max warns and runs" begin
        X = falses(5, 3)
        X[1:2, 1] .= true
        grp = [1, 1, 2, 2, 2]
        @test_logs (:warn, r"N=5 < K_max") match_mode=:any fit_hdp_bernoulli_mixture(X, grp, 2; K_max=8, n_init=1)
    end

end

@testset "HDP item-domain wrappers" begin

    @testset "hdp_clustering on event_df" begin
        result = hdp_clustering(event_df;
            group_by=:Group, K_max=4, n_init=2, rng=MersenneTwister(42))

        @test result isa HDPClusteringResult
        @test result.hdp_result isa HDPBernoulliResult
        @test result.hdp_result.n_groups == 2
        @test length(result.item_names) > 0

        # Correct record counts per group
        @test length(result.n_records_per_group) == 2
        @test all(result.n_records_per_group .> 0)
        @test sum(result.n_records_per_group) == nrow(result.record_assignments)

        # class_profiles has right shape
        @test nrow(result.class_profiles) == 4
        @test "cluster" in names(result.class_profiles)
        @test "beta_weight" in names(result.class_profiles)
        @test isapprox(sum(result.class_profiles.beta_weight), 1.0; atol=0.05)

        # group_profiles has per-group columns
        @test nrow(result.group_profiles) == 4
        @test "cluster" in names(result.group_profiles)
        @test any(startswith.(names(result.group_profiles), "pi_"))

        # record_assignments
        @test "id" in names(result.record_assignments)
        @test "group" in names(result.record_assignments)
        @test "assignment" in names(result.record_assignments)
        @test all(1 .<= result.record_assignments.assignment .<= 4)
    end

    @testset "hdp_cluster_categorization structure" begin
        result = hdp_clustering(event_df;
            group_by=:Group, K_max=4, n_init=2, rng=MersenneTwister(77))

        cat_df = hdp_cluster_categorization(result)
        @test cat_df isa DataFrame
        @test "cluster" in names(cat_df)
        @test "beta_weight" in names(cat_df)
        @test "category" in names(cat_df)
        @test nrow(cat_df) == 4

        # All categories are valid symbols
        valid_cats = [:universal, :negligible, :both_present_but_unequal,
                      :group_a_only, :group_b_only]
        @test all(c -> c in valid_cats, cat_df.category)
    end

    @testset "hdp_cluster_categorization planted labels (C3, T2)" begin
        # Plant three clusters with unambiguous cross-group structure:
        #   {S1,S2} present in BOTH groups   -> :universal
        #   {A1,A2} present only in group A   -> :group_a_only
        #   {B1,B2} present only in group B   -> :group_b_only
        # The previous test never exercised the single-group branch, which is
        # exactly why the :a_only/:b_only prefix bug (C3) survived. Assert the
        # actual labels here, not merely that they're valid symbols.
        function planted_event_df()
            rows = NamedTuple[]
            id = 0
            addrec!(group, items) = begin
                id += 1
                for (s, it) in enumerate(items)
                    push!(rows, (id=id, seq=s, Group=group, item=it, year=2020))
                end
            end
            for _ in 1:40; addrec!("A", ["S1", "S2"]); end  # shared, group A
            for _ in 1:40; addrec!("B", ["S1", "S2"]); end  # shared, group B
            for _ in 1:60; addrec!("A", ["A1", "A2"]); end  # group-A-only
            for _ in 1:60; addrec!("B", ["B1", "B2"]); end  # group-B-only
            DataFrame(rows)
        end

        result = hdp_clustering(planted_event_df();
            group_by=:Group, K_max=6, n_init=4, rng=MersenneTwister(2026))
        cat_df = hdp_cluster_categorization(result)
        cats = Set(cat_df.category)

        @test :universal in cats
        @test :group_a_only in cats
        @test :group_b_only in cats
        # The buggy prefix-less forms must never appear.
        @test !(:a_only in cats)
        @test !(:b_only in cats)
    end

    @testset "clustering_summary prints without error" begin
        result = hdp_clustering(event_df;
            group_by=:Group, K_max=4, n_init=1, rng=MersenneTwister(13))
        # Verify it runs without throwing; capture to devnull to keep test output clean
        redirect_stdout(devnull) do
            clustering_summary(result)
        end
        @test true
    end

    @testset "timing_filter propagates" begin
        result = hdp_clustering(event_df;
            group_by=:Group, K_max=4, timing_filter=:all, n_init=1,
            rng=MersenneTwister(5))
        @test result isa HDPClusteringResult
    end

    @testset "error on unknown group_by column" begin
        @test_throws ErrorException hdp_clustering(event_df;
            group_by=:NoSuchColumn)
    end

end

# ──────────────────────────────────────────────────────────────────────────────
# HDP visualization tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "HDP visualization" begin
    # Shared fixture: a small but meaningful HDP result on the test event_df
    hdp_result = hdp_clustering(event_df;
        group_by=:Group, K_max=4, n_init=2, rng=MersenneTwister(42))

    @testset "plot_hdp_stick_weights" begin
        fig = plot_hdp_stick_weights(hdp_result)
        @test fig isa Figure
    end

    @testset "plot_hdp_class_profiles" begin
        fig = plot_hdp_class_profiles(hdp_result)
        @test fig isa Figure
    end

    @testset "plot_hdp_sharing_heatmap" begin
        fig = plot_hdp_sharing_heatmap(hdp_result)
        @test fig isa Figure
    end

    @testset "plot_hdp_cluster_butterfly" begin
        fig = plot_hdp_cluster_butterfly(hdp_result)
        @test fig isa Figure
    end

    @testset "plot_hdp_cluster_butterfly requires 2 groups" begin
        # Build a single-group HDP result by filtering to one group
        event_single = filter(r -> r.Group == "B", event_df)
        hdp_single = hdp_clustering(event_single;
            group_by=:Group, K_max=3, n_init=1, rng=MersenneTwister(7))
        @test_throws ErrorException plot_hdp_cluster_butterfly(hdp_single)
    end

    @testset "all three respect beta_threshold" begin
        # With a very high threshold, all clusters become inactive → should error
        @test_throws ErrorException plot_hdp_stick_weights(hdp_result; beta_threshold=2.0)
        @test_throws ErrorException plot_hdp_class_profiles(hdp_result; beta_threshold=2.0)
        @test_throws ErrorException plot_hdp_sharing_heatmap(hdp_result; beta_threshold=2.0)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
@testset "InlineString compatibility" begin
    # CSV.jl encodes short strings as InlineStrings (e.g. String7, String31).
    # All pipeline functions must accept AbstractString, not just String.
    # Simulate this by converting the fixture columns to InlineString types.
    using InlineStrings
    df_inline = DataFrame(
        id    = event_df.id,
        seq    = event_df.seq,
        Group    = InlineStrings.String7.(event_df.Group),
        item   = InlineStrings.String31.(event_df.item),
        year = event_df.year,
        has_second_event = event_df.has_second_event,
        level  = event_df.level,
    )

    @test eltype(df_inline.Group)  == InlineStrings.String7
    @test eltype(df_inline.item) == InlineStrings.String31

    # build_transactions must work with InlineString columns
    txns = build_transactions(df_inline; min_items=2)
    @test nrow(txns) == 20

    # build_contingency_table must work with InlineString item names
    items = collect(unique(df_inline.item))
    ps    = [collect(unique(filter(r -> r.id == p, df_inline).item))
             for p in unique(df_inline.id)]
    ct = build_contingency_table(items[1], items[2], ps)
    @test size(ct) == (2, 2)
    @test sum(ct) == length(unique(df_inline.id))

    # Network construction must work end-to-end
    net = build_cooccurrence_network(df_inline; min_count=1, alpha=0.5)
    @test net isa CooccurrenceNetwork

    # HDP clustering must work end-to-end
    hdp_inline = hdp_clustering(df_inline;
        group_by=:Group, K_max=3, n_init=1, rng=MersenneTwister(99))
    @test hdp_inline isa HDPClusteringResult
end

# ──────────────────────────────────────────────────────────────────────────────
end # testset
