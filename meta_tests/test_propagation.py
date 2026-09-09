"""Tests for the proposed method (embeddings + graph + matrix propagation).

These live in meta_tests/ (not tests/) so they are NOT picked up as artifacts
of the system under analysis -- tests/ holds only the domain tests.

Uses the offline `lexical` semantic backend so no model download is needed.
These lock in the core research claim: an artifact with low text similarity to
the change can still receive a high impact score via dependency propagation.
"""

from extract_artifacts import extract
from graph_propagation_impact import rank


def _rank(target):
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nodes, edges = extract(root)
    results, _ = rank(list(nodes.values()), edges, target, semantic="lexical",
                      model_name="all-MiniLM-L6-v2")
    return {r["artifact"]: r for r in results}


def test_propagation_lifts_structurally_reachable_artifact():
    ranked = _rank("R1")
    complete_order = ranked["order_service.completeOrder"]
    # completeOrder's text is unrelated to "timeout" yet it is reachable
    # R1 -> payment.timeout -> processPayment -> completeOrder, so the
    # structural signal must exceed the semantic one.
    assert complete_order["structural"] > complete_order["semantic"]
    assert complete_order["structural"] > 0


def test_purely_structural_artifact_gets_nonzero_score():
    ranked = _rank("R1")
    # This test shares no vocabulary with R1 (semantic == 0) but is reachable.
    order_test = ranked["test_order.test_complete_order"]
    assert order_test["semantic"] == 0
    assert order_test["structural"] > 0


def test_directly_traced_config_ranks_top():
    ranked = _rank("R1")
    top = max(ranked.values(), key=lambda r: r["score"])
    assert top["artifact"] == "payment.timeout"
