"""Unit tests for Reciprocal Rank Fusion — the heart of hybrid retrieval."""
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.models import RetrievedHit, RetrieverSource


def _hit(doc_id, text="x"):
    return RetrievedHit(doc_id=doc_id, text=text, source=RetrieverSource.GRAPH)


def test_agreement_beats_single_list_dominance():
    """A doc ranked well in BOTH lists should beat a doc that is #1 in one."""
    graph = [_hit("A"), _hit("B"), _hit("C")]      # A is #1 only here
    vector = [_hit("B"), _hit("C"), _hit("A")]     # B,C strong in both
    fused = reciprocal_rank_fusion(
        [graph, vector], source_names=["graph", "vector"], k=60
    )
    order = [h.doc_id for h in fused]
    assert order[0] == "B"  # corroborated across both retrievers
    assert set(order) == {"A", "B", "C"}


def test_provenance_is_recorded():
    fused = reciprocal_rank_fusion(
        [[_hit("A")], [_hit("A")]], source_names=["graph", "vector"]
    )
    assert fused[0].source_ranks == {"graph": 1, "vector": 1}
    assert "both" in fused[0].explanation


def test_weighting_shifts_ranking():
    graph = [_hit("G"), _hit("X")]
    vector = [_hit("V"), _hit("X")]
    # Heavily upweight vector: V should outrank G.
    fused = reciprocal_rank_fusion(
        [graph, vector], source_names=["graph", "vector"], weights=[0.1, 5.0]
    )
    order = [h.doc_id for h in fused]
    assert order.index("V") < order.index("G")


def test_top_n_truncation():
    lists = [[_hit(c) for c in "ABCDE"]]
    fused = reciprocal_rank_fusion(lists, top_n=2)
    assert len(fused) == 2


def test_misaligned_inputs_raise():
    try:
        reciprocal_rank_fusion([[_hit("A")]], weights=[1.0, 2.0])
        assert False, "expected ValueError"
    except ValueError:
        pass
