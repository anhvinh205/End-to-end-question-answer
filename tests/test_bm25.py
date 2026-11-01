import math

import numpy as np
import pytest

from nq_qa.models.bm25 import BM25Index, tokenize

DOCS = [
    "paris is the capital of france",
    "tokyo is the capital of japan and a large city",
    "the eiffel tower is in paris paris paris",
    "bananas are yellow",
]


def test_tokenize_removes_stopwords_but_keeps_all_stopword_queries():
    assert tokenize("Who invented the World Wide Web?") == ["invented", "world", "wide", "web"]
    assert tokenize("who is who") == ["who", "is", "who"]
    assert tokenize("Who is the capital", remove_stopwords=False) == ["who", "is", "the", "capital"]


def test_score_matches_the_formula_by_hand():
    index = BM25Index(k1=1.5, b=0.75).fit(DOCS)
    n, df = 4, 2  # "paris" appears in docs 0 and 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    lengths = [len(tokenize(d)) for d in DOCS]
    avgdl = sum(lengths) / n
    tf, dl = 1, lengths[0]
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert index.get_scores("paris")[0] == pytest.approx(expected, rel=1e-5)
    assert index.get_scores("paris")[1] == 0.0


def test_ranking_prefers_more_relevant_documents():
    index = BM25Index().fit(DOCS)
    top = index.search("paris", top_k=3)
    assert [doc for doc, _ in top] == [2, 0]  # three mentions beat one, non-matches are excluded
    assert all(score > 0 for _, score in top)
    assert index.search("capital of japan")[0][0] == 1


def test_unknown_terms_and_degenerate_inputs():
    index = BM25Index().fit(DOCS)
    assert index.search("zzzzzz") == []
    assert index.search("paris", top_k=0) == []
    assert BM25Index().fit([]).search("paris") == []
    assert np.all(BM25Index().fit([]).get_scores("x") == 0)


def test_top_k_truncates_and_is_sorted():
    index = BM25Index().fit(DOCS)
    scores = [s for _, s in index.search("capital paris city", top_k=2)]
    assert len(scores) == 2 and scores == sorted(scores, reverse=True)


def test_idf_is_never_negative_even_for_ubiquitous_terms():
    index = BM25Index().fit(["common word here", "common word there", "common word again"])
    assert (index.get_scores("common") > 0).all()


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        BM25Index(b=1.5)
    with pytest.raises(ValueError):
        BM25Index(k1=-1)


def test_agrees_with_rank_bm25_on_the_ranking():
    rank_bm25 = pytest.importorskip("rank_bm25")
    # filler documents keep every query term's document frequency below N/2, where the
    # Okapi idf used by rank_bm25 degenerates to zero on toy corpora
    docs = DOCS + [
        f"unrelated filler document number {i} about gardening and music" for i in range(10)
    ]
    corpus = [tokenize(d) for d in docs]
    reference = rank_bm25.BM25Okapi(corpus, k1=1.5, b=0.75)
    ours = BM25Index(k1=1.5, b=0.75).fit(docs)
    for query in ["paris", "capital japan", "eiffel tower paris"]:
        ref_order = np.argsort(-reference.get_scores(tokenize(query)), kind="stable")[:2]
        our_order = [doc for doc, _ in ours.search(query, top_k=2)]
        assert list(ref_order[: len(our_order)]) == our_order
