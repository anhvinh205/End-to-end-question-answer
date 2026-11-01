import faiss
import numpy as np
import pandas as pd
import pytest

from nq_qa.data.passage import Passage
from nq_qa.models.retriever import (
    DenseEncoder,
    HybridRetriever,
    IndexMismatchError,
    PassageIndex,
    reciprocal_rank_fusion,
)

from .conftest import CAPITALS

PASSAGES = [
    Passage(id=f"p-{country}", text=f"the capital of {country} is {city} and it is a large city.")
    for country, city in CAPITALS.items()
]


@pytest.fixture(scope="module")
def encoder(tiny_encoder_dir):
    return DenseEncoder(str(tiny_encoder_dir), pooling="mean", normalize=True, max_length=48)


# ----------------------------------------------------------------------------- fusion
def test_rrf_scores_by_hand():
    fused = reciprocal_rank_fusion([[10, 20, 30], [20, 40]], k=60)
    assert fused[10] == pytest.approx(1 / 61)
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[40] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == 20  # ranked by both lists


def test_rrf_missing_document_contributes_nothing():
    fused = reciprocal_rank_fusion([[1], [2]], k=60)
    assert fused[1] == fused[2] == pytest.approx(1 / 61)


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


# ---------------------------------------------------------------------------- encoder
def test_encoder_shapes_dtype_and_normalisation(encoder):
    vectors = encoder.encode([p.text for p in PASSAGES])
    assert vectors.shape == (len(PASSAGES), encoder.dim) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoder_keeps_input_order_regardless_of_length_sorting(encoder):
    texts = [
        "short",
        "a considerably longer text about the capital of france and japan",
        "mid length text",
    ]
    batched = encoder.encode(texts, batch_size=2)
    singles = np.vstack([encoder.encode([t]) for t in texts])
    assert np.allclose(batched, singles, atol=1e-5)


def test_encoder_progress_bar_does_not_change_the_output(encoder):
    texts = [p.text for p in PASSAGES[:4]]
    assert np.allclose(encoder.encode(texts), encoder.encode(texts, show_progress=True))


def test_encoder_empty_input(encoder):
    assert encoder.encode([]).shape == (0, encoder.dim)


def test_cls_pooling_is_not_normalised_by_default_choice(tiny_encoder_dir):
    cls = DenseEncoder(str(tiny_encoder_dir), pooling="cls", normalize=False, max_length=48)
    vectors = cls.encode(["the capital of france"])
    assert vectors.shape == (1, cls.dim)
    assert cls.spec["pooling"] == "cls" and cls.spec["normalize"] is False


def test_invalid_pooling(tiny_encoder_dir):
    with pytest.raises(ValueError):
        DenseEncoder(str(tiny_encoder_dir), pooling="max")


# ------------------------------------------------------------------------------ index
def test_bm25_only_index(encoder):
    index = PassageIndex.build(PASSAGES, encoder=None)
    assert not index.has_dense and index.metric is None and len(index) == len(PASSAGES)
    with pytest.raises(RuntimeError):
        index.dense_search(np.zeros((1, 4), dtype=np.float32), 3)


def test_build_deduplicates_and_validates(encoder):
    index = PassageIndex.build(PASSAGES + PASSAGES[:3], encoder=None)
    assert len(index) == len(PASSAGES)
    with pytest.raises(ValueError):
        PassageIndex.build([], encoder=None)
    with pytest.raises(ValueError):
        PassageIndex.build(PASSAGES, encoder, index_type="HNSW")


def test_flat_ip_finds_the_passage_itself(encoder):
    index = PassageIndex.build(PASSAGES, encoder, index_type="FlatIP")
    assert index.metric == "ip"
    query = encoder.encode([PASSAGES[4].text])
    (position, score), *_ = index.dense_search(query, 3)
    assert position == 4 and score == pytest.approx(1.0, abs=1e-4)


def test_flat_l2_scores_are_negated_distances(tiny_encoder_dir):
    enc = DenseEncoder(str(tiny_encoder_dir), pooling="cls", normalize=False, max_length=48)
    index = PassageIndex.build(PASSAGES, enc, index_type="FlatL2")
    assert index.metric == "l2"
    hits = index.dense_search(enc.encode([PASSAGES[2].text]), 4)
    assert hits[0][0] == 2 and hits[0][1] == pytest.approx(0.0, abs=1e-4)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_ivf_falls_back_to_flat_for_tiny_corpora(encoder):
    index = PassageIndex.build(PASSAGES, encoder, index_type="IVFFlat", nlist=64)
    assert isinstance(index.faiss_index, faiss.IndexFlat)


def test_ivf_is_used_when_there_is_enough_data(encoder):
    # 132 distinct texts built only from words the tiny tokenizer knows (an unknown word
    # maps to [UNK] and would make many passages share one embedding)
    pairs = [(c, k) for c in CAPITALS for k in CAPITALS.values() if CAPITALS[c] != k]
    passages = [
        Passage(id=f"x{i}", text=f"the capital of {c} is {k} and it is a large city.")
        for i, (c, k) in enumerate(pairs)
    ]
    index = PassageIndex.build(passages, encoder, index_type="IVFFlat", nlist=64, nprobe=64)
    assert isinstance(index.faiss_index, faiss.IndexIVFFlat) and index.faiss_index.ntotal == len(
        pairs
    )
    assert index.faiss_index.nlist == len(pairs) // 39  # nlist is capped by the training-set size
    index.faiss_index.nprobe = index.faiss_index.nlist  # exhaustive probing makes the check exact
    top = index.dense_search(encoder.encode([passages[17].text]), 1)
    assert top[0][0] == 17


# ------------------------------------------------------------------------ persistence
def test_save_load_roundtrip_preserves_results(encoder, tmp_path):
    retriever = HybridRetriever(encoder=encoder, mode="hybrid")
    retriever.index = retriever.build_index(PASSAGES)
    before = retriever.retrieve("capital of peru", top_k=5)
    retriever.save(tmp_path / "idx.faiss", tmp_path / "meta.parquet")

    manifest = (tmp_path / "idx.manifest.json").read_text()
    assert '"num_passages": 12' in manifest and '"has_dense": true' in manifest

    fresh = HybridRetriever(encoder=encoder, mode="hybrid")
    fresh.load(tmp_path / "idx.faiss", tmp_path / "meta.parquet")
    after = fresh.retrieve("capital of peru", top_k=5)
    assert [p.doc_id for p in before] == [p.doc_id for p in after]
    assert [p.score for p in before] == [p.score for p in after]


def test_load_rejects_an_index_built_with_another_encoder(encoder, tiny_encoder_dir, tmp_path):
    retriever = HybridRetriever(encoder=encoder)
    retriever.index = retriever.build_index(PASSAGES)
    retriever.save(tmp_path / "i.faiss", tmp_path / "m.parquet")
    other = DenseEncoder(str(tiny_encoder_dir), pooling="cls", normalize=False, max_length=48)
    with pytest.raises(IndexMismatchError, match="Rebuild the index"):
        HybridRetriever(encoder=other).load(tmp_path / "i.faiss", tmp_path / "m.parquet")


def test_load_without_manifest_explains_how_to_rebuild(tmp_path):
    with pytest.raises(FileNotFoundError, match="make index"):
        PassageIndex.load(tmp_path / "missing.faiss", tmp_path / "missing.parquet")


def test_load_detects_out_of_sync_artifacts(encoder, tmp_path):
    retriever = HybridRetriever(encoder=encoder)
    retriever.index = retriever.build_index(PASSAGES)
    retriever.save(tmp_path / "i.faiss", tmp_path / "m.parquet")
    frame = pd.read_parquet(tmp_path / "m.parquet").iloc[:-2]
    frame.to_parquet(tmp_path / "m.parquet", index=False)
    with pytest.raises(IndexMismatchError, match="out of sync"):
        PassageIndex.load(tmp_path / "i.faiss", tmp_path / "m.parquet")


def test_bm25_only_index_roundtrip(tmp_path):
    PassageIndex.build(PASSAGES, None).save(tmp_path / "i.faiss", tmp_path / "m.parquet")
    loaded = PassageIndex.load(tmp_path / "i.faiss", tmp_path / "m.parquet")
    assert not loaded.has_dense and len(loaded) == len(PASSAGES)


# ---------------------------------------------------------------------------- retriever
def test_bm25_mode_needs_no_encoder():
    retriever = HybridRetriever(encoder=None, mode="bm25")
    retriever.index = retriever.build_index(PASSAGES)
    top = retriever.retrieve("what is the capital of peru", top_k=3)
    assert top[0].doc_id == "p-peru" and top[0].sparse_score and top[0].dense_score is None
    assert [p.rank for p in top] == [1, 2, 3]
    assert retriever.available_modes() == ("bm25",)


def test_dense_and_hybrid_modes_require_an_encoder():
    retriever = HybridRetriever(encoder=None, mode="bm25")
    retriever.index = retriever.build_index(PASSAGES)
    for mode in ("dense", "hybrid"):
        with pytest.raises(ValueError, match="needs a dense index"):
            retriever.retrieve("capital", mode=mode)


def test_hybrid_returns_both_signals(encoder):
    retriever = HybridRetriever(encoder=encoder, mode="hybrid", candidate_pool=12)
    retriever.index = retriever.build_index(PASSAGES)
    results = retriever.retrieve("what is the capital of peru", top_k=5)
    assert len(results) == 5
    assert "p-peru" in {r.doc_id for r in results}
    assert results[0].dense_score is not None or results[0].sparse_score is not None
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert retriever.available_modes() == ("bm25", "dense", "hybrid")


def test_dense_mode_ranks_by_similarity(encoder):
    retriever = HybridRetriever(encoder=encoder, mode="dense")
    retriever.index = retriever.build_index(PASSAGES)
    top = retriever.retrieve(PASSAGES[7].text, top_k=2)
    assert top[0].doc_id == PASSAGES[7].id and top[0].score >= top[1].score


def test_top_k_is_clipped_to_the_corpus_size(encoder):
    retriever = HybridRetriever(encoder=None, mode="bm25")
    retriever.index = retriever.build_index(PASSAGES[:3])
    assert len(retriever.retrieve("capital city", top_k=50)) <= 3


def test_explicit_index_does_not_touch_the_shared_one(encoder):
    retriever = HybridRetriever(encoder=encoder, mode="hybrid")
    retriever.index = shared = retriever.build_index(PASSAGES)
    ephemeral = retriever.build_index(
        [Passage(id="doc-1", text="completely different pdf text about gardening and music")]
    )
    results = retriever.retrieve("gardening", top_k=1, index=ephemeral)
    assert results[0].doc_id == "doc-1"
    assert retriever.index is shared and len(shared) == len(PASSAGES)


def test_error_paths(encoder):
    retriever = HybridRetriever(encoder=encoder)
    with pytest.raises(RuntimeError, match="No index"):
        retriever.retrieve("anything")
    with pytest.raises(ValueError):
        HybridRetriever(mode="semantic")
    retriever.index = retriever.build_index(PASSAGES)
    with pytest.raises(ValueError):
        retriever.retrieve("q", mode="semantic")
    with pytest.raises(RuntimeError, match="Nothing to save"):
        HybridRetriever().save("a", "b")
