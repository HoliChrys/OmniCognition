"""
The production encoder stack (mnema's) and the encoder-identity stamp.

  - `make_encoder` resolves METACOG_ENCODER : simple[:dim] / fastembed[:model]
    / auto (fastembed, else a WARNED fallback to SimpleEncoder).
  - a persisted brain stamps its `encoder_id` ; reopening it with another
    encoder re-encodes every point once (cosines in the old space are garbage),
    keeps content / tags / keywords, and retrieval works again.
  - the real fastembed model is exercised only when cached locally or
    METACOG_REAL_EMBED=1 (it is a 0.2 GB download).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from metacog import defaults as D
from metacog.defaults import SimpleEncoder, encoder_id, make_encoder
from metacog.memory import Memory


def test_encoder_ids():
    assert encoder_id(SimpleEncoder()) == "simple:32"
    assert encoder_id(SimpleEncoder(64)) == "simple:64"

    class Anon:
        dim = 7
    assert encoder_id(Anon()) == "Anon:7"


def test_make_encoder_simple_and_bad_spec(monkeypatch):
    assert isinstance(make_encoder("simple"), SimpleEncoder)
    assert make_encoder("simple:16").dim == 16
    monkeypatch.setenv("METACOG_ENCODER", "simple:8")
    assert make_encoder().dim == 8
    with pytest.raises(ValueError):
        make_encoder("bogus")


def test_make_encoder_auto_falls_back_with_a_warning(monkeypatch, capsys):
    class Boom:
        def __init__(self, *a, **k):
            raise ImportError("no fastembed here")
    monkeypatch.setattr(D, "FastEmbedEncoder", Boom)
    enc = make_encoder("auto")
    assert isinstance(enc, SimpleEncoder)
    assert "falling back to SimpleEncoder" in capsys.readouterr().err
    with pytest.raises(ImportError):              # explicit -> no silent downgrade
        make_encoder("fastembed")


def test_reopening_with_another_encoder_reencodes_once():
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "mem.pkl")
        m = Memory(encoder=SimpleEncoder(32), storage_path=store)
        a = m.ingest("swollen finger after exertion", kind="FACT", id="A")
        a.add_tag("health:x")
        m.ingest("quarterly finance report", kind="FACT", id="B")
        m.save()
        m2 = Memory(encoder=SimpleEncoder(64), storage_path=store)   # other space
        assert getattr(m2, "_reencoded_from", None) == "simple:32"
        for p in m2.points:
            assert len(p.embedding_orig) == 64 and len(p.delta_active) == 64
        a2 = next(p for p in m2.points if p.id == "A")
        assert "health:x" in a2.tags and a2.keywords            # nothing lost
        assert a2.keywords_embedding is not None and len(a2.keywords_embedding) == 64
        assert [h["id"] for h in m2.retrieve("swollen finger", k=1)] == ["A"]
        m2.save()
        m3 = Memory(encoder=SimpleEncoder(64), storage_path=store)  # same : no-op
        assert getattr(m3, "_reencoded_from", None) is None


def test_legacy_snapshot_without_stamp_reencodes_on_dim_mismatch():
    import pickle
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "mem.pkl")
        m = Memory(encoder=SimpleEncoder(32), storage_path=store)
        m.ingest("legacy point", kind="FACT", id="L")
        m.save()
        with open(store, "rb") as f:
            snap = pickle.load(f)
        snap.pop("encoder_id")                                    # pre-stamp file
        with open(store, "wb") as f:
            pickle.dump(snap, f)
        m2 = Memory(encoder=SimpleEncoder(48), storage_path=store)
        assert len(m2.points[0].embedding_orig) == 48


def _model_cached() -> bool:
    if os.environ.get("METACOG_REAL_EMBED") == "1":
        return True
    try:
        from fastembed import TextEmbedding
        TextEmbedding(model_name=D.DEFAULT_EMBED_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(), reason="fastembed model not cached ; "
                    "set METACOG_REAL_EMBED=1 to download it")
def test_real_fastembed_encoder_is_semantic_and_multilingual():
    enc = make_encoder("fastembed")
    assert encoder_id(enc) == f"fastembed:{D.DEFAULT_EMBED_MODEL}"
    assert enc.dim == 384
    from metacog.geometry import cosine
    fr = enc.encode("le chat dort sur le canapé")
    en = enc.encode("the cat is sleeping on the sofa")
    far = enc.encode("quarterly revenue grew twelve percent")
    assert cosine(fr, en) > 0.7 > cosine(fr, far)                 # cross-lingual
    para = enc.encode("index this repository's history one commit at a time")
    need = enc.encode("ingest a git repo commit by commit into the wiki")
    assert cosine(para, need) > 0.3 and cosine(para, need) > cosine(para, far)  # paraphrase
    # batched == single, and memoised
    assert enc.encode_batch(["a b", "c d"])[0] == enc.encode("a b")
