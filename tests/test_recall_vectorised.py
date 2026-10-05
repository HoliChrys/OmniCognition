"""
TAC-248 — the O(n) recall signals vectorised, the fuzzy channel memoised.

Each new computation is compared with the pre-TAC-248 code, kept verbatim
below as the reference :

R1  `_within_edits(a, b, k)` == `levenshtein(a, b) <= k` on random pairs
R2  `fuzzy_score` returns the same (score, point) list as the former
    per-point × per-token loop
R3  `_effective_matrix` rows are BIT-identical to `effective_embedding` /
    `effective_keyword_embedding`
    — also after a pull replaced a point's tuples (rows are cached on tuple
    identity) and when a vector is a list edited in place
R4  the keyword and content cosine pools of `retrieve_hybrid` list the same
    points in the same order as the former loop + list sort, scores to 1e-12
R5  same checks on a copy of a real store (the `global` context) when
    METACOG_STORE_COPY points to one — skipped otherwise (needs fastembed)
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pytest

from metacog import Point, PointKind

from metacog.fuzzy import (
    _MIN_TOKEN_LEN,
    _tokens,
    _within_edits,
    fuzzy_match,
    fuzzy_score,
    levenshtein,
)
from metacog.geometry import (
    _cosines,
    _effective_matrix,
    _top_pool,
    apply_pull,
    cosine,
    effective_embedding,
    effective_keyword_embedding,

)


# ---- the pre-TAC-248 computations, verbatim -------------------------------

def _ref_fuzzy_match(qtok, dtok):
    if qtok == dtok:
        return True
    budget = len(qtok) // 4
    if budget == 0:
        return False
    if abs(len(qtok) - len(dtok)) > budget:
        return False
    return levenshtein(qtok, dtok) <= budget


def _ref_fuzzy_score(query_text, points, k_pool):
    q_tokens = {t for t in _tokens(query_text) if len(t) >= _MIN_TOKEN_LEN}
    if not q_tokens:
        return []
    scored = []
    for p in points:
        d_tokens = {t for t in _tokens(p.content) if len(t) >= _MIN_TOKEN_LEN}
        if not d_tokens:
            continue
        matched = 0
        for qt in q_tokens:
            if any(_ref_fuzzy_match(qt, dt) for dt in d_tokens):
                matched += 1
        if matched:
            scored.append((matched / len(q_tokens), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k_pool]


def _ref_keyword_pool(q, points, k):
    pool = []
    for p in points:
        if not p.keywords_embedding:
            continue
        pool.append((cosine(q, tuple(p.keywords_embedding)), p))
    pool.sort(key=lambda x: x[0], reverse=True)
    return pool[:k]


def _ref_content_pool(q, points, t_now, k):
    pool = [(cosine(q, effective_embedding(p, t_now)), p) for p in points]
    pool.sort(key=lambda x: x[0], reverse=True)
    return pool[:k]


def _new_keyword_pool(q, points, k):
    kw = [p for p in points if p.keywords_embedding]
    K = np.asarray([p.keywords_embedding for p in kw], dtype=np.float64)
    return _top_pool(_cosines(q, K), kw, k)


def _new_content_pool(q, points, t_now, k):
    return _top_pool(_cosines(q, _effective_matrix(points, t_now, keyword=False)),
                     points, k)


def _assert_same_pool(got, ref):
    assert [p.id for _s, p in got] == [p.id for _s, p in ref]
    for (s_got, _), (s_ref, _) in zip(got, ref):
        assert abs(s_got - s_ref) < 1e-12


# ---- synthetic population -------------------------------------------------

_WORDS = ("berkeley berkley caroline carolyn necklace necklaces memoire memory "
          "architecture architectures tachikoma omni gateway gate context "
          "contexte retrieval recall rappel capture captures").split()


def _points(n, dim, seed):
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        kw = tuple(x / norm for x in v)
        if i % 9 == 0:
            kw_emb = None                     # no keyword embedding
        elif i % 13 == 0:
            kw_emb = pts[-1].keywords_embedding or kw   # exact cosine tie
        else:
            kw_emb = kw
        pts.append(Point(
            id=f"P{i}",
            content=" ".join(rng.choice(_WORDS) for _ in range(rng.randint(0, 8))),
            embedding_orig=tuple(rng.gauss(0.0, 1.0) for _ in range(dim)),
            kind=PointKind.FACT, keywords_embedding=kw_emb,
            delta_active=tuple(rng.gauss(0.0, 0.05) for _ in range(dim)),
            delta_latent=tuple(rng.gauss(0.0, 0.05) for _ in range(dim)),
            t_last_obs=float(i % 7),
        ))
    return pts


# ---- tests ----------------------------------------------------------------

def test_within_edits_matches_levenshtein():
    rng = random.Random(5)
    alphabet = "abcde"
    for _ in range(3000):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
        k = rng.randint(0, 4)
        assert _within_edits(a, b, k) == (levenshtein(a, b) <= k), (a, b, k)


def test_fuzzy_match_unchanged_on_word_pairs():
    for a in _WORDS:
        for b in _WORDS:
            assert fuzzy_match(a, b) == _ref_fuzzy_match(a, b)


def test_fuzzy_score_matches_reference():
    pts = _points(300, 8, seed=2)
    for q in ("Berkley caroline necklace", "memoire architecture tachikoma",
              "gate contexte rappel capture omni", "xyz", ""):
        for k in (5, 20, 1000):
            got = fuzzy_score(q, pts, k_pool=k)
            ref = _ref_fuzzy_score(q, pts, k_pool=k)
            assert [(s, p.id) for s, p in got] == [(s, p.id) for s, p in ref]


def _assert_rows_exact(pts, t_now):
    E = _effective_matrix(pts, t_now, keyword=False)
    KE = _effective_matrix(pts, t_now, keyword=True)
    for i, p in enumerate(pts):
        assert tuple(E[i].tolist()) == effective_embedding(p, t_now)
        assert tuple(KE[i].tolist()) == effective_keyword_embedding(p, t_now)


def test_effective_matrix_is_bit_identical():
    pts = _points(120, 16, seed=4)
    for t_now in (0.0, 3.5, 10.0):
        _assert_rows_exact(pts, t_now)


def test_row_cache_follows_replaced_vectors():
    """Rows are cached on tuple identity : a pull REPLACES the tuples, and
    the next matrix must carry the new values ; a list is never cached."""
    pts = _points(40, 8, seed=9)
    _assert_rows_exact(pts, 5.0)
    apply_pull(pts[3], pts[4], 1.0, 6.0)                  # new delta tuples
    pts[5].keywords_embedding = list(pts[6].embedding_orig)
    _assert_rows_exact(pts, 6.0)
    pts[5].keywords_embedding[0] += 1.0                   # in-place list edit
    _assert_rows_exact(pts, 6.0)


def test_cosine_pools_match_reference():
    pts = _points(400, 32, seed=6)
    rng = random.Random(8)
    queries = [tuple(rng.gauss(0.0, 1.0) for _ in range(32)) for _ in range(10)]
    queries.append(tuple(0.0 for _ in range(32)))          # zero-norm guard
    queries.append(pts[1].keywords_embedding)               # exact self match
    for q in queries:
        for k in (5, 20, 1000):
            _assert_same_pool(_new_keyword_pool(q, pts, k), _ref_keyword_pool(q, pts, k))
            _assert_same_pool(_new_content_pool(q, pts, 10.0, k),
                              _ref_content_pool(q, pts, 10.0, k))


@pytest.mark.skipif(not os.environ.get("METACOG_STORE_COPY"),
                    reason="set METACOG_STORE_COPY to a COPY of a real store")
def test_real_store_copy_matches_reference():
    """Run on a COPY of `global` : the store is loaded read-only in effect,
    nothing is ingested, nothing is saved."""
    from metacog.defaults import make_encoder
    from metacog.memory import Memory

    m = Memory(storage_path=os.environ["METACOG_STORE_COPY"], encoder=make_encoder())
    pts = list(m.points)
    t_now = m._now(None)
    step = max(1, len(pts) // 40)
    queries = ["quelle est l'architecture de la mémoire omni et du gate tachikoma ?"]
    queries += [(p.content or "")[:160] for p in pts[::step]]
    for q in queries:
        qe = tuple(m.encoder.encode(q))
        _assert_same_pool(_new_keyword_pool(qe, pts, 20), _ref_keyword_pool(qe, pts, 20))
        _assert_same_pool(_new_content_pool(qe, pts, t_now, 20),
                          _ref_content_pool(qe, pts, t_now, 20))
        got = fuzzy_score(q, pts, k_pool=20)
        ref = _ref_fuzzy_score(q, pts, k_pool=20)
        assert [(s, p.id) for s, p in got] == [(s, p.id) for s, p in ref]
    # geometric_spread only changed how its matrix is built : bit-identical
    # rows mean the same threshold, neighbours and distances.
    KE = _effective_matrix(pts, t_now, keyword=True)
    for i, p in enumerate(pts):
        assert tuple(KE[i].tolist()) == effective_keyword_embedding(p, t_now)
