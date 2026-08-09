"""
Need-odds decay (Anderson-Schooler) — the one fitted hyperparameter (step 4).

need_odds is the power-law of a node's access ages ; fit_exponent grid-searches
the exponent that best separates useful nodes from useless ones by AUC ;
Memory.fit_decay closes the L3 loop from the journal's mark_useful labels +
access history, and the fitted exponent persists with the cloud.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.journal import Journal
from metacog.memory import Memory
from metacog.need_odds import DEFAULT_EXPONENT, fit_exponent, need_odds


def test_need_odds_power_law_recent_scores_higher():
    now = 100.0
    recent = need_odds([99.0], now, 0.5)      # age 1
    old = need_odds([1.0], now, 0.5)          # age 99
    assert recent > old > 0.0
    assert need_odds([], now) == 0.0          # empty history


def test_need_odds_clamps_same_instant():
    assert need_odds([100.0], 100.0, 0.5) == 1.0   # age 0 -> clamp to 1


def test_fit_exponent_picks_separating_d():
    now = 100.0
    # positives accessed often & recently ; negatives once, long ago.
    pos = [[99.0, 98.0, 97.0], [99.5, 98.5]]
    neg = [[2.0], [1.0]]
    res = fit_exponent(pos, neg, now)
    assert res["auc"] == 1.0                    # perfectly separable
    assert res["n_pos"] == 2 and res["n_neg"] == 2


def test_fit_exponent_empty_class_falls_back():
    res = fit_exponent([], [[1.0]], 10.0)
    assert res["exponent"] == DEFAULT_EXPONENT and res["auc"] == 0.5


# -- Memory integration ------------------------------------------------------

def test_memory_fit_decay_from_journal():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    # 'A' served in a useful retrieval, recently & often ; 'B' in a useless one.
    r_good = j.log_retrieval("q1", ["A"], ts=98.0)
    j.log_retrieval("q1b", ["A"], ts=99.0)
    r_bad = j.log_retrieval("q2", ["B"], ts=2.0)
    j.mark_useful(r_good, 2)
    j.mark_useful(r_bad, 0)
    res = m.fit_decay(now=100.0)
    assert res["n_pos"] == 1 and res["n_neg"] == 1
    assert res["auc"] == 1.0                    # A separates from B
    assert m.decay_exponent == float(res["exponent"])   # stored on the memory


def test_memory_need_odds_uses_fitted_exponent():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    j.log_retrieval("q", ["A"], ts=99.0)
    assert m.need_odds("A", now=100.0) > 0.0
    assert m.need_odds("Z", now=100.0) == 0.0   # no history


def test_memory_fit_decay_noop_without_both_classes():
    j = Journal()
    m = Memory(encoder=SimpleEncoder(), journal=j)
    r = j.log_retrieval("q", ["A"], ts=1.0)
    j.mark_useful(r, 2)                          # only positives
    before = m.decay_exponent
    res = m.fit_decay(now=10.0)
    assert res["n_neg"] == 0
    assert m.decay_exponent == before           # unchanged


def test_decay_exponent_persists(tmp_path):
    path = str(tmp_path / "m.pkl")
    m = Memory(encoder=SimpleEncoder(), storage_path=path)
    m.decay_exponent = 0.9
    m.save()
    m2 = Memory(encoder=SimpleEncoder(), storage_path=path)
    m2.load()
    assert m2.decay_exponent == 0.9


def test_memory_need_odds_noop_without_journal():
    m = Memory(encoder=SimpleEncoder())
    assert m.need_odds("A") == 0.0
    assert m.fit_decay()["n_pos"] == 0          # no crash, default
