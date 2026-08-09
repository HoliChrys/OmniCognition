"""
Funnel judge (Phase 3) — embedding pre-accept on stance cards.

Stage (a): cos(proposition, doc|stance|gloss) with an EMERGENT threshold
(mean+std of the candidate set's own sims). Pre-accepted items skip the batch
and are never overturned (recall-first). The batch listing carries the STANCE
card line. Deterministic via scripted fakes.
"""

from __future__ import annotations

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory


class _RejectAllLLM:
    """Distills the proposition verbatim ; batch and per-item reject ALL."""
    def __init__(self):
        self.prompts = []
    def generate(self, prompt, max_tokens=160):
        self.prompts.append(prompt)
        if "Attitude (broad)" in prompt:
            return "mocks official triumphalism about the strike"
        if "ITEM :" in prompt:
            return "VERDICT: IRRELEVANT"
        out = []
        for ln in prompt.splitlines():
            s = ln.strip()
            if s.startswith("[") and "]" in s and s[1:s.index("]")].isdigit():
                out.append(f"{s[1:s.index(']')]}: irrelevant")
        return "\n".join(out)


def _mem_with_stances():
    m = Memory(encoder=SimpleEncoder(), llm=_RejectAllLLM())
    target = m.ingest("we totally won lol", kind="FACT", id="A")
    others = [m.ingest(c, kind="FACT", id=i) for i, c in [
        ("B", "the weather is nice today in the mountains"),
        ("C", "recipe for chocolate cake and pastries"),
        ("D", "football match results from the weekend"),
        ("E", "gardening tips for spring vegetables")]]
    # tone tags so the auto gate picks the BATCH path
    for p in [target] + others:
        p.tags.append("tone:plain")
    # A's stance card = (verbatim) the distilled proposition -> max cosine
    m.bag_clear()
    from metacog.epistemic import Point, PointKind, SourceClass
    sp = Point(id="stance_A",
               content="mocks official triumphalism about the strike",
               embedding_orig=tuple(m.encoder.encode(
                   "mocks official triumphalism about the strike")),
               kind=PointKind.THOUGHT, keywords=["mocks"],
               keywords_source=SourceClass.GENERATOR,
               tags=["stance", "stance:of:a"])
    m.points.append(sp)
    return m, target, others


def test_pre_accept_survives_an_all_rejecting_judge():
    m, target, others = _mem_with_stances()
    labels = m.oblique_labels("find mocking tweets", [target] + others)
    assert labels[0] == "relevant"            # pre-accepted, never overturned
    assert all(lab == "irrelevant" for lab in labels[1:])


def test_stance_line_in_batch_prompt():
    m, target, others = _mem_with_stances()
    m.oblique_labels("find mocking tweets", others + [target])
    batch = next((p for p in m.llm.prompts if "ITEMS :" in p), "")
    # target may be pre-accepted (absent) ; check the mechanism on a non-pre one
    m2, t2, o2 = _mem_with_stances()
    labels = m2.oblique_labels("find mocking tweets", [t2] + o2[:2])  # 3 < 4
    batch2 = next((p for p in m2.llm.prompts if "ITEMS :" in p), "")
    assert "STANCE: mocks official triumphalism" in batch2


def test_small_sets_skip_pre_accept():
    m, target, others = _mem_with_stances()
    labels = m.oblique_labels("find mocking tweets", [target, others[0]])
    # 2 candidates < 4 -> no pre-accept ; all-rejecting judge rejects both
    assert labels == ["irrelevant", "irrelevant"]


def test_stance_of_helper():
    m, target, _ = _mem_with_stances()
    assert "triumphalism" in m.stance_of("A")
    assert m.stance_of("B") == ""
