"""Phase 1 of the skill system : custom ingestion of a NAMED skill —
a theoretical directory tree of tools — as ordinary tagged points.

Verifies the contract the later phases (walk retrieval of tools, double
session-filtered query, latent distiller) depend on : tree topology via
lineage + ref: tokens, the section-scoping tags (name/user/session/date),
GENERATOR provenance, and id namespacing.
"""
from metacog.memory import Memory
from metacog.epistemic import PointKind, SourceClass


TREE = {
    "retrieval": {
        "hybrid.py": "RRF over 4 signals",
        "bm25.py": "content-first BM25",
    },
    "walk.py": "uncertainty-governed depth",
}


def _ingest():
    m = Memory()
    pts = m.ingest_skill(
        TREE, name="locomo_solver", user_id="u42",
        session_id="s7", date="2026-06-04",
    )
    return m, pts


def test_creates_root_plus_every_node():
    _m, pts = _ingest()
    # root + retrieval/ + hybrid.py + bm25.py + walk.py = 5
    assert len(pts) == 5
    assert pts[0].content.endswith("locomo_solver")  # root first


def test_section_scoping_tags_on_every_node():
    _m, pts = _ingest()
    for p in pts:
        for tag in ("skill", "tool", "name:locomo_solver",
                    "user:u42", "session:s7", "date:2026-06-04"):
            assert tag in p.tags, f"{tag} missing on {p.id}"
        assert ("skill_dir" in p.tags) ^ ("skill_file" in p.tags)


def test_tree_topology_via_lineage():
    _m, pts = _ingest()
    by_id = {p.id: p for p in pts}
    root = pts[0]
    # every non-root node chains up to the root through parents
    for p in pts[1:]:
        assert p.parents, f"{p.id} has no parent"
        cur, seen = p, 0
        while cur.parents and seen < 10:
            cur = by_id[cur.parents[0]]
            seen += 1
        assert cur.id == root.id


def test_topology_also_encoded_as_ref_tokens():
    _m, pts = _ingest()
    files = [p for p in pts if "skill_file" in p.tags]
    hybrid = next(p for p in files if "hybrid" in p.keywords)
    # its own path token and a parent-link token are present
    assert any(k == "ref:skill:locomo_solver:path:"
               "locomo_solver/retrieval/hybrid.py" for k in hybrid.keywords)
    assert any(k == "ref:skill:locomo_solver:parent:"
               "locomo_solver/retrieval" for k in hybrid.keywords)


def test_generator_provenance_and_id_namespace():
    _m, pts = _ingest()
    for p in pts:
        assert p.kind == PointKind.FACT
        assert p.keywords_source == SourceClass.GENERATOR
        assert p.id.startswith("skill_")  # never collides with evidence ids


def test_default_date_is_filled():
    m = Memory()
    pts = m.ingest_skill({"a.py": "x"}, name="n", user_id="u", session_id="s")
    assert any(t.startswith("date:") for t in pts[0].tags)


def test_no_laundering_after_skill_ingest():
    m, _ = _ingest()
    # skill content is GENERATOR but enters as content, never as an
    # Observation — the audit must stay clean.
    assert m.audit().get("laundering_violations", 0) == 0
