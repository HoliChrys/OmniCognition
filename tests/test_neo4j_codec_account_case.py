"""`decode_point` and the exact case of `account:<id>` (TAC-2363).

Measured on 07/11/2026 (B3 of TAC-2334): the 4 stores of `tachikoma.paralelle.GenAI` were
written to Neo4j, then refused on re-read — `account:tachikoma-GenAI-archiviste` came back
lowered by `Point.__post_init__`, and `decode_point` refused the difference."""

from __future__ import annotations

import pytest

from metacog.defaults import SimpleEncoder
from metacog.memory import Memory
from metacog.store.neo4j_backend import decode_point, encode_point
from metacog.tachikoma_gate import _tag_account


def _point(account: str = ""):
    m = Memory(encoder=SimpleEncoder())
    p = m.ingest("the archiviste files its facts")
    if account:
        _tag_account(p, account)
    return p


def test_an_exact_case_account_tag_survives_the_round_trip():
    p = _point("tachikoma-GenAI-archiviste")
    t, f = encode_point(p, 0)
    q = decode_point(t, p.id, f)
    assert q.tags == p.tags == ["fact", "account:tachikoma-GenAI-archiviste"]


def test_any_other_tag_difference_is_still_refused():
    p = _point()
    t, f = encode_point(p, 0)
    f["memory_tags"] = ["fact", "Unnormalised Tag "]
    with pytest.raises(ValueError, match="tags changed on decode"):
        decode_point(t, p.id, f)
