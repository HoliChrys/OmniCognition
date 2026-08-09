"""The event channel is ADDITIVE : event_search ids union into agent_recall."""

from __future__ import annotations

import json
from benchmarks.locomo.mcp_meta_agent import _extract_fact_ids


class _Block:
    def __init__(self, text): self.text = text


class _Result:
    def __init__(self, obj): self.content = [_Block(json.dumps(obj))]; \
        self.structuredContent = None


def test_extract_harvests_walk_and_event_ids():
    # walk result
    assert _extract_fact_ids(_Result({"fact_ids_cumulative": ["A", "B"]})) == ["A", "B"]
    # event_search result -> cluster_ids + fact_ids unioned, deduped
    out = _extract_fact_ids(_Result(
        {"cluster_ids": ["C", "D"], "fact_ids": ["D", "E"]}))
    assert out == ["C", "D", "E"]
    # nothing relevant -> empty
    assert _extract_fact_ids(_Result({"other": 1})) == []
