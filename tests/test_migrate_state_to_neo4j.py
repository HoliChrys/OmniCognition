"""D10 (Paperclip TAC-2465) — `scripts/migrate_state_to_neo4j.py` : add-only, verified,
the file archived and never touched."""

import os
import sys

from metacog.journal import Journal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import migrate_state_to_neo4j as mig  # noqa: E402
from tests.test_journal_state_neo4j import FakeApi, _scenario  # noqa: E402


def _file(root, store="global"):
    path = os.path.join(root, store, mig.JOURNAL_FILE)
    os.makedirs(os.path.dirname(path))
    j = Journal(path)
    _scenario(j)
    j.log_tags("fact_1", ["a"])
    j.close()
    return path


def test_dry_run_then_apply_then_nothing_left(tmp_path):
    src = _file(str(tmp_path / "root"))
    before = open(src, "rb").read()
    api = FakeApi()
    run = dict(root=str(tmp_path / "root"), archive_dir=str(tmp_path / "arch"),
               client_for=lambda ctx, acc: api)
    dry = mig.migrate_store("global", apply=False, **run)
    assert dry["to_send"] == {"docs": 2, "store_lists": ["okf_vocab", "ref_remaps"]}
    assert api.docs == {}                                    # a dry run writes nothing
    r = mig.migrate_store("global", apply=True, **run)
    assert r["verify"]["equal"] is True and r["sent"] == {"docs": 2, "store": 1}
    assert os.path.exists(tmp_path / "arch" / "global" / mig.JOURNAL_FILE)
    again = mig.migrate_store("global", apply=True, **run)
    assert again["to_send"]["docs"] == 0 and again["verify"]["equal"] is True
    assert again["archive"].endswith("(already there)")
    assert open(src, "rb").read() == before                  # the file is never written


def test_a_doc_the_store_already_holds_is_reported_never_overwritten(tmp_path):
    _file(str(tmp_path / "root"))
    api = FakeApi()
    run = dict(root=str(tmp_path / "root"), archive_dir=str(tmp_path / "arch"),
               client_for=lambda ctx, acc: api)
    mig.migrate_store("global", apply=True, **run)
    api.docs["notes:a"]["fields"]["title"] = "written by the gate after the switch"
    r = mig.migrate_store("global", apply=True, **run)
    assert r["differ"] == ["notes:a"] and r["to_send"]["docs"] == 0
    assert api.docs["notes:a"]["fields"]["title"] == "written by the gate after the switch"
    assert r["verify"]["equal"] is False                     # said, not repaired


def test_an_absent_file_is_named(tmp_path):
    r = mig.migrate_store("demo", root=str(tmp_path), archive_dir=None, apply=False,
                          client_for=lambda c, a: FakeApi())
    assert "absent" in r


def test_the_tag_report_compares_the_file_index_with_the_points_tags(tmp_path):
    _file(str(tmp_path / "root"))            # the file indexes fact_1 -> a

    class Api(FakeApi):
        def _call(self, method, path, body=None):
            if path == "/api/memory/points":
                return 200, {"points": [{"point_id": "fact_1", "fields": {"memory_tags": ["a", "b"]}}]}
            return super()._call(method, path, body)

    r = mig.migrate_store("global", root=str(tmp_path / "root"), archive_dir=None,
                          apply=False, client_for=lambda c, a: Api(), tags=True)
    assert r["tags"] == {"file": 1, "points": 2, "only_file": 0, "only_file_not_a_point": 0,
                         "only_points": 1}
