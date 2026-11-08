"""JC2-6 (TAC2-1212) — the `.journal.db` usage tables move to the proxy journal
idempotently, archived first, never deleted ; the migrated journal answers like
the SQLite one. The proxy and the API are an in-memory fake (`_Proxy`)."""

import hashlib
import os
import sys

import pytest

from metacog.journal import SQLiteJournal
from metacog.journal_proxy import ProxyJournal
from tests.test_journal_backends import _reads, _script
from tests.test_journal_proxy import _Proxy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
import migrate_journal_to_proxy as mig  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "contexts"
    for store in ("ctx.a", "ctx.a/accounts/acc-1"):
        os.makedirs(r / store)
    j = SQLiteJournal(str(r / "ctx.a" / mig.JOURNAL_FILE))
    ids = _script(j)
    j.close()
    a = SQLiteJournal(str(r / "ctx.a/accounts/acc-1" / mig.JOURNAL_FILE))
    a.log_retrieval("acc", ["Z"], ts=1.0)
    a.log_tags("Z", ["x:y"])                    # mutable state : stays in SQLite
    a.close()
    return r, ids


def _run(root, proxy, archive, apply=True, stores=("ctx.a", "ctx.a/accounts/acc-1")):
    return [mig.migrate_store(s, root=str(root), archive_dir=str(archive), writer=proxy,
                              loader=proxy, apply=apply) for s in stores]


def test_dry_run_sends_and_archives_nothing(root, tmp_path):
    r, _ = root
    p = _Proxy()
    reps = _run(r, p, tmp_path / "arch", apply=False)
    assert all(x["status"] == "dry-run" for x in reps)
    assert reps[0]["to_send"]["retrievals"] == 5
    assert p.calls == 0 and not (tmp_path / "arch").exists()


def test_apply_archives_sends_verifies_and_a_second_pass_sends_nothing(root, tmp_path):
    r, ids = root
    src = r / "ctx.a" / mig.JOURNAL_FILE
    before = _sha(src)
    p = _Proxy()
    first = _run(r, p, tmp_path / "arch")
    assert [x["status"] for x in first] == ["ok", "ok"]
    assert os.path.isfile(tmp_path / "arch" / "ctx.a" / mig.JOURNAL_FILE)
    assert _sha(src) == before                   # the original is untouched
    v = first[0]["verify"]
    assert v["retrievals"] == {"sqlite": 5, "clickhouse": 5, "absent": 0}
    assert all(x["absent"] == 0 for x in v.values())
    calls = p.calls
    second = _run(r, p, tmp_path / "arch")
    assert [x["to_send"] for x in second] == [{}, {}]
    assert "already there" in second[0]["archived"]
    assert p.calls == calls
    # the migrated journal answers like the SQLite one
    ref = SQLiteJournal(str(src))
    migrated = ProxyJournal(p, "ctx.a", loader=p, start_flusher=False)
    assert _reads(migrated, ids) == _reads(ref, ids)
    acc = ProxyJournal(p, "ctx.a", account="acc-1", loader=p, start_flusher=False)
    assert acc.counts() == (1, 1)
    assert migrated.counts() == ref.counts()


def test_a_flag_set_after_the_first_pass_reaches_clickhouse(root, tmp_path):
    r, ids = root
    p = _Proxy()
    _run(r, p, tmp_path / "a1", stores=("ctx.a",))
    j = SQLiteJournal(str(r / "ctx.a" / mig.JOURNAL_FILE))
    f2 = ids["forgets"][1]
    j.mark_forget_merged(f2)                     # monotonic : must follow
    j.mark_useful(ids["retrievals"][0], 0)       # re-mark : never overrides
    j.close()
    rep = _run(r, p, tmp_path / "a2", stores=("ctx.a",))[0]
    assert rep["to_send"] == {"journal_flags": 1}
    pj = ProxyJournal(p, "ctx.a", loader=p, start_flusher=False)
    assert f2 not in [f["id"] for f in pj.pending_forgets()]
    assert pj.node_usefulness("A")["useful"] >= 1


def test_apply_without_an_archive_refuses(root):
    r, _ = root
    with pytest.raises(SystemExit):
        mig.migrate_store("ctx.a", root=str(r), archive_dir=None, writer=_Proxy(),
                          loader=_Proxy(), apply=True)


def test_a_store_without_a_journal_is_named(tmp_path):
    rep = mig.migrate_store("nothing", root=str(tmp_path), archive_dir=None,
                            writer=_Proxy(), loader=_Proxy(), apply=False)
    assert rep["status"] == "no-journal"
