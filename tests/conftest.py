"""
Pytest conftest — placeholder.

Test encoder stubs use hashlib (deterministic) so test outcomes are
reproducible regardless of PYTHONHASHSEED.
"""
import os as _os


def pytest_configure(config):
    # OO·V2·M3 (TAC2-1157): the store tests that run against a REAL tachikoma API + Neo4j
    # (`tests/neo4j_live.py`). Skipped, with their reason, without OMNI_TEST_API_URL.
    config.addinivalue_line(
        "markers", "neo4j: runs Memory on Neo4jBackend against a live test API + Neo4j")


# JC2-3 (TAC2-1202): the whole suite runs on the ClickHouse usage journal when
# OMNI_TEST_JOURNAL=clickhouse and OMNI_TEST_CLICKHOUSE_URL points at a THROWAWAY
# server (http://user:password@host:port). `Journal` is swapped before any test
# module imports it : every `Journal(path)` then builds a ClickHouseJournal whose
# mutable state stays in SQLite at `path`. A file path keeps its context across
# reopenings (the persistence tests reopen it) ; `:memory:` gets a fresh one.
if _os.environ.get("OMNI_TEST_JOURNAL") == "clickhouse":
    import uuid as _uuid
    from urllib.parse import urlsplit as _urlsplit

    import metacog.journal as _journal_mod
    from metacog.journal_clickhouse import (ClickHouseJournal as _CHJ,
                                            HttpClickHouseTransport as _HTTP)

    _url = _urlsplit(_os.environ["OMNI_TEST_CLICKHOUSE_URL"])
    _TRANSPORT = _HTTP(f"{_url.scheme}://{_url.hostname}:{_url.port}",
                       user=_url.username or "default", password=_url.password or "")
    _RUN = _uuid.uuid4().hex[:12]
    _DB = _os.environ.get("OMNI_TEST_CLICKHOUSE_DB", "omni_journal_test")
    _SQLiteJournal = _journal_mod.Journal

    def _clickhouse_journal(path: str = ":memory:"):
        ctx = (f"test.{_RUN}.{_uuid.uuid4().hex}" if path == ":memory:"
               else f"test.{_RUN}.{path}")
        j = _CHJ(_TRANSPORT, ctx, database=_DB, state=_SQLiteJournal(path))
        j.path = path
        return j

    _journal_mod.Journal = _clickhouse_journal

    def clickhouse_usage_rows(path: str, sql: str, **params) -> list:
        """Rows of the usage journal the conftest journal at `path` writes
        (`{db}` and `{ctx:String}` are bound) — for the tests that look at
        WHERE a write landed."""
        return _TRANSPORT.select(sql.replace("{db}", _DB),
                                 {"ctx": f"test.{_RUN}.{path}", **params})


# JC2-5 (TAC2-1204): the whole suite runs on the PROXY journal (card J5 :
# proxy_ray + load_then_memory) when OMNI_TEST_JOURNAL=proxy. The throwaway
# ClickHouse stands for the proxy's target (rows inserted with the context POSED,
# as `infra-proxy.insert_omni_journal` does) and its SELECT for the API's read
# (`GET /api/memory/journal`). Every journal call sends its rows before it returns
# (`flush_interval_s=0`) ; reads are answered from memory, and a reopening of the
# same path reloads what was sent.
if _os.environ.get("OMNI_TEST_JOURNAL") == "proxy":
    import uuid as _uuid
    from urllib.parse import urlsplit as _urlsplit

    import metacog.journal as _journal_mod
    from metacog.journal_clickhouse import (ClickHouseJournal as _CHJ,
                                            HttpClickHouseTransport as _HTTP)
    from metacog.journal_proxy import JOURNAL_TABLES as _TABLES
    from metacog.journal_proxy import ProxyJournal as _PJ

    _url = _urlsplit(_os.environ["OMNI_TEST_CLICKHOUSE_URL"])
    _TRANSPORT = _HTTP(f"{_url.scheme}://{_url.hostname}:{_url.port}",
                       user=_url.username or "default", password=_url.password or "")
    _RUN = _uuid.uuid4().hex[:12]
    _DB = _os.environ.get("OMNI_TEST_CLICKHOUSE_DB", "omni_journal_test")
    _SQLiteJournal = _journal_mod.Journal
    _CHJ(_TRANSPORT, "schema", database=_DB, state=_SQLiteJournal())   # the DDL

    class _ClickHouseAsProxy:
        def __init__(self, ctx: str) -> None:
            self.ctx = ctx

        def append(self, table, rows):
            _TRANSPORT.insert(f"{_DB}.{table}",
                              [{**r, "context_path": self.ctx} for r in rows])

        def load(self, ctx, account=""):
            return {t: _TRANSPORT.select(
                f"SELECT * EXCEPT logged_at FROM {_DB}.{t} "
                "WHERE context_path = {ctx:String} AND account = {account:String} "
                "ORDER BY id", {"ctx": ctx, "account": account}) for t in _TABLES}

    def _proxy_journal(path: str = ":memory:"):
        ctx = (f"test.{_RUN}.{_uuid.uuid4().hex}" if path == ":memory:"
               else f"test.{_RUN}.{path}")
        proxy = _ClickHouseAsProxy(ctx)
        j = _PJ(proxy, ctx, loader=proxy, database=_DB, state=_SQLiteJournal(path),
                flush_interval_s=0)
        j.path = path
        return j

    _journal_mod.Journal = _proxy_journal

    def clickhouse_usage_rows(path: str, sql: str, **params) -> list:
        """Rows the proxy received for the conftest journal at `path`."""
        return _TRANSPORT.select(sql.replace("{db}", _DB),
                                 {"ctx": f"test.{_RUN}.{path}", **params})
