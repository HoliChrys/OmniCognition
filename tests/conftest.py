"""
Pytest conftest — placeholder.

Test encoder stubs use hashlib (deterministic) so test outcomes are
reproducible regardless of PYTHONHASHSEED.
"""


def pytest_configure(config):
    # OO·V2·M3 (TAC2-1157): the store tests that run against a REAL tachikoma API + Neo4j
    # (`tests/neo4j_live.py`). Skipped, with their reason, without OMNI_TEST_API_URL.
    config.addinivalue_line(
        "markers", "neo4j: runs Memory on Neo4jBackend against a live test API + Neo4j")
