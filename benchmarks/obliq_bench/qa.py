"""Alias : `benchmarks.obliq_bench.qa` -> the OBLIQ-Bench debugger.

Kept so both `-m benchmarks.obliq_bench.qa` and `...debug_qa` work."""
from benchmarks.obliq_bench.debug_qa import (  # noqa: F401
    build_obliq_memory, main,
)

if __name__ == "__main__":
    main()
