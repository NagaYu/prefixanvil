"""PrefixAnvil Phase 3 measurement harness.

Pure, deterministic pieces (metric parsing, replay planning, summaries,
fidelity tables) live in modules without I/O state; the thin callers
(`runner`, `free_gpu/run_all.py`) do HTTP and process management.
"""

BENCH_SCHEMA_VERSION = "prefixanvil.bench/v1"
