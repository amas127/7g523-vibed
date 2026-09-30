"""Test-process defaults.

The RL tests train tiny models on a handful of rows.  The default OpenMP /
torch pool oversubscribes the machine (24 threads here) and spends more time
synchronising than computing, inflating the suite's wall clock.  Cap intra-op
parallelism to one thread unless the caller overrides it (``OMP_NUM_THREADS=4
uv run pytest`` still wins).  Numerics are unaffected: every assertion is
either exact on integers or tolerant via ``pytest.approx``.
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
