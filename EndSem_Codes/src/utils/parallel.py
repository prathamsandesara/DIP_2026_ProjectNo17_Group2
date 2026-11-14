"""
Shared multiprocessing helpers.

Why this module exists
-------------------------
NumPy/SciPy/OpenCV on most machines (this project was benchmarked on
macOS, where NumPy links against Apple's Accelerate/vecLib BLAS backend)
internally multithread their own array operations across every available
CPU core by default. When `src.pipeline` and
`src.adaptive.scale_estimation` spawn N worker *processes* to parallelize
independent images across N cores, each of those worker processes ALSO
tries to internally multithread its own NumPy/SciPy calls -- so N
processes x (up to N) internal threads each means far more threads than
physical cores, and the OS scheduler thrashes switching between them
instead of doing useful work. Measured effect on this project's actual
workload: with 8 worker processes left free to also multithread
internally, per-image time was UNCHANGED from fully serial execution
(no speedup at all); forcing each worker to use exactly 1 internal
thread (via `limit_thread_oversubscription`, below) restored the
expected near-linear speedup.

The fix: force every underlying threaded math library to use exactly
ONE thread *inside* each worker process, so the only parallelism comes
from running separate OS processes across separate cores (which is what
we actually want -- one image's independent computation per core, not
one image's computation spread thinly across many cores while seven
other images sit idle waiting their turn).

This MUST run before NumPy/SciPy/OpenCV are imported in that process for
the environment variables to take effect (these libraries read them once,
at import/initialization time). Passing this function as a
`ProcessPoolExecutor(initializer=...)` guarantees it runs immediately
when each worker process starts, before any task (and therefore before
any of these libraries) is imported in that process.
"""
from __future__ import annotations

import os


def limit_thread_oversubscription() -> None:
    """Force NumPy/SciPy/OpenCV's internal threading (BLAS/OpenMP/etc.)
    to exactly 1 thread per process. Intended as a
    `ProcessPoolExecutor(initializer=...)` for worker processes that are
    themselves the unit of parallelism (one image per process) -- letting
    each process ALSO internally multithread causes severe
    oversubscription and, in this project's measured case, completely
    erased the benefit of process-level parallelism.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",   # OpenBLAS (common NumPy backend on Windows/Linux)
        "MKL_NUM_THREADS",        # Intel MKL (Windows NumPy often links against MKL)
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"

    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:  # pragma: no cover
        pass
