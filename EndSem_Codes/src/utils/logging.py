"""
Centralized logging configuration for the fabric-defect-dip project.

All scripts should call `get_logger(__name__)` rather than configuring
`logging` directly, so that log format/level stays consistent and logs
are optionally mirrored to `outputs/logs/`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_DEFAULT_LOG_DIR = Path("outputs/logs")


def _configure_root(log_to_file: bool, log_dir: Path, level: int) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("fabric_defect_dip")
    root.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(
    name: str,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a namespaced logger under the `fabric_defect_dip` root logger.

    Parameters
    ----------
    name : str
        Typically `__name__` of the calling module.
    log_to_file : bool
        Whether to additionally write logs to `outputs/logs/run.log`.
    log_dir : Path, optional
        Directory to write log files into. Defaults to `outputs/logs`.
    level : int
        Logging level (default INFO).
    """
    _configure_root(log_to_file, log_dir or _DEFAULT_LOG_DIR, level)
    logger_name = f"fabric_defect_dip.{name}" if not name.startswith("fabric_defect_dip") else name
    return logging.getLogger(logger_name)
