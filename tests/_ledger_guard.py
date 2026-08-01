"""Production-ledger immutability guard, shared by every test module.

The toolkit's default log path resolves to a REAL user file
(`$FORECAST_LOG`, else `~/.superforecasting/forecasts.jsonl`). An injectable
seam that defaults to production is opt-in safety: it protects only the tests
that remember to inject. So the guard lives here and every test module binds it:

    from _ledger_guard import setUpModule, tearDownModule  # noqa: F401

`setUpModule`/`tearDownModule` are looked up per-module by unittest, so importing
the names into each module's namespace is what makes one implementation cover the
whole suite. Splitting the suite without this would silently shrink the guard to
whichever file still defined it (SPEC-013).

Compares `(sha256, mtime_ns)` rather than content alone: an atomic writer can
replace the inode with byte-identical bytes, which a content-only check misses.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

GUARDED_PATHS = (
    Path(os.path.expanduser("~/.superforecasting/forecasts.jsonl")),
    ROOT / "data" / "forecasts.jsonl",
)

_snapshots: dict[Path, tuple[bytes, str, int] | None] = {}


def snapshot() -> None:
    """Record (bytes, sha256, mtime_ns) for each production ledger, or None if absent."""
    for path in GUARDED_PATHS:
        if path.exists():
            payload = path.read_bytes()
            _snapshots[path] = (payload, hashlib.sha256(payload).hexdigest(),
                                path.stat().st_mtime_ns)
        else:
            _snapshots[path] = None


def verify_and_restore() -> list[str]:
    """Restore any touched production ledger and return the list of violations."""
    violations: list[str] = []
    for path in GUARDED_PATHS:
        before = _snapshots.get(path)
        if before is None:
            if path.exists():
                violations.append(f"{path} was CREATED by the test suite")
                path.unlink()
            continue
        payload, digest, mtime_ns = before
        if not path.exists():
            violations.append(f"{path} was DELETED by the test suite")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            os.utime(path, ns=(mtime_ns, mtime_ns))
            continue
        current = path.read_bytes()
        if hashlib.sha256(current).hexdigest() != digest:
            violations.append(f"{path} was MODIFIED by the test suite")
            path.write_bytes(payload)
            os.utime(path, ns=(mtime_ns, mtime_ns))
        elif path.stat().st_mtime_ns != mtime_ns:
            violations.append(f"{path} was REWRITTEN (same bytes, new mtime) by the test suite")
            os.utime(path, ns=(mtime_ns, mtime_ns))
    return violations


def setUpModule() -> None:  # noqa: N802 — unittest's required name
    snapshot()


def tearDownModule() -> None:  # noqa: N802 — unittest's required name
    violations = verify_and_restore()
    if violations:
        raise AssertionError(
            "test suite touched a production ledger (restored): " + "; ".join(violations)
            + " — set FORECAST_LOG or pass --log to a tmpdir in the offending test"
        )
