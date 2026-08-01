"""Structural tests about the test suite itself (SPEC-013 AC2, AC6, AC8).

These assert properties no single script owns:
  * every script has a matching test module,
  * every test module actually has the production-ledger guard installed,
  * every public helper in forecast_common is referenced by its test module,
  * a checkout that carries local tooling has its commit guard installed.

They are the mechanism that stops the suite silently rotting — a new script with
no tests, or a new test module that forgets the guard, fails here rather than
being noticed by whoever happens to look.
"""
from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _ledger_guard  # noqa: E402
from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

TEST_MODULES = sorted(p.stem for p in TESTS_DIR.glob("test_*.py"))
SCRIPT_NAMES = sorted(p.stem for p in SCRIPTS_DIR.glob("*.py"))

# --- commit guard installation ---------------------------------------------
# The commit guard is tracked at tools/hooks/pre-commit and copied into the
# repository's hooks directory by tools/install-hooks.sh. Git never clones
# hooks, so a fresh checkout is unprotected until somebody runs the installer,
# and nothing at all signals that it is missing.
#
# The check below is conditional on purpose. Failing whenever hooks are absent
# would break `python3 -m unittest discover -s tests` on a plain clone and in
# CI, contradicting this toolkit's stdlib-only, runs-anywhere promise. But a
# checkout that also carries the local tooling .gitignore protects has
# something to leak, and there a missing guard is a silent, one-way hazard.
# So it fails only where both halves hold: local tooling on disk *and* no
# working guard.

HOOK_SOURCE = "tools/hooks/pre-commit"
HOOK_NAMES = ("pre-commit", "commit-msg")  # one script, installed under two names
_IGNORE_NOISE = frozenset({"__pycache__", ".DS_Store"})


def local_tooling_present(repo_root: Path) -> list:
    """Entries .gitignore protects that actually exist in this checkout.

    The list is read from .gitignore rather than restated here: one source of
    truth, and this file is published, so it should not carry an inventory of
    what is not. Existence is the test, not contents — an empty local directory
    is still a development checkout.
    """
    present = []
    for raw in (repo_root / ".gitignore").read_text().splitlines():
        entry = raw.strip().rstrip("/")
        if not entry or entry.startswith("#") or "*" in entry or entry in _IGNORE_NOISE:
            continue
        if (repo_root / entry).exists():
            present.append(entry)
    return present


def hooks_dir(repo_root: Path):
    """Where git would look for hooks here, or None when this is not a checkout.

    Asking git rather than assuming `.git/hooks` is what the installer does, and
    for the same reasons: it is correct inside a linked worktree and it honours
    a configured hooks path. An export, a tarball or a vendored copy has no
    `.git/` and cannot commit at all, so it is not an unguarded contributor.
    """
    if not (repo_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=str(repo_root), capture_output=True, text=True,
        )
    except OSError:  # no git on PATH — nothing here can commit either
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return repo_root / result.stdout.strip()


def commit_guard_problems(repo_root: Path, hooks: Path) -> list:
    """Why this checkout is unguarded, or [] when committing here is safe.

    Compares by content, not by existence, so a stale or hand-edited hook is
    caught too — the same drift `cmp -s` detects inside the installer.
    """
    if hooks is None or not local_tooling_present(repo_root):
        return []
    source = (repo_root / HOOK_SOURCE).read_bytes()
    problems = []
    for name in HOOK_NAMES:
        installed = hooks / name
        if not installed.exists():
            problems.append(f"{name} is not installed")
        elif installed.read_bytes() != source:
            problems.append(f"{name} differs from {HOOK_SOURCE}")
    return problems


class CoverageMappingTests(unittest.TestCase):
    def test_every_script_has_at_least_one_test_module(self) -> None:
        """A new script without tests fails here (AC2)."""
        missing = [
            name for name in SCRIPT_NAMES
            if not any(m == f"test_{name}" or m.startswith(f"test_{name}_")
                       for m in TEST_MODULES)
        ]
        self.assertEqual(missing, [], f"scripts with no test module: {missing}")

    def test_every_test_module_maps_to_a_script_or_is_structural(self) -> None:
        structural = {"test_commit_guard_nightshift", "test_suite_structure"}
        orphans = [
            m for m in TEST_MODULES
            if m not in structural
            and not any(m == f"test_{s}" or m.startswith(f"test_{s}_") for s in SCRIPT_NAMES)
        ]
        self.assertEqual(orphans, [], f"test modules with no script: {orphans}")

    def test_the_monolith_is_gone(self) -> None:
        self.assertNotIn("test_scripts", TEST_MODULES,
                         "tests/test_scripts.py should have been split by SPEC-013")

    def test_suite_is_not_a_package(self) -> None:
        """Stdlib discovery needs no __init__.py; adding one changes the import path."""
        self.assertFalse((TESTS_DIR / "__init__.py").exists())


class LedgerGuardInstallationTests(unittest.TestCase):
    def test_every_test_module_installs_the_shared_guard(self) -> None:
        """AC6 — the guard is per-module, so every module must bind it."""
        unguarded = []
        for name in TEST_MODULES:
            module = importlib.import_module(name)
            if (getattr(module, "setUpModule", None) is not _ledger_guard.setUpModule
                    or getattr(module, "tearDownModule", None)
                    is not _ledger_guard.tearDownModule):
                unguarded.append(name)
        self.assertEqual(unguarded, [],
                         f"test modules missing the production-ledger guard: {unguarded}")

    def test_guard_watches_both_production_ledgers(self) -> None:
        watched = {p.name for p in _ledger_guard.GUARDED_PATHS}
        self.assertEqual(watched, {"forecasts.jsonl"})
        self.assertEqual(len(_ledger_guard.GUARDED_PATHS), 2)

    def test_guard_reports_a_violation_for_a_created_ledger(self) -> None:
        """Proves the guard detects, restores, and reports — without touching real files."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "forecasts.jsonl"
            original = _ledger_guard.GUARDED_PATHS
            try:
                _ledger_guard.GUARDED_PATHS = (fake,)
                _ledger_guard.snapshot()
                fake.write_text('{"id": 1}\n')  # simulate a forgetful test
                violations = _ledger_guard.verify_and_restore()
            finally:
                _ledger_guard.GUARDED_PATHS = original
                _ledger_guard.snapshot()
            self.assertEqual(len(violations), 1)
            self.assertIn("CREATED", violations[0])
            self.assertFalse(fake.exists(), "the guard must restore, not just report")

    def test_guard_detects_a_same_bytes_rewrite(self) -> None:
        """mtime_ns is why content-only comparison is not enough."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "forecasts.jsonl"
            fake.write_text('{"id": 1}\n')
            original = _ledger_guard.GUARDED_PATHS
            try:
                _ledger_guard.GUARDED_PATHS = (fake,)
                _ledger_guard.snapshot()
                stat = fake.stat()
                os.utime(fake, ns=(stat.st_mtime_ns + 10**9, stat.st_mtime_ns + 10**9))
                violations = _ledger_guard.verify_and_restore()
            finally:
                _ledger_guard.GUARDED_PATHS = original
                _ledger_guard.snapshot()
            self.assertEqual(len(violations), 1)
            self.assertIn("REWRITTEN", violations[0])


class ForecastCommonCoverageTests(unittest.TestCase):
    def test_every_public_helper_is_named_in_its_test_module(self) -> None:
        """AC8 — a new helper added without a test fails here."""
        import forecast_common

        source = (TESTS_DIR / "test_forecast_common.py").read_text()
        public = [
            name for name, obj in vars(forecast_common).items()
            if not name.startswith("_")
            and inspect.isfunction(obj)
            and obj.__module__ == "forecast_common"
        ]
        self.assertTrue(public, "expected to find public helpers to check")
        untested = [name for name in public if name not in source]
        self.assertEqual(untested, [],
                         f"forecast_common helpers with no test reference: {untested}")


class CommitGuardInstallationTests(unittest.TestCase):
    """The live check against whatever checkout the suite is running in."""

    def test_a_checkout_carrying_local_tooling_has_the_commit_guard_installed(self) -> None:
        problems = commit_guard_problems(REPO_ROOT, hooks_dir(REPO_ROOT))
        detail = "".join(f"\n  {p}" for p in problems)
        self.assertEqual(
            problems, [],
            "this checkout carries local tooling but its commit guard is not in place:"
            f"{detail}\n"
            f"  on disk: {', '.join(local_tooling_present(REPO_ROOT))}\n"
            "  fix: tools/install-hooks.sh   (--force replaces a modified hook)",
        )

    def test_the_hooks_directory_resolves_where_there_is_a_checkout(self) -> None:
        """Without this, a broken resolution would make the check above vacuous."""
        if not (REPO_ROOT / ".git").exists():
            return  # an export or vendored copy: nothing to resolve
        self.assertIsNotNone(hooks_dir(REPO_ROOT),
                             "the hooks directory could not be resolved")


class CommitGuardConditionTests(unittest.TestCase):
    """Both halves of the condition must be real checks, so both are tested."""

    def _checkout(self, root: Path, tooling: bool) -> Path:
        """A miniature repository: a hook source, an ignore file, a hooks dir."""
        (root / "tools" / "hooks").mkdir(parents=True)
        (root / HOOK_SOURCE).write_text("#!/bin/sh\nexit 0\n")
        (root / ".gitignore").write_text(
            "# a comment\nlocal-tooling/\nnotes.md\n\n*.pyc\n__pycache__/\n")
        if tooling:
            (root / "local-tooling").mkdir()
            (root / "notes.md").write_text("local\n")
        hooks = root / "hooks"
        hooks.mkdir()
        return hooks

    def _install(self, root: Path, hooks: Path, mutate: bool = False) -> None:
        body = (root / HOOK_SOURCE).read_text()
        if mutate:
            body += "# hand-edited\n"
        for name in HOOK_NAMES:
            (hooks / name).write_text(body)

    def test_a_clone_without_local_tooling_never_fails(self) -> None:
        """The reader's clone and the CI checkout: nothing on disk to leak."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hooks = self._checkout(root, tooling=False)
            self.assertEqual(commit_guard_problems(root, hooks), [])

    def test_local_tooling_without_the_guard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hooks = self._checkout(root, tooling=True)
            problems = commit_guard_problems(root, hooks)
            self.assertEqual(len(problems), len(HOOK_NAMES), problems)
            for name in HOOK_NAMES:
                self.assertIn(f"{name} is not installed", problems)

    def test_a_modified_hook_is_caught_not_only_a_missing_one(self) -> None:
        """Comparison is by content: a hand-edited hook is drift, not protection."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hooks = self._checkout(root, tooling=True)
            self._install(root, hooks, mutate=True)
            problems = commit_guard_problems(root, hooks)
            self.assertEqual(len(problems), len(HOOK_NAMES), problems)
            for name in HOOK_NAMES:
                self.assertIn(f"{name} differs from {HOOK_SOURCE}", problems)

    def test_an_installed_guard_reports_clean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hooks = self._checkout(root, tooling=True)
            self._install(root, hooks)
            self.assertEqual(commit_guard_problems(root, hooks), [])

    def test_a_directory_that_is_not_a_checkout_is_not_a_failure(self) -> None:
        """Local tooling present, but nothing here can commit — so nothing to report."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._checkout(root, tooling=True)
            self.assertIsNone(hooks_dir(root))
            self.assertEqual(commit_guard_problems(root, hooks_dir(root)), [])

    def test_only_existing_ignored_entries_count_as_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._checkout(root, tooling=True)
            self.assertEqual(local_tooling_present(root), ["local-tooling", "notes.md"])
            (root / "__pycache__").mkdir()
            self.assertEqual(local_tooling_present(root), ["local-tooling", "notes.md"],
                             "build noise is not local tooling")


if __name__ == "__main__":
    unittest.main()
