"""Regression coverage for the published managed-Kit exception in the commit guard."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "tools" / "hooks" / "pre-commit"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _fixture_repo() -> tempfile.TemporaryDirectory[str]:
    fixture = tempfile.TemporaryDirectory()
    root = Path(fixture.name)
    _git(root, "init", "-q", "-b", "main").check_returncode()
    _git(root, "config", "user.email", "hook-test@example.test").check_returncode()
    _git(root, "config", "user.name", "Hook Test").check_returncode()
    hooks = root / ".git" / "hooks"
    shutil.copy2(HOOK_SOURCE, hooks / "pre-commit")
    shutil.copy2(HOOK_SOURCE, hooks / "commit-msg")
    return fixture


class CommitGuardNightshiftTests(unittest.TestCase):
    def test_allows_managed_nightshift_payload(self) -> None:
        with _fixture_repo() as fixture:
            root = Path(fixture)
            payload = root / ".nightshift" / "release-marker.json"
            payload.parent.mkdir()
            payload.write_text('{"kit_version": "2.60.1"}\n', encoding="utf-8")
            _git(root, "add", "-f", ".nightshift/release-marker.json").check_returncode()

            committed = _git(root, "commit", "-m", "release managed kit")

            self.assertEqual(committed.returncode, 0, committed.stderr)

    def test_still_rejects_other_private_paths(self) -> None:
        with _fixture_repo() as fixture:
            root = Path(fixture)
            private = root / ".argo" / "private.txt"
            private.parent.mkdir()
            private.write_text("private\n", encoding="utf-8")
            _git(root, "add", "-f", ".argo/private.txt").check_returncode()

            committed = _git(root, "commit", "-m", "attempt private commit")

            self.assertNotEqual(committed.returncode, 0)
            self.assertIn("private tooling paths", committed.stderr)
