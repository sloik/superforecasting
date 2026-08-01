"""Tests for the ledger path resolution contract (SPEC-007).

    --log <path>  >  $FORECAST_LOG  >  ~/.superforecasting/forecasts.jsonl

The skill must never write inside its own checkout, so it keeps working when
installed read-only or symlinked into a harness's skills directory.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

import brier  # noqa: E402
import forecastlog  # noqa: E402
from forecast_common import default_log_path  # noqa: E402


class LedgerPathResolutionTests(unittest.TestCase):
    """--log > $FORECAST_LOG > ~/.superforecasting/forecasts.jsonl (SPEC-007)."""

    def test_env_var_is_used_when_no_log_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "f.jsonl"
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(log)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    forecastlog.main(["add", "--question", "Env?", "--prob", "0.4"])
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r["event"] for r in rows], ["forecast"])

    def test_falls_back_to_home_when_env_unset(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            env = {k: v for k, v in os.environ.items() if k != "FORECAST_LOG"}
            env["HOME"] = home
            with mock.patch.dict(os.environ, env, clear=True):
                resolved = default_log_path()
            self.assertEqual(resolved, os.path.join(home, ".superforecasting", "forecasts.jsonl"))
            self.assertFalse(Path(resolved).exists(), "resolving must not create the ledger")

    def test_explicit_log_flag_beats_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chosen = Path(td) / "chosen.jsonl"
            ignored = Path(td) / "ignored.jsonl"
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(ignored)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    forecastlog.main([
                        "--log", str(chosen), "add", "--question", "Flag?", "--prob", "0.4",
                    ])
            self.assertTrue(chosen.exists())
            self.assertFalse(ignored.exists(), "$FORECAST_LOG must not be written when --log given")

    def test_write_creates_missing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "nested" / "deeper" / "f.jsonl"
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(log)}):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    forecastlog.main(["add", "--question", "Nested?", "--prob", "0.4"])
                    forecastlog.main(["list"])
            self.assertTrue(log.exists())
            self.assertIn("Nested?", buf.getvalue())

    def test_read_does_not_create_ledger_or_parent(self) -> None:
        for command in (["list"], ["score"]):
            with self.subTest(command=command):
                with tempfile.TemporaryDirectory() as td:
                    log = Path(td) / "absent" / "f.jsonl"
                    with mock.patch.dict(os.environ, {"FORECAST_LOG": str(log)}):
                        with contextlib.redirect_stdout(io.StringIO()):
                            code = forecastlog.main(command)
                    self.assertEqual(code, 0)
                    self.assertFalse(log.exists(), "read path must not create the ledger")
                    self.assertFalse(log.parent.exists(), "read path must not create the parent")

    def test_blank_env_var_is_treated_as_unset(self) -> None:
        for blank in ("", "   "):
            with self.subTest(value=repr(blank)):
                with tempfile.TemporaryDirectory() as home:
                    with mock.patch.dict(
                        os.environ, {"FORECAST_LOG": blank, "HOME": home}, clear=False
                    ):
                        resolved = default_log_path()
                    self.assertEqual(
                        resolved, os.path.join(home, ".superforecasting", "forecasts.jsonl")
                    )

    def test_brier_defaults_to_the_same_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "f.jsonl"
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(log)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    for question, prob in (("A?", "0.7"), ("B?", "0.3")):
                        forecastlog.main(["add", "--question", question, "--prob", prob])
                    forecastlog.main(["resolve", "--id", "1", "--outcome", "1"])
                    forecastlog.main(["resolve", "--id", "2", "--outcome", "0"])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code = brier.main([])
            self.assertEqual(code, 0)
            self.assertIn("forecasts scored : 2", buf.getvalue())

    def test_brier_with_explicit_pairs_ignores_the_default_ledger(self) -> None:
        """--prob/--pairs runs must not silently fold in the whole ledger."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "f.jsonl"
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(log)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    forecastlog.main(["add", "--question", "Ledger?", "--prob", "0.7"])
                    forecastlog.main(["resolve", "--id", "1", "--outcome", "1"])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    brier.main(["--pairs", "0.6:1"])
            self.assertIn("forecasts scored : 1", buf.getvalue())

    def test_resolution_is_evaluated_at_call_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first.jsonl"
            second = Path(td) / "second.jsonl"
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(first)}):
                self.assertEqual(default_log_path(), str(first))
            with mock.patch.dict(os.environ, {"FORECAST_LOG": str(second)}):
                self.assertEqual(default_log_path(), str(second))

    def test_no_script_hardcodes_the_old_in_tree_default(self) -> None:
        for script in sorted(SCRIPTS.glob("*.py")):
            text = script.read_text()
            with self.subTest(script=script.name):
                self.assertNotIn('"..", "data"', text)
                self.assertNotIn("../data/forecasts.jsonl", text)


if __name__ == "__main__":
    unittest.main()
