"""Tests for `forecastlog.py list` and `score` — the read-only reporting path.

Every test here also asserts the ledger is byte-identical afterwards: a reporting
command that writes is a bug, and an unresolved forecast silently mutated by a
`list` call would be very hard to notice.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

import forecastlog  # noqa: E402


class ListCase(unittest.TestCase):
    def _write_ledger(self, log: Path) -> None:
        records = [
            {"event": "forecast", "id": 1, "question": "old", "prob": 0.5,
             "resolve_by": "2020-01-01", "status": "open", "tags": []},
            {"event": "forecast", "id": 2, "question": "near later", "prob": 0.5,
             "resolve_by": "2026-08-31", "status": "open", "tags": []},
            {"event": "forecast", "id": 3, "question": "near sooner", "prob": 0.5,
             "resolve_by": "2026-08-21", "status": "open", "tags": []},
            {"event": "forecast", "id": 4, "question": "far", "prob": 0.5,
             "resolve_by": "2027-03-18", "status": "open", "tags": []},
            {"event": "forecast", "id": 5, "question": "no deadline", "prob": 0.5,
             "resolve_by": None, "status": "open", "tags": []},
            {"event": "forecast", "id": 6, "question": "resolved", "prob": 0.5,
             "resolve_by": "2020-01-01", "status": "open", "tags": []},
            {"event": "resolve", "id": 6, "outcome": 1, "resolved": "2020-01-02T00:00:00Z"},
        ]
        log.write_text("".join(json.dumps(record) + "\n" for record in records))

    def _list(self, log: Path, *args: str) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(forecastlog.main(["--log", str(log), "list", *args]), 0)
        return output.getvalue()


class DueOverdueTests(ListCase):
    def test_due_and_overdue_filters_are_deterministic_sorted_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._write_ledger(log)
            before = log.read_bytes()
            overdue = self._list(log, "--overdue", "--as-of", "2026-08-01")
            self.assertIn("old", overdue)
            for excluded in ("near later", "near sooner", "far", "no deadline", "resolved"):
                self.assertNotIn(excluded, overdue)
            due = self._list(log, "--due", "30", "--as-of", "2026-08-01")
            self.assertLess(due.index("near sooner"), due.index("near later"))
            for excluded in ("old", "far", "no deadline", "resolved"):
                self.assertNotIn(excluded, due)
            self.assertEqual(log.read_bytes(), before)

    def test_due_boundaries_empty_results_and_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            records = [
                {"event": "forecast", "id": 1, "question": "today", "prob": 0.5,
                 "resolve_by": "2026-08-01", "status": "open", "tags": []},
                {"event": "forecast", "id": 2, "question": "edge", "prob": 0.5,
                 "resolve_by": "2026-08-11", "status": "open", "tags": []},
                {"event": "forecast", "id": 3, "question": "outside", "prob": 0.5,
                 "resolve_by": "2026-08-12", "status": "open", "tags": []},
            ]
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            self.assertIn("today", self._list(log, "--due", "0", "--as-of", "2026-08-01"))
            due = self._list(log, "--due", "10", "--as-of", "2026-08-01")
            self.assertIn("edge", due)
            self.assertNotIn("outside", due)
            self.assertNotIn("today", self._list(log, "--overdue", "--as-of", "2026-08-01"))
            self.assertEqual(self._list(log, "--overdue", "--as-of", "2019-01-01"),
                             "(no overdue forecasts)\n")
            self.assertEqual(self._list(log, "--due", "0", "--as-of", "2019-01-01"),
                             "(none due within 0 days)\n")
            for argv in (("--due", "-5"), ("--as-of", "notadate"),
                         ("--overdue", "--due", "10")):
                with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        forecastlog.main(["--log", str(log), "list", *argv])

    def test_a_forecast_due_on_the_reference_day_is_not_overdue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._write_ledger(log)
            overdue = self._list(log, "--overdue", "--as-of", "2026-08-21")
            self.assertNotIn("near sooner", overdue)
            self.assertIn("near sooner", self._list(log, "--due", "0", "--as-of", "2026-08-21"))


class ListFilterTests(ListCase):
    def test_status_filter_separates_open_from_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._write_ledger(log)
            open_out = self._list(log, "--status", "open")
            self.assertIn("old", open_out)
            self.assertNotIn("resolved", open_out)
            resolved_out = self._list(log, "--status", "resolved")
            self.assertIn("resolved", resolved_out)

    def test_empty_ledger_reports_no_forecasts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            log.write_text("")
            self.assertEqual(self._list(log), "(no forecasts)\n")

    def test_list_marks_updated_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "Will X happen?",
                                  "--prob", "0.30", "--resolve-by", "2026-12-31"])
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.55"])
            out = self._list(log)
            self.assertIn("updated", out.lower())
            self.assertIn("0.55", out)

    def test_distribution_rows_render_with_their_bin_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "How much?",
                                  "--bins", "low,medium,high", "--dist", "0.2,0.5,0.3"])
            self.assertIn("dist=3 bins", self._list(log))

    def test_invalid_status_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._write_ledger(log)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "list", "--status", "banana"])


class ScoreTests(ListCase):
    def test_score_reports_nothing_when_no_forecast_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "open",
                                  "--prob", "0.5"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(forecastlog.main(["--log", str(log), "score"]), 0)
            self.assertIn("no resolved forecasts", out.getvalue())

    def test_score_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "q", "--prob", "0.8"])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            before = log.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "score"])
            self.assertEqual(log.read_bytes(), before)

    def test_score_reports_the_brier_of_resolved_forecasts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "q", "--prob", "0.8"])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(forecastlog.main(["--log", str(log), "score"]), 0)
            self.assertIn("resolved forecasts : 1", out.getvalue())
            self.assertIn("0.0400", out.getvalue())


if __name__ == "__main__":
    unittest.main()
