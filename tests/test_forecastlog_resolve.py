"""Tests for `forecastlog.py update` and `resolve` — the append-only mutation path.

Both commands append events rather than rewriting rows, so the ledger keeps the
full belief trajectory. These tests are what stop a future change from silently
turning the ledger into a read-modify-write snapshot.
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

import brier  # noqa: E402
import forecastlog  # noqa: E402
from forecast_common import fold_forecast_log, ranked_probability_score  # noqa: E402


class _LedgerCase(unittest.TestCase):
    def _add(self, log: Path, prob: str, question: str = "Will X happen?") -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            forecastlog.main([
                "--log", str(log), "add",
                "--question", question, "--prob", prob,
                "--resolve-by", "2026-12-31",
            ])

    def _add_distribution(self, log: Path) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            forecastlog.main([
                "--log", str(log), "add", "--question", "How much?",
                "--bins", "low,medium,high", "--dist", "0.2,0.5,0.3",
                "--resolve-by", "2026-12-31",
            ])


class ResolveTests(_LedgerCase):
    def test_resolve_appends_event_without_rewriting_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "add",
                    "--question", "Will X happen?",
                    "--prob", "0.7",
                ])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            raw_rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r["event"] for r in raw_rows], ["forecast", "resolve"])
            folded = fold_forecast_log(str(log))
            self.assertEqual(folded[0]["status"], "resolved")
            self.assertAlmostEqual(folded[0]["brier"], 0.09)

    def test_resolving_twice_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.7")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(
                    forecastlog.main(["--log", str(log), "resolve", "--id", "1",
                                      "--outcome", "0"]), 1)
            self.assertIn("already resolved", err.getvalue().lower())

    def test_resolve_unknown_id_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.7")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(
                    forecastlog.main(["--log", str(log), "resolve", "--id", "42",
                                      "--outcome", "1"]), 1)
            self.assertIn("42", err.getvalue())

    def test_invalid_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.7")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "resolve", "--id", "1",
                                      "--outcome", "7"])

    def test_binary_forecast_requires_outcome_not_bin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.7")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "resolve", "--id", "1",
                                      "--outcome-bin", "low"])

    def test_brier_is_stored_on_the_resolve_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.8")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            event = [json.loads(line) for line in log.read_text().splitlines()][-1]
            self.assertAlmostEqual(event["brier"], 0.04)


class ResolveDistributionTests(_LedgerCase):
    def test_distribution_resolve_fold_and_scoring_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add_distribution(log)
            before = log.read_text().splitlines()
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1",
                                  "--outcome-bin", "medium"])
            raw = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(raw), len(before) + 1)
            self.assertEqual(raw[-1]["outcome_bin"], "medium")
            folded = fold_forecast_log(str(log))
            self.assertEqual(folded[0]["bins"], ["low", "medium", "high"])
            self.assertEqual(folded[0]["outcome_bin"], "medium")
            self.assertEqual(folded[0]["rps"], ranked_probability_score([0.2, 0.5, 0.3], 1))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(forecastlog.main(["--log", str(log), "score"]), 0)
            self.assertIn("Mean RPS", out.getvalue())

    def test_distribution_resolution_failures_are_clean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add_distribution(log)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "resolve", "--id", "1",
                                      "--outcome-bin", "missing"])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(forecastlog.main([
                    "--log", str(log), "resolve", "--id", "99", "--outcome-bin", "low",
                ]), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome-bin", "0"])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(forecastlog.main([
                    "--log", str(log), "resolve", "--id", "1", "--outcome-bin", "low",
                ]), 1)

    def test_outcome_bin_accepts_an_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add_distribution(log)
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome-bin", "2"])
            self.assertEqual(fold_forecast_log(str(log))[0]["outcome_bin"], "high")

    def test_out_of_range_bin_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add_distribution(log)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "resolve", "--id", "1",
                                      "--outcome-bin", "9"])

    def test_mixed_log_keeps_binary_brier_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "binary",
                                  "--prob", "0.8"])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            binary_out = io.StringIO()
            with contextlib.redirect_stdout(binary_out):
                brier.main(["--log", str(log)])
            self._add_distribution(log)
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "2",
                                  "--outcome-bin", "high"])
            mixed_out = io.StringIO()
            with contextlib.redirect_stdout(mixed_out):
                brier.main(["--log", str(log)])
            self.assertIn("Brier score      : 0.0400", binary_out.getvalue())
            self.assertIn("Brier score      : 0.0400", mixed_out.getvalue())
            self.assertIn("Mean RPS", mixed_out.getvalue())


class UpdateTests(_LedgerCase):
    def test_update_appends_event_and_shows_old_new(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            before = log.read_text().splitlines()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = forecastlog.main([
                    "--log", str(log), "update", "--id", "1", "--prob", "0.45",
                    "--note", "earnings surprise",
                ])
            self.assertEqual(code, 0)
            after = log.read_text().splitlines()
            self.assertEqual(len(after), len(before) + 1)
            self.assertEqual(after[0], before[0])
            raw_rows = [json.loads(line) for line in after]
            self.assertEqual([r["event"] for r in raw_rows], ["forecast", "update"])
            out = buf.getvalue()
            self.assertIn("0.3", out)
            self.assertIn("0.45", out)

    def test_update_rejects_invalid_probability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main([
                        "--log", str(log), "update", "--id", "1", "--prob", "1.5",
                    ])

    def test_update_missing_prob_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "update", "--id", "1"])

    def test_update_unknown_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = forecastlog.main([
                    "--log", str(log), "update", "--id", "99", "--prob", "0.5",
                ])
            self.assertEqual(code, 1)
            self.assertIn("99", err.getvalue())

    def test_update_resolved_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = forecastlog.main([
                    "--log", str(log), "update", "--id", "1", "--prob", "0.5",
                ])
            self.assertEqual(code, 1)
            self.assertIn("resolved", err.getvalue().lower())

    def test_update_of_a_distribution_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add_distribution(log)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(forecastlog.main([
                    "--log", str(log), "update", "--id", "1", "--prob", "0.5",
                ]), 1)
            self.assertIn("distribution", err.getvalue().lower())

    def test_fold_builds_prob_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.50"])
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.70"])
            folded = fold_forecast_log(str(log))
            self.assertEqual(len(folded), 1)
            snap = folded[0]
            self.assertAlmostEqual(snap["prob"], 0.70)
            hist = snap["prob_history"]
            self.assertEqual([h["prob"] for h in hist], [0.30, 0.50, 0.70])
            for h in hist:
                self.assertIn("at", h)

    def test_update_then_resolve_scores_latest_prob(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.80"])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            folded = fold_forecast_log(str(log))
            self.assertAlmostEqual(folded[0]["brier"], round((0.80 - 1) ** 2, 4))
            rows = brier.load_rows(type("Args", (), {
                "prob": None, "outcome": None, "pairs": None, "log": str(log),
            }))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["prob"], 0.80)

    def test_backward_compatible_fold_without_update_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.70")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            folded = fold_forecast_log(str(log))
            self.assertNotIn("prob_history", folded[0])
            self.assertEqual(folded[0]["status"], "resolved")
            self.assertAlmostEqual(folded[0]["brier"], 0.09)

    def test_note_is_preserved_on_the_update_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.45",
                                  "--note", "trigger fired"])
            event = [json.loads(line) for line in log.read_text().splitlines()][-1]
            self.assertEqual(event["note"], "trigger fired")


if __name__ == "__main__":
    unittest.main()
