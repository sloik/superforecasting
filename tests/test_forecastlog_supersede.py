"""Tests for retiring withdrawn forecasts without inventing outcomes or beliefs."""
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
from forecast_common import fold_forecast_log  # noqa: E402


class SupersedeCase(unittest.TestCase):
    def _add(self, log: Path, question: str, prob: str = "0.70",
             resolve_by: str = "2020-01-01") -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(forecastlog.main([
                "--log", str(log), "add", "--question", question,
                "--prob", prob, "--resolve-by", resolve_by,
            ]), 0)

    def _supersede(self, log: Path, forecast_id: int, reason: str,
                   replaced_by: int | None = None) -> tuple[int, str]:
        argv = [
            "--log", str(log), "supersede", "--id", str(forecast_id),
            "--reason", reason,
        ]
        if replaced_by is not None:
            argv.extend(["--replaced-by", str(replaced_by)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = forecastlog.main(argv)
        return code, output.getvalue()


class SupersedeAppendTests(SupersedeCase):
    def test_supersede_appends_event_without_rewriting_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "withdrawn")
            before = log.read_bytes()

            code, output = self._supersede(
                log, 1, "withdrawn measurement; bookkeeping move to 0.55", replaced_by=26,
            )

            self.assertEqual(code, 0)
            self.assertIn("superseded #1", output)
            after = log.read_bytes()
            self.assertTrue(after.startswith(before))
            raw = [json.loads(line) for line in after.splitlines()]
            self.assertEqual(len(raw), 2)
            self.assertEqual(raw[0], json.loads(before))
            self.assertEqual(raw[1]["event"], "supersede")
            self.assertEqual(raw[1]["schema_version"], 2)
            self.assertEqual(raw[1]["status"], "superseded")
            self.assertEqual(raw[1]["reason"], "withdrawn measurement; bookkeeping move to 0.55")
            self.assertEqual(raw[1]["replaced_by"], 26)
            self.assertIn("superseded", raw[1])
            folded = fold_forecast_log(str(log))[0]
            self.assertEqual(folded["status"], "superseded")
            self.assertEqual(folded["reason"], raw[1]["reason"])
            self.assertEqual(folded["replaced_by"], 26)

    def test_supersede_requires_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "withdrawn")
            before = log.read_bytes()
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as raised:
                forecastlog.main(["--log", str(log), "supersede", "--id", "1"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--reason", err.getvalue())
            self.assertEqual(log.read_bytes(), before)

    def test_supersede_preserves_probability_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "withdrawn", "0.70")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "update", "--id", "1", "--prob", "0.62",
                    "--note", "genuine evidence update",
                ])
            before = fold_forecast_log(str(log))[0]
            self._supersede(log, 1, "question withdrawn")
            after = fold_forecast_log(str(log))[0]
            self.assertEqual(after["prob"], before["prob"])
            self.assertEqual(after["prob_history"], before["prob_history"])
            self.assertNotIn("prob", json.loads(log.read_text().splitlines()[-1]))

    def test_distribution_forecast_can_be_superseded_without_an_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "add", "--question", "withdrawn range",
                    "--bins", "low,high", "--dist", "0.4,0.6",
                ])
            self.assertEqual(self._supersede(log, 1, "withdrawn range question")[0], 0)
            folded = fold_forecast_log(str(log))[0]
            self.assertEqual(folded["status"], "superseded")
            self.assertIsNone(folded["outcome"])
            self.assertNotIn("rps", folded)


class SupersedeScoringAndListingTests(SupersedeCase):
    def test_superseded_forecast_is_excluded_from_all_brier_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "resolved", "0.80")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "resolve", "--id", "1", "--outcome", "1",
                ])
            self._add(log, "superseded", "0.10")
            self._supersede(log, 2, "question withdrawn")

            quick = io.StringIO()
            with contextlib.redirect_stdout(quick):
                self.assertEqual(forecastlog.main(["--log", str(log), "score"]), 0)
            self.assertIn("resolved forecasts : 1", quick.getvalue())
            self.assertIn("Brier score        : 0.0400", quick.getvalue())

            full = io.StringIO()
            with contextlib.redirect_stdout(full):
                self.assertEqual(brier.main([
                    "--log", str(log), "--baseline-prob", "0.5", "--time-weighted",
                ]), 0)
            self.assertIn("forecasts scored : 1", full.getvalue())
            self.assertIn("Brier score      : 0.0400", full.getvalue())
            self.assertIn("baseline Brier    : 0.2500", full.getvalue())

    def test_superseded_forecast_is_not_overdue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "withdrawn overdue")
            self._supersede(log, 1, "question withdrawn")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(forecastlog.main([
                    "--log", str(log), "list", "--overdue", "--as-of", "2026-08-01",
                ]), 0)
            self.assertEqual(output.getvalue(), "(no overdue forecasts)\n")

    def test_list_distinguishes_and_filters_superseded_forecasts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "open question")
            self._add(log, "resolved question")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "resolve", "--id", "2", "--outcome", "1",
                ])
            self._add(log, "superseded question")
            self._supersede(log, 3, "withdrawn")

            plain = io.StringIO()
            with contextlib.redirect_stdout(plain):
                forecastlog.main(["--log", str(log), "list"])
            self.assertIn("superseded", plain.getvalue())
            self.assertIn("superseded question", plain.getvalue())

            filtered = io.StringIO()
            with contextlib.redirect_stdout(filtered):
                forecastlog.main(["--log", str(log), "list", "--status", "superseded"])
            self.assertIn("superseded question", filtered.getvalue())
            self.assertNotIn("open question", filtered.getvalue())
            self.assertNotIn("resolved question", filtered.getvalue())


class SupersedeTerminalTests(SupersedeCase):
    def test_resolve_refuses_superseded_forecast_and_names_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "withdrawn")
            reason = "truth conditions were private information"
            self._supersede(log, 1, reason)
            before = log.read_bytes()
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = forecastlog.main([
                    "--log", str(log), "resolve", "--id", "1", "--outcome", "1",
                ])
            self.assertEqual(code, 1)
            self.assertIn(reason, err.getvalue())
            self.assertEqual(log.read_bytes(), before)

    def test_update_and_repeat_supersede_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "withdrawn")
            self._supersede(log, 1, "question withdrawn")
            before = log.read_bytes()
            for argv in (["update", "--id", "1", "--prob", "0.25"],
                         ["supersede", "--id", "1", "--reason", "again"]):
                with self.subTest(command=argv[0]):
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err):
                        self.assertEqual(forecastlog.main(["--log", str(log), *argv]), 1)
                    self.assertIn("question withdrawn", err.getvalue())
                    self.assertEqual(log.read_bytes(), before)


class SupersedeBackwardCompatibilityTests(unittest.TestCase):
    def test_pre_change_ledger_loads_and_scores_identically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            records = [
                {"id": 1, "question": "legacy open", "prob": 0.25, "outcome": None,
                 "status": "open", "tags": []},
                {"id": 2, "question": "legacy resolved", "prob": 0.70, "outcome": 1,
                 "status": "resolved", "tags": []},
            ]
            log.write_text("".join(json.dumps(row) + "\n" for row in records))
            before = log.read_bytes()
            folded = fold_forecast_log(str(log))
            self.assertEqual([row["status"] for row in folded], ["open", "resolved"])
            self.assertAlmostEqual(folded[1]["brier"], 0.09)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(brier.main(["--log", str(log)]), 0)
            self.assertIn("forecasts scored : 1", output.getvalue())
            self.assertIn("Brier score      : 0.0900", output.getvalue())
            self.assertEqual(log.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
