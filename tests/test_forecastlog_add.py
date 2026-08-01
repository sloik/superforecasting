"""Tests for `forecastlog.py add` — registering binary and distribution forecasts."""
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
from _personas import CASSANDRA, FOX  # noqa: E402

import forecastlog  # noqa: E402
from forecast_common import fold_forecast_log  # noqa: E402


class AddBinaryTests(unittest.TestCase):
    def test_add_rejects_invalid_probability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main([
                        "--log", str(log), "add",
                        "--question", "Invalid",
                        "--prob", "1.2",
                    ])

    def test_add_writes_one_forecast_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add",
                                  "--question", "Will X ship?", "--prob", "0.35"])
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event"], "forecast")
            self.assertEqual(rows[0]["id"], 1)
            self.assertAlmostEqual(rows[0]["prob"], 0.35)
            self.assertEqual(rows[0]["status"], "open")

    def test_ids_increment_across_adds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                for i in range(3):
                    forecastlog.main(["--log", str(log), "add",
                                      "--question", f"q{i}", "--prob", "0.5"])
            self.assertEqual([r["id"] for r in fold_forecast_log(str(log))], [1, 2, 3])

    def test_optional_metadata_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "add", "--question", "Will X?", "--prob", "0.4",
                    "--resolve-by", "2026-12-31", "--rationale", "base rate 0.3",
                    "--forecaster", FOX.name, "--tags", "product,deadlines",
                    "--baseline-prob", "0.30", "--crowd-prob", "0.35",
                ])
            row = fold_forecast_log(str(log))[0]
            self.assertEqual(row["forecaster"], FOX.name)
            self.assertEqual(row["tags"], ["product", "deadlines"])
            self.assertAlmostEqual(row["baseline_prob"], 0.30)
            self.assertAlmostEqual(row["crowd_prob"], 0.35)
            self.assertEqual(row["resolve_by"], "2026-12-31")

    def test_two_forecasters_can_log_against_one_ledger(self) -> None:
        """Copilot mode logs the user and the agent separately."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                for persona in (FOX, CASSANDRA):
                    forecastlog.main(["--log", str(log), "add", "--question", "same question",
                                      "--prob", "0.6", "--forecaster", persona.name])
            names = [r["forecaster"] for r in fold_forecast_log(str(log))]
            self.assertEqual(names, [FOX.name, CASSANDRA.name])

    def test_invalid_optional_probabilities_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            for flag in ("--baseline-prob", "--crowd-prob", "--ensemble-prob",
                         "--base-rate-prob"):
                with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        forecastlog.main(["--log", str(log), "add", "--question", "q",
                                          "--prob", "0.5", flag, "1.4"])

    def test_prob_and_dist_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "add", "--question", "q",
                                      "--prob", "0.5", "--bins", "a,b", "--dist", "0.5,0.5"])

    def test_question_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "add", "--prob", "0.5"])


class AddDistributionTests(unittest.TestCase):
    def _add(self, log: Path) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            forecastlog.main([
                "--log", str(log), "add", "--question", "How much?",
                "--bins", "low,medium,high", "--dist", "0.2,0.5,0.3",
                "--resolve-by", "2026-12-31",
            ])

    def test_distribution_is_stored_with_its_bins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log)
            row = fold_forecast_log(str(log))[0]
            self.assertEqual(row["bins"], ["low", "medium", "high"])
            self.assertEqual(row["dist"], [0.2, 0.5, 0.3])
            self.assertEqual(row["question_type"], "ordered_distribution")
            self.assertIsNone(row["outcome"])

    def test_distribution_invalid_inputs_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            invalid = [
                ["--bins", "one", "--dist", "1"],
                ["--bins", "a,a", "--dist", "0.5,0.5"],
                ["--bins", "a,b", "--dist", "0.2,0.2"],
                ["--bins", "a,b", "--dist", "0.2,0.3,0.5"],
                ["--bins", "a,b", "--dist", "-0.1,1.1"],
            ]
            for args in invalid:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        forecastlog.main(["--log", str(log), "add", "--question", "bad"] + args)

    def test_dist_without_bins_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "add", "--question", "q",
                                      "--dist", "0.5,0.5"])

    def test_blank_bin_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "add", "--question", "q",
                                      "--bins", "a,,b", "--dist", "0.3,0.3,0.4"])

    def test_rounding_tolerance_is_tight(self) -> None:
        """Sums must be within 1e-9 — ordinary decimals pass, sloppy ones do not."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "ok",
                                  "--bins", "a,b,c", "--dist", "0.33,0.33,0.34"])
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "add", "--question", "bad",
                                      "--bins", "a,b", "--dist", "0.5,0.6"])


if __name__ == "__main__":
    unittest.main()
