from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregate  # noqa: E402
import bayes  # noqa: E402
import baserate  # noqa: E402
import brier  # noqa: E402
import forecastlog  # noqa: E402
import montecarlo  # noqa: E402
from forecast_common import (  # noqa: E402
    brier_score, default_log_path, fold_forecast_log, ranked_probability_score,
    time_weighted_brier_score,
)

# --- Production-ledger immutability guard -----------------------------------
# The default log path resolves to a REAL user file. An injectable seam that
# defaults to production is opt-in safety: it protects only the tests that
# remember to inject. So guard it at module scope, where it covers every test in
# this file including ones written later. Compare (sha256, mtime_ns) rather than
# content alone — an atomic writer can replace the inode with identical bytes.
_GUARDED_PATHS = (
    Path(os.path.expanduser("~/.superforecasting/forecasts.jsonl")),
    ROOT / "data" / "forecasts.jsonl",
)
_guard_snapshots: dict[Path, tuple[bytes, str, int] | None] = {}


def setUpModule() -> None:
    for path in _GUARDED_PATHS:
        if path.exists():
            payload = path.read_bytes()
            _guard_snapshots[path] = (
                payload, hashlib.sha256(payload).hexdigest(), path.stat().st_mtime_ns,
            )
        else:
            _guard_snapshots[path] = None


def tearDownModule() -> None:
    violations: list[str] = []
    for path in _GUARDED_PATHS:
        before = _guard_snapshots.get(path)
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
        now = path.read_bytes()
        if hashlib.sha256(now).hexdigest() != digest:
            violations.append(f"{path} was MODIFIED by the test suite")
            path.write_bytes(payload)
            os.utime(path, ns=(mtime_ns, mtime_ns))
        elif path.stat().st_mtime_ns != mtime_ns:
            violations.append(f"{path} was REWRITTEN (same bytes, new mtime) by the test suite")
            os.utime(path, ns=(mtime_ns, mtime_ns))
    if violations:
        raise AssertionError(
            "test suite touched a production ledger (restored): " + "; ".join(violations)
            + " — set FORECAST_LOG or pass --log to a tmpdir in the offending test"
        )


class AggregateTests(unittest.TestCase):
    def test_geo_mean_of_odds(self) -> None:
        pooled = aggregate.geomean_of_odds([0.6, 0.7, 0.55, 0.8])
        self.assertAlmostEqual(pooled, 0.6704, places=4)

    def test_invalid_weights_fail_cleanly(self) -> None:
        with self.assertRaises(ValueError):
            aggregate.geomean_of_odds([0.4, 0.6], [-1, 1])

    def test_extremize_less_than_one_fails(self) -> None:
        with self.assertRaises(ValueError):
            aggregate.extremize(0.6, 0.5)

    def test_track_log_requires_forecaster_cli(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                aggregate.main(["0.6", "--track-log", "missing.jsonl"])

    def test_track_record_weights_are_smoothed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            for rec in [
                {"event": "forecast", "id": 1, "prob": 0.8, "forecaster": "good", "outcome": 1},
                {"event": "forecast", "id": 2, "prob": 0.8, "forecaster": "bad", "outcome": 0},
            ]:
                log.write_text(log.read_text() + json.dumps(rec) + "\n" if log.exists() else json.dumps(rec) + "\n")
            weights, stats = aggregate.track_record_weights(str(log), ["good", "bad"])
            self.assertGreater(weights[0], weights[1])
            self.assertEqual(stats["good"][0], 1)
            self.assertEqual(stats["bad"][0], 1)

    def test_time_decay_downweights_older_forecasts(self) -> None:
        as_of = aggregate.parse_datetime("2026-07-15")
        weights = aggregate.time_decay_weights(["2026-07-01", "2026-07-15"], as_of, 14)
        self.assertAlmostEqual(weights[0], 0.5, places=4)
        self.assertAlmostEqual(weights[1], 1.0, places=4)


class BrierTests(unittest.TestCase):
    def test_invalid_probability_rejected(self) -> None:
        with self.assertRaises(ValueError):
            brier.brier([(1.2, 1)])

    def test_baseline_skill_score_data(self) -> None:
        rows = [{"prob": 0.7, "outcome": 1}, {"prob": 0.3, "outcome": 0}]
        args = type("Args", (), {"baseline_prob": 0.5, "baseline_field": None})
        label, pairs = brier.baseline_pairs(rows, args)
        self.assertEqual(label, "constant 0.500")
        self.assertEqual(pairs, [(0.5, 1), (0.5, 0)])

    def test_event_log_is_folded_for_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main([
                    "--log", str(log), "add",
                    "--question", "Will X happen?",
                    "--prob", "0.7",
                    "--resolve-by", "2026-12-31",
                    "--baseline-prob", "0.5",
                ])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            rows = brier.load_rows(type("Args", (), {
                "prob": None,
                "outcome": None,
                "pairs": None,
                "log": str(log),
            }))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["outcome"], 1)
            self.assertEqual(rows[0]["baseline_prob"], 0.5)

    def test_time_weighted_brier_penalizes_earlier_wrong_probability(self) -> None:
        snapshot = {
            "prob": 0.80,
            "outcome": 1,
            "resolved": "2026-01-11T00:00:00Z",
            "prob_history": [
                {"prob": 0.30, "at": "2026-01-01T00:00:00Z"},
                {"prob": 0.80, "at": "2026-01-03T00:00:00Z"},
            ],
        }
        self.assertAlmostEqual(time_weighted_brier_score(snapshot), 0.13, places=9)

    def test_time_weighted_brier_degrades_for_unusable_trajectories(self) -> None:
        standard = (0.80 - 1) ** 2
        for snapshot in [
            {"prob": 0.80, "outcome": 1, "resolved": "2026-01-11T00:00:00Z"},
            {"prob": 0.80, "outcome": 1, "resolved": "not-a-date", "prob_history": [
                {"prob": 0.30, "at": "2026-01-01T00:00:00Z"},
                {"prob": 0.80, "at": "2026-01-03T00:00:00Z"},
            ]},
            {"prob": 0.80, "outcome": 1, "resolved": "2026-01-11T00:00:00Z", "prob_history": [
                {"prob": 0.30, "at": None},
                {"prob": 0.80, "at": "2026-01-03T00:00:00Z"},
            ]},
            {"prob": 0.80, "outcome": 1, "resolved": "2026-01-01T00:00:00Z", "prob_history": [
                {"prob": 0.30, "at": "2026-01-02T00:00:00Z"},
                {"prob": 0.80, "at": "2026-01-01T00:00:00Z"},
            ]},
        ]:
            self.assertEqual(time_weighted_brier_score(snapshot), standard)

    def test_time_weighted_log_output_and_ledger_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            log.write_text("\n".join(json.dumps(record) for record in [
                {"event": "forecast", "id": 1, "prob": 0.30, "created": "2026-01-01T00:00:00Z"},
                {"event": "update", "id": 1, "prob": 0.80, "updated": "2026-01-03T00:00:00Z"},
                {"event": "resolve", "id": 1, "outcome": 1, "resolved": "2026-01-11T00:00:00Z"},
                {"event": "forecast", "id": 2, "prob": 0.60, "created": "2026-01-01T00:00:00Z"},
                {"event": "resolve", "id": 2, "outcome": 1, "resolved": "2026-01-11T00:00:00Z"},
            ]) + "\n")
            before = log.read_bytes()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(brier.main(["--log", str(log), "--time-weighted"]), 0)
            self.assertEqual(log.read_bytes(), before)
            self.assertIn("Brier score      : 0.1000", out.getvalue())
            self.assertIn("Time-weighted Brier: 0.1450", out.getvalue())

    def test_time_weighted_requires_log(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit):
                brier.main(["--time-weighted", "--pairs", "0.7:1"])
        self.assertIn("requires --log", err.getvalue())


class ForecastLogTests(unittest.TestCase):
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


class ForecastDueListingTests(unittest.TestCase):
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


class ForecastUpdateTests(unittest.TestCase):
    def _add(self, log: Path, prob: str, question: str = "Will X happen?") -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            forecastlog.main([
                "--log", str(log), "add",
                "--question", question, "--prob", prob,
                "--resolve-by", "2026-12-31",
            ])

    def test_update_appends_event_and_shows_old_new(self) -> None:  # AC1
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
            # Ledger grew by exactly one line; the prior line is untouched.
            self.assertEqual(len(after), len(before) + 1)
            self.assertEqual(after[0], before[0])
            raw_rows = [json.loads(line) for line in after]
            self.assertEqual([r["event"] for r in raw_rows], ["forecast", "update"])
            out = buf.getvalue()
            self.assertIn("0.3", out)
            self.assertIn("0.45", out)

    def test_update_rejects_invalid_probability(self) -> None:  # AC2
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main([
                        "--log", str(log), "update", "--id", "1", "--prob", "1.5",
                    ])

    def test_update_missing_prob_fails(self) -> None:  # AC2
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "update", "--id", "1"])

    def test_update_unknown_id_fails(self) -> None:  # AC3
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

    def test_update_resolved_id_fails(self) -> None:  # AC3
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

    def test_fold_builds_prob_history(self) -> None:  # AC4
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

    def test_update_then_resolve_scores_latest_prob(self) -> None:  # AC5
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.80"])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            folded = fold_forecast_log(str(log))
            # Brier is against the latest updated prob 0.8, not the original 0.3.
            self.assertAlmostEqual(folded[0]["brier"], round((0.80 - 1) ** 2, 4))
            rows = brier.load_rows(type("Args", (), {
                "prob": None, "outcome": None, "pairs": None, "log": str(log),
            }))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["prob"], 0.80)

    def test_backward_compatible_fold_without_update_events(self) -> None:  # AC6
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.70")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            folded = fold_forecast_log(str(log))
            # No update events → no prob_history key leaks onto the snapshot.
            self.assertNotIn("prob_history", folded[0])
            self.assertEqual(folded[0]["status"], "resolved")
            self.assertAlmostEqual(folded[0]["brier"], 0.09)

    def test_list_marks_updated_forecast(self) -> None:  # AC7
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log, "0.30")
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "update", "--id", "1", "--prob", "0.55"])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                forecastlog.main(["--log", str(log), "list"])
            out = buf.getvalue()
            self.assertIn("updated", out.lower())
            self.assertIn("0.55", out)


class OrderedDistributionTests(unittest.TestCase):
    def _add(self, log: Path) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            forecastlog.main([
                "--log", str(log), "add", "--question", "How much?",
                "--bins", "low,medium,high", "--dist", "0.2,0.5,0.3",
                "--resolve-by", "2026-12-31",
            ])

    def test_ranked_probability_score_properties_and_binary_reduction(self) -> None:
        self.assertEqual(ranked_probability_score([0.0, 1.0, 0.0], 1), 0.0)
        self.assertLess(ranked_probability_score([0.0, 1.0, 0.0], 0),
                        ranked_probability_score([0.0, 0.0, 1.0], 0))
        self.assertAlmostEqual(ranked_probability_score([0.2, 0.5, 0.3], 0),
                               ranked_probability_score([0.3, 0.5, 0.2], 2))
        for prob, outcome in ((0.2, 0), (0.2, 1), (0.8, 0), (0.8, 1)):
            self.assertEqual(ranked_probability_score([1 - prob, prob], outcome),
                             brier_score(prob, outcome))

    def test_distribution_add_resolve_fold_and_scoring_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            self._add(log)
            before = log.read_text().splitlines()
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome-bin", "medium"])
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

    def test_distribution_invalid_inputs_and_resolution_fail_cleanly(self) -> None:
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
            self._add(log)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome-bin", "missing"])
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

    def test_mixed_log_keeps_binary_brier_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "binary", "--prob", "0.8"])
                forecastlog.main(["--log", str(log), "resolve", "--id", "1", "--outcome", "1"])
            binary_out = io.StringIO()
            with contextlib.redirect_stdout(binary_out):
                brier.main(["--log", str(log)])
            self._add(log)
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "resolve", "--id", "2", "--outcome-bin", "high"])
            mixed_out = io.StringIO()
            with contextlib.redirect_stdout(mixed_out):
                brier.main(["--log", str(log)])
            self.assertIn("Brier score      : 0.0400", binary_out.getvalue())
            self.assertIn("Brier score      : 0.0400", mixed_out.getvalue())
            self.assertIn("Mean RPS", mixed_out.getvalue())


class BaseRateTests(unittest.TestCase):
    def test_wilson_interval_matches_reference_bounds(self) -> None:
        lo, hi = baserate.wilson(3, 11)
        self.assertLessEqual(0.0, lo)
        self.assertLessEqual(lo, 3 / 11)
        self.assertLessEqual(3 / 11, hi)
        self.assertLessEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.0975, places=4)
        self.assertAlmostEqual(hi, 0.5657, places=4)

    def test_wilson_zero_total_returns_full_interval(self) -> None:
        self.assertEqual(baserate.wilson(0, 0), (0.0, 1.0))

    def test_main_prints_smoothed_anchors(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(baserate.main(["--hits", "3", "--total", "11"]), 0)
        self.assertIn("0.3077", out.getvalue())
        self.assertIn("0.2917", out.getvalue())

    def test_main_rejects_out_of_range_counts(self) -> None:
        for argv in (["--hits", "5", "--total", "3"], ["--hits", "-1", "--total", "3"],
                     ["--hits", "0", "--total", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    baserate.main(argv)

    def test_checklist_is_printed_for_flag_and_no_args(self) -> None:
        for argv in (["--checklist"], []):
            out = io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stdout(out):
                self.assertEqual(baserate.main(argv), 0)
            self.assertIn("REFERENCE-CLASS CHECKLIST", out.getvalue())

    def test_small_sample_note_is_conditional(self) -> None:
        small = io.StringIO()
        with contextlib.redirect_stdout(small):
            self.assertEqual(baserate.main(["--hits", "1", "--total", "5"]), 0)
        self.assertIn("n < 10", small.getvalue())

        large = io.StringIO()
        with contextlib.redirect_stdout(large):
            self.assertEqual(baserate.main(["--hits", "3", "--total", "20"]), 0)
        self.assertNotIn("n < 10", large.getvalue())


class BayesTests(unittest.TestCase):
    def test_prior_must_be_open_probability(self) -> None:
        with self.assertRaises(ValueError):
            bayes.update(0.0, [2.0])

    def test_probability_odds_round_trip_and_open_validation(self) -> None:
        for probability in (0.1, 0.5, 0.9):
            with self.subTest(probability=probability):
                self.assertAlmostEqual(
                    bayes.odds_to_p(bayes.p_to_odds(probability)), probability, places=6
                )
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                bayes.p_to_odds(probability)

    def test_update_applies_multiple_likelihood_ratios(self) -> None:
        self.assertAlmostEqual(bayes.update(0.5, [2.0, 2.0, 0.5]), 2 / 3, places=4)

    def test_conditional_probabilities_convert_to_likelihood_ratio(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                bayes.main(["--prior", "0.10", "--pe-h", "0.80", "--pe-nh", "0.20"]), 0
            )
        self.assertIn("combined LR : 4.0000", out.getvalue())
        self.assertIn("posterior   : 0.3077", out.getvalue())

    def test_main_rejects_invalid_or_missing_evidence(self) -> None:
        invalid_argvs = (
            ["--prior", "0.10", "--pe-h", "0.80"],
            ["--prior", "0.10", "--pe-h", "0", "--pe-nh", "0.20"],
            ["--prior", "0.10"],
        )
        for argv in invalid_argvs:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bayes.main(argv)

    def test_steps_print_running_update_trace(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                bayes.main([
                    "--prior", "0.50", "--lr", "2", "--lr", "2", "--lr", "0.5", "--steps",
                ]),
                0,
            )
        self.assertIn("after evidence 1", out.getvalue())
        self.assertIn("POSTERIOR", out.getvalue())


class MonteCarloTests(unittest.TestCase):
    def test_happy_path_is_reproducible(self) -> None:
        model = {
            "trials": 100,
            "vars": {"x": {"dist": "point", "value": 10}},
            "expr": "x * 2",
            "threshold": 15,
            "direction": ">",
        }
        out = montecarlo.run(model)
        self.assertEqual(out["p50"], 20)
        self.assertEqual(out["prob"], 1.0)

    def test_unsafe_expression_is_rejected(self) -> None:
        model = {
            "trials": 1,
            "vars": {"x": {"dist": "point", "value": 10}},
            "expr": "__import__('os').system('echo bad')",
        }
        with self.assertRaises(ValueError):
            montecarlo.run(model)

    def test_bad_distribution_params_are_rejected(self) -> None:
        model = {
            "trials": 1,
            "vars": {"x": {"dist": "bernoulli", "p": 1.2}},
            "expr": "x",
        }
        with self.assertRaises(ValueError):
            montecarlo.run(model)


class CliSmokeTests(unittest.TestCase):
    def test_brier_cli_outputs_skill_score(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = brier.main(["--pairs", "0.7:1", "0.3:0", "--baseline-prob", "0.5"])
        self.assertEqual(code, 0)
        self.assertIn("Brier Skill Score", buf.getvalue())


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
