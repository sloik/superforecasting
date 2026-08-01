"""Direct tests for scripts/forecast_common.py — the module every script imports.

A regression here breaks the whole toolkit at once, so each public function is
tested directly rather than incidentally through a caller.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

import forecast_common as fc  # noqa: E402


class ValidationTests(unittest.TestCase):
    def test_validate_probability_accepts_closed_interval(self) -> None:
        for value in (0.0, 0.5, 1.0):
            self.assertEqual(fc.validate_probability(value), value)

    def test_validate_probability_rejects_out_of_range(self) -> None:
        for value in (-0.001, 1.001, 2.0, -1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fc.validate_probability(value)

    def test_validate_probability_rejects_non_finite(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fc.validate_probability(value)

    def test_validate_probability_message_names_the_field(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            fc.validate_probability(1.5, "crowd_prob")
        self.assertIn("crowd_prob", str(ctx.exception))

    def test_validate_open_probability_excludes_endpoints(self) -> None:
        self.assertEqual(fc.validate_open_probability(0.5), 0.5)
        for value in (0.0, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fc.validate_open_probability(value)

    def test_validate_outcome_accepts_only_zero_and_one(self) -> None:
        self.assertEqual(fc.validate_outcome(0), 0)
        self.assertEqual(fc.validate_outcome(1), 1)
        for value in (-1, 2, 10):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fc.validate_outcome(value)

    def test_validate_positive_rejects_zero_and_negative(self) -> None:
        self.assertEqual(fc.validate_positive(0.5, "weight"), 0.5)
        for value in (0.0, -1.0, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fc.validate_positive(value, "weight")

    def test_validate_non_negative_allows_zero_but_not_negative(self) -> None:
        self.assertEqual(fc.validate_non_negative(0.0, "days"), 0.0)
        self.assertEqual(fc.validate_non_negative(3.0, "days"), 3.0)
        for value in (-0.001, float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fc.validate_non_negative(value, "days")


class BrierScoreTests(unittest.TestCase):
    def test_perfect_and_worst_case(self) -> None:
        self.assertEqual(fc.brier_score(1.0, 1), 0.0)
        self.assertEqual(fc.brier_score(0.0, 0), 0.0)
        self.assertEqual(fc.brier_score(0.0, 1), 1.0)
        self.assertEqual(fc.brier_score(1.0, 0), 1.0)

    def test_coin_flip_is_quarter(self) -> None:
        self.assertEqual(fc.brier_score(0.5, 1), 0.25)
        self.assertEqual(fc.brier_score(0.5, 0), 0.25)

    def test_is_symmetric_about_the_outcome(self) -> None:
        self.assertAlmostEqual(fc.brier_score(0.7, 1), fc.brier_score(0.3, 0))

    def test_validates_its_inputs(self) -> None:
        with self.assertRaises(ValueError):
            fc.brier_score(1.5, 1)
        with self.assertRaises(ValueError):
            fc.brier_score(0.5, 2)


class RankedProbabilityScoreTests(unittest.TestCase):
    def test_perfect_forecast_scores_zero(self) -> None:
        self.assertEqual(fc.ranked_probability_score([0.0, 0.0, 1.0], 2), 0.0)

    def test_near_miss_beats_far_miss(self) -> None:
        near = fc.ranked_probability_score([0.0, 1.0, 0.0], 2)
        far = fc.ranked_probability_score([1.0, 0.0, 0.0], 2)
        self.assertLess(near, far)

    def test_two_bins_reduce_to_brier(self) -> None:
        for prob, outcome in ((0.2, 0), (0.2, 1), (0.8, 0), (0.8, 1)):
            with self.subTest(prob=prob, outcome=outcome):
                self.assertEqual(fc.ranked_probability_score([1 - prob, prob], outcome),
                                 fc.brier_score(prob, outcome))

    def test_rejects_too_few_bins(self) -> None:
        with self.assertRaises(ValueError):
            fc.ranked_probability_score([1.0], 0)

    def test_rejects_out_of_range_outcome_index(self) -> None:
        for index in (-1, 3):
            with self.subTest(index=index), self.assertRaises(ValueError):
                fc.ranked_probability_score([0.2, 0.5, 0.3], index)

    def test_rejects_distribution_that_does_not_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            fc.ranked_probability_score([0.2, 0.2, 0.2], 1)


class ParseDatetimeTests(unittest.TestCase):
    def test_none_and_empty_return_none(self) -> None:
        self.assertIsNone(fc.parse_datetime(None))
        self.assertIsNone(fc.parse_datetime(""))

    def test_bare_date_is_midnight_utc(self) -> None:
        parsed = fc.parse_datetime("2026-08-21")
        self.assertEqual(parsed, datetime(2026, 8, 21, tzinfo=timezone.utc))

    def test_trailing_z_is_utc(self) -> None:
        self.assertEqual(fc.parse_datetime("2026-08-21T12:00:00Z"),
                         datetime(2026, 8, 21, 12, tzinfo=timezone.utc))

    def test_naive_timestamp_is_assumed_utc(self) -> None:
        self.assertEqual(fc.parse_datetime("2026-08-21T12:00:00"),
                         datetime(2026, 8, 21, 12, tzinfo=timezone.utc))

    def test_offset_is_normalised_to_utc(self) -> None:
        self.assertEqual(fc.parse_datetime("2026-08-21T14:00:00+02:00"),
                         datetime(2026, 8, 21, 12, tzinfo=timezone.utc))

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        self.assertEqual(fc.parse_datetime("  2026-08-21  "),
                         datetime(2026, 8, 21, tzinfo=timezone.utc))

    def test_unparseable_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            fc.parse_datetime("notadate")


class AgeDaysTests(unittest.TestCase):
    def test_counts_whole_days(self) -> None:
        as_of = fc.parse_datetime("2026-08-21")
        self.assertAlmostEqual(fc.age_days("2026-08-11", as_of), 10.0)

    def test_fractional_days(self) -> None:
        as_of = fc.parse_datetime("2026-08-21T12:00:00Z")
        self.assertAlmostEqual(fc.age_days("2026-08-21T00:00:00Z", as_of), 0.5)

    def test_future_stamp_clamps_to_zero(self) -> None:
        as_of = fc.parse_datetime("2026-08-21")
        self.assertEqual(fc.age_days("2026-09-01", as_of), 0.0)

    def test_unparseable_stamp_is_zero_age(self) -> None:
        as_of = fc.parse_datetime("2026-08-21")
        self.assertEqual(fc.age_days("", as_of), 0.0)


class TimeWeightedBrierTests(unittest.TestCase):
    def test_duration_weights_each_held_probability(self) -> None:
        snapshot = {
            "prob": 0.80, "outcome": 1, "resolved": "2026-01-11T00:00:00Z",
            "prob_history": [
                {"prob": 0.30, "at": "2026-01-01T00:00:00Z"},
                {"prob": 0.80, "at": "2026-01-03T00:00:00Z"},
            ],
        }
        self.assertAlmostEqual(fc.time_weighted_brier_score(snapshot), 0.13, places=9)

    def test_single_entry_history_degrades_to_standard_brier(self) -> None:
        snapshot = {
            "prob": 0.80, "outcome": 1, "resolved": "2026-01-11T00:00:00Z",
            "prob_history": [{"prob": 0.80, "at": "2026-01-01T00:00:00Z"}],
        }
        self.assertEqual(fc.time_weighted_brier_score(snapshot), fc.brier_score(0.80, 1))

    def test_zero_total_duration_degrades_to_standard_brier(self) -> None:
        snapshot = {
            "prob": 0.80, "outcome": 1, "resolved": "2026-01-01T00:00:00Z",
            "prob_history": [
                {"prob": 0.30, "at": "2026-01-01T00:00:00Z"},
                {"prob": 0.80, "at": "2026-01-01T00:00:00Z"},
            ],
        }
        self.assertEqual(fc.time_weighted_brier_score(snapshot), fc.brier_score(0.80, 1))

    def test_malformed_history_entry_degrades_rather_than_raises(self) -> None:
        snapshot = {
            "prob": 0.80, "outcome": 1, "resolved": "2026-01-11T00:00:00Z",
            "prob_history": [
                {"at": "2026-01-01T00:00:00Z"},  # missing prob
                {"prob": 0.80, "at": "2026-01-03T00:00:00Z"},
            ],
        }
        self.assertEqual(fc.time_weighted_brier_score(snapshot), fc.brier_score(0.80, 1))


class JsonlIoTests(unittest.TestCase):
    def test_read_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(fc.read_jsonl(str(Path(td) / "absent.jsonl")), [])

    def test_append_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "deep" / "nested" / "log.jsonl"
            fc.append_jsonl(str(path), {"id": 1})
            self.assertTrue(path.exists())
            self.assertEqual(fc.read_jsonl(str(path)), [{"id": 1}])

    def test_append_is_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            fc.append_jsonl(str(path), {"id": 1})
            first = path.read_text()
            fc.append_jsonl(str(path), {"id": 2})
            self.assertTrue(path.read_text().startswith(first))
            self.assertEqual([r["id"] for r in fc.read_jsonl(str(path))], [1, 2])

    def test_blank_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            path.write_text('{"id": 1}\n\n   \n{"id": 2}\n')
            self.assertEqual([r["id"] for r in fc.read_jsonl(str(path))], [1, 2])

    def test_malformed_line_raises_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            path.write_text('{"id": 1}\nnot json\n')
            with self.assertRaises(ValueError) as ctx:
                fc.read_jsonl(str(path))
            self.assertIn(":2:", str(ctx.exception))

    def test_non_ascii_survives_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            fc.append_jsonl(str(path), {"question": "Czy zdarzy się X?"})
            self.assertEqual(fc.read_jsonl(str(path))[0]["question"], "Czy zdarzy się X?")


class FoldForecastLogTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_preserves_first_seen_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [
                {"event": "forecast", "id": 7, "prob": 0.5},
                {"event": "forecast", "id": 3, "prob": 0.5},
                {"event": "forecast", "id": 5, "prob": 0.5},
            ])
            self.assertEqual([r["id"] for r in fc.fold_forecast_log(str(path))], [7, 3, 5])

    def test_resolve_for_unknown_id_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [
                {"event": "forecast", "id": 1, "prob": 0.5},
                {"event": "resolve", "id": 99, "outcome": 1},
            ])
            folded = fc.fold_forecast_log(str(path))
            self.assertEqual(len(folded), 1)
            self.assertEqual(folded[0]["status"], "open")

    def test_update_before_forecast_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [
                {"event": "update", "id": 1, "prob": 0.9},
                {"event": "forecast", "id": 1, "prob": 0.5},
            ])
            self.assertAlmostEqual(fc.fold_forecast_log(str(path))[0]["prob"], 0.5)

    def test_rows_without_id_or_prob_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [
                {"event": "forecast", "prob": 0.5},
                {"event": "forecast", "id": 2},
                {"event": "forecast", "id": 3, "prob": 0.5},
            ])
            self.assertEqual([r["id"] for r in fc.fold_forecast_log(str(path))], [3])

    def test_legacy_row_with_inline_outcome_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [{"event": "forecast", "id": 1, "prob": 0.8, "outcome": 1}])
            folded = fc.fold_forecast_log(str(path))
            self.assertEqual(folded[0]["status"], "resolved")
            self.assertAlmostEqual(folded[0]["brier"], 0.04)

    def test_incoherent_distribution_row_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [
                {"event": "forecast", "id": 1, "bins": ["a", "b"], "dist": [0.2, 0.2]},
                {"event": "forecast", "id": 2, "bins": ["a"], "dist": [1.0]},
                {"event": "forecast", "id": 3, "bins": ["a", "b"], "dist": [0.4, 0.6]},
            ])
            self.assertEqual([r["id"] for r in fc.fold_forecast_log(str(path))], [3])

    def test_distribution_resolve_with_unknown_bin_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            self._write(path, [
                {"event": "forecast", "id": 1, "bins": ["a", "b"], "dist": [0.4, 0.6]},
                {"event": "resolve", "id": 1, "outcome_bin": "zzz"},
            ])
            self.assertEqual(fc.fold_forecast_log(str(path))[0]["status"], "open")

    def test_empty_log_folds_to_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.jsonl"
            path.write_text("")
            self.assertEqual(fc.fold_forecast_log(str(path)), [])


class DefaultLogPathTests(unittest.TestCase):
    """Resolution-order specifics live in test_forecastlog_paths.py; this is the contract."""

    def test_returns_an_absolute_path(self) -> None:
        self.assertTrue(Path(fc.default_log_path()).is_absolute())

    def test_never_points_inside_the_skill_directory(self) -> None:
        resolved = Path(fc.default_log_path()).resolve()
        skill_root = Path(fc.__file__).resolve().parents[1]
        self.assertFalse(str(resolved).startswith(str(skill_root)),
                         "the ledger must never default inside the skill checkout")

    def test_resolving_creates_nothing(self) -> None:
        path = Path(fc.default_log_path())
        existed = path.exists()
        fc.default_log_path()
        self.assertEqual(path.exists(), existed)


class ModuleContractTests(unittest.TestCase):
    def test_math_import_is_used_for_finiteness_checks(self) -> None:
        """Guards the isfinite path that rejects nan/inf across validators."""
        self.assertTrue(math.isfinite(1.0))
        with self.assertRaises(ValueError):
            fc.validate_non_negative(float("nan"), "x")


if __name__ == "__main__":
    unittest.main()
