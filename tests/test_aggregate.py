"""Tests for scripts/aggregate.py — pooling many forecasts into one."""
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
from _personas import COINFLIPPER, HEDGEHOG, ORACLE, write_ledger  # noqa: E402

import aggregate  # noqa: E402


class LogitTests(unittest.TestCase):
    def test_logit_round_trips_through_inv_logit(self) -> None:
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            with self.subTest(p=p):
                self.assertAlmostEqual(aggregate.inv_logit(aggregate.logit(p)), p, places=9)

    def test_logit_of_half_is_zero(self) -> None:
        self.assertAlmostEqual(aggregate.logit(0.5), 0.0)

    def test_clip_keeps_values_off_the_endpoints(self) -> None:
        clipped = aggregate._clip([0.0, 1.0, 0.5])
        self.assertGreater(clipped[0], 0.0)
        self.assertLess(clipped[1], 1.0)
        self.assertEqual(clipped[2], 0.5)


class GeoMeanOfOddsTests(unittest.TestCase):
    def test_geo_mean_of_odds(self) -> None:
        pooled = aggregate.geomean_of_odds([0.6, 0.7, 0.55, 0.8])
        self.assertAlmostEqual(pooled, 0.6704, places=4)

    def test_identical_forecasts_pool_to_themselves(self) -> None:
        self.assertAlmostEqual(aggregate.geomean_of_odds([0.7, 0.7, 0.7]), 0.7, places=9)

    def test_pooled_value_lies_within_the_input_range(self) -> None:
        probs = [0.2, 0.45, 0.9]
        pooled = aggregate.geomean_of_odds(probs)
        self.assertGreater(pooled, min(probs))
        self.assertLess(pooled, max(probs))

    def test_symmetric_inputs_pool_to_one_half(self) -> None:
        self.assertAlmostEqual(aggregate.geomean_of_odds([0.3, 0.7]), 0.5, places=9)

    def test_weighting_pulls_toward_the_heavier_forecast(self) -> None:
        even = aggregate.geomean_of_odds([0.2, 0.8])
        skewed = aggregate.geomean_of_odds([0.2, 0.8], [3.0, 1.0])
        self.assertLess(skewed, even)

    def test_invalid_weights_fail_cleanly(self) -> None:
        with self.assertRaises(ValueError):
            aggregate.geomean_of_odds([0.4, 0.6], [-1, 1])

    def test_mismatched_weight_count_fails(self) -> None:
        with self.assertRaises(ValueError):
            aggregate.geomean_of_odds([0.4, 0.6], [1.0])

    def test_endpoint_probabilities_are_clipped_in_the_library_path(self) -> None:
        """SPEC-014 AC1. Was `test_endpoint_probabilities_raise_in_the_library_path`.

        SPEC-013 wrote that test to pin the crash (0.0 gave `ValueError`, 1.0 gave
        `ZeroDivisionError`) and to fail loudly once SPEC-014 landed. SPEC-014 chose
        option A: the pooling functions clip the endpoints themselves, so the library
        and the CLI now behave identically instead of diverging.
        """
        for probs in ([0.0, 0.5], [0.5, 1.0], [0.0, 1.0]):
            with self.subTest(probs=probs):
                pooled = aggregate.geomean_of_odds(probs)
                self.assertGreater(pooled, 0.0)
                self.assertLess(pooled, 1.0)
        self.assertAlmostEqual(aggregate.geomean_of_odds([0.0, 1.0]), 0.5, places=9)

    def test_out_of_range_probability_names_the_function_and_value(self) -> None:
        """SPEC-014 R3: clipping is for the endpoints, not for nonsense input."""
        with self.assertRaises(ValueError) as ctx:
            aggregate.geomean_of_odds([0.5, 5.0])
        message = str(ctx.exception)
        self.assertIn("geomean_of_odds", message)
        self.assertIn("probability 2", message)
        self.assertIn("5.0", message)

    def test_cli_clips_endpoints_so_the_command_still_works(self) -> None:
        """SPEC-014 AC4 (R6 regression): the CLI path keeps working unchanged."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(aggregate.main(["0.0", "1.0"]), 0)
        self.assertIn("0.5000", out.getvalue())

    def test_cli_extremize_on_endpoints_still_works(self) -> None:
        """SPEC-014 AC5: the extremize path via the CLI also survives endpoints."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(aggregate.main(["0.0", "1.0", "--extremize", "2"]), 0)
        self.assertIn("extremized", out.getvalue())

    def test_no_endpoint_input_ever_divides_by_zero(self) -> None:
        """SPEC-014 AC3: sweep both pooling entry points; ZeroDivisionError is banned."""
        sweep = [0.0, 1e-12, 0.5, 1 - 1e-12, 1.0]
        for p in sweep:
            with self.subTest(p=p):
                try:
                    pooled = aggregate.geomean_of_odds([p, 0.5])
                    sharpened = aggregate.extremize(p, 2.0)
                except ZeroDivisionError as exc:  # pragma: no cover - the thing under test
                    self.fail(f"ZeroDivisionError for p={p}: {exc}")
                for value in (pooled, sharpened):
                    self.assertGreater(value, 0.0)
                    self.assertLess(value, 1.0)


class ExtremizeTests(unittest.TestCase):
    def test_extremize_less_than_one_fails(self) -> None:
        with self.assertRaises(ValueError):
            aggregate.extremize(0.6, 0.5)

    def test_exponent_one_is_the_identity(self) -> None:
        self.assertAlmostEqual(aggregate.extremize(0.62, 1.0), 0.62, places=9)

    def test_pushes_away_from_one_half(self) -> None:
        self.assertGreater(aggregate.extremize(0.7, 2.0), 0.7)
        self.assertLess(aggregate.extremize(0.3, 2.0), 0.3)

    def test_one_half_is_a_fixed_point(self) -> None:
        self.assertAlmostEqual(aggregate.extremize(0.5, 3.0), 0.5, places=9)

    def test_endpoint_input_is_clipped(self) -> None:
        """SPEC-014 AC2. Was `test_endpoint_input_raises`.

        Same fix as `geomean_of_odds`: the endpoints are clipped inside the
        function, so neither `ValueError` nor `ZeroDivisionError` escapes, and
        extremizing still pushes each endpoint further from one half.
        """
        low = aggregate.extremize(0.0, 2.0)
        high = aggregate.extremize(1.0, 2.0)
        self.assertGreater(low, 0.0)
        self.assertLess(low, aggregate.extremize(0.0, 1.0))
        self.assertLess(high, 1.0)
        self.assertGreater(high, aggregate.extremize(1.0, 1.0))

    def test_out_of_range_probability_names_the_function_and_value(self) -> None:
        """SPEC-014 R3: an impossible probability still fails, and says so clearly."""
        with self.assertRaises(ValueError) as ctx:
            aggregate.extremize(-0.5, 2.0)
        message = str(ctx.exception)
        self.assertIn("extremize", message)
        self.assertIn("probability", message)
        self.assertIn("-0.5", message)

    def test_exponent_below_one_names_the_function_and_value(self) -> None:
        """SPEC-014 R3 for the other argument of the same function."""
        with self.assertRaises(ValueError) as ctx:
            aggregate.extremize(0.6, 0.5)
        message = str(ctx.exception)
        self.assertIn("extremize", message)
        self.assertIn("0.5", message)


class TrackRecordWeightTests(unittest.TestCase):
    def test_track_log_requires_forecaster_cli(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                aggregate.main(["0.6", "--track-log", "missing.jsonl"])

    def test_track_record_weights_are_smoothed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            records = [
                {"event": "forecast", "id": 1, "prob": 0.8, "forecaster": "good", "outcome": 1},
                {"event": "forecast", "id": 2, "prob": 0.8, "forecaster": "bad", "outcome": 0},
            ]
            log.write_text("".join(json.dumps(r) + "\n" for r in records))
            weights, stats = aggregate.track_record_weights(str(log), ["good", "bad"])
            self.assertGreater(weights[0], weights[1])
            self.assertEqual(stats["good"][0], 1)
            self.assertEqual(stats["bad"][0], 1)

    def test_calibrated_persona_outweighs_the_uninformed_one(self) -> None:
        """oracle (Brier 0.006) must earn more weight than coinflipper (0.25)."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            write_ledger(log, ORACLE, COINFLIPPER)
            weights, stats = aggregate.track_record_weights(
                str(log), [ORACLE.name, COINFLIPPER.name])
            self.assertGreater(weights[0], weights[1])
            self.assertEqual(stats[ORACLE.name][0], len(ORACLE.record))

    def test_worse_than_chance_persona_ranks_last(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            write_ledger(log, ORACLE, COINFLIPPER, HEDGEHOG)
            weights, _ = aggregate.track_record_weights(
                str(log), [ORACLE.name, COINFLIPPER.name, HEDGEHOG.name])
            self.assertEqual(weights.index(min(weights)), 2)

    def test_unknown_forecaster_still_receives_a_prior_weight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            write_ledger(log, ORACLE)
            weights, _ = aggregate.track_record_weights(str(log), [ORACLE.name, "stranger"])
            self.assertGreater(weights[1], 0.0)


class TimeDecayTests(unittest.TestCase):
    def test_time_decay_downweights_older_forecasts(self) -> None:
        as_of = aggregate.parse_datetime("2026-07-15")
        weights = aggregate.time_decay_weights(["2026-07-01", "2026-07-15"], as_of, 14)
        self.assertAlmostEqual(weights[0], 0.5, places=4)
        self.assertAlmostEqual(weights[1], 1.0, places=4)

    def test_two_half_lives_quarter_the_weight(self) -> None:
        as_of = aggregate.parse_datetime("2026-07-29")
        weights = aggregate.time_decay_weights(["2026-07-01"], as_of, 14)
        self.assertAlmostEqual(weights[0], 0.25, places=4)

    def test_missing_date_is_not_penalised_to_zero(self) -> None:
        as_of = aggregate.parse_datetime("2026-07-15")
        weights = aggregate.time_decay_weights([None], as_of, 14)
        self.assertGreater(weights[0], 0.0)

    def test_non_positive_half_life_fails(self) -> None:
        as_of = aggregate.parse_datetime("2026-07-15")
        for half_life in (0.0, -14.0):
            with self.subTest(half_life=half_life), self.assertRaises(ValueError):
                aggregate.time_decay_weights(["2026-07-01"], as_of, half_life)


class AggregateCliTests(unittest.TestCase):
    def test_positional_probabilities_are_pooled(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(aggregate.main(["0.6", "0.7", "0.55", "0.8"]), 0)
        self.assertIn("0.67", out.getvalue())

    def test_invalid_probability_is_rejected_cleanly(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                aggregate.main(["1.4"])

    def test_no_input_fails_cleanly(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                aggregate.main([])


if __name__ == "__main__":
    unittest.main()
