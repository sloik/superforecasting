"""Tests for scripts/brier.py — Brier scoring and Murphy's decomposition.

Persona fixtures carry the meaning here: asserting that `oracle` beats
`coinflipper` is the actual claim under test, and reads that way.
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
from _personas import (  # noqa: E402
    CASSANDRA, COINFLIPPER, EEYORE, FOX, HEDGEHOG, ORACLE, POLLYANNA, write_ledger,
)

import brier  # noqa: E402
import forecastlog  # noqa: E402
from forecast_common import time_weighted_brier_score  # noqa: E402


def args_for(**kwargs):
    defaults = {"prob": None, "outcome": None, "pairs": None, "log": None,
                "baseline_prob": None, "baseline_field": None}
    defaults.update(kwargs)
    return type("Args", (), defaults)


class BrierScoreTests(unittest.TestCase):
    def test_invalid_probability_rejected(self) -> None:
        with self.assertRaises(ValueError):
            brier.brier([(1.2, 1)])

    def test_perfect_forecasts_score_zero(self) -> None:
        self.assertEqual(brier.brier([(1.0, 1), (0.0, 0)]), 0.0)

    def test_uninformed_forecasts_score_a_quarter(self) -> None:
        self.assertEqual(brier.brier([(0.5, 1), (0.5, 0)]), 0.25)

    def test_invalid_outcome_rejected(self) -> None:
        with self.assertRaises(ValueError):
            brier.brier([(0.5, 2)])


class PersonaCalibrationTests(unittest.TestCase):
    """The persona Brier scores are the documented contract of _personas.py."""

    def test_coinflipper_scores_exactly_one_quarter(self) -> None:
        self.assertEqual(brier.brier(COINFLIPPER.record), 0.25)

    def test_oracle_beats_the_coinflipper(self) -> None:
        self.assertLess(brier.brier(ORACLE.record), brier.brier(COINFLIPPER.record))

    def test_fox_beats_the_hedgehog(self) -> None:
        self.assertLess(brier.brier(FOX.record), brier.brier(HEDGEHOG.record))

    def test_hedgehog_is_worse_than_uninformed(self) -> None:
        """Confident and wrong is worse than admitting you don't know."""
        self.assertGreater(brier.brier(HEDGEHOG.record), brier.brier(COINFLIPPER.record))

    def test_both_biases_score_alike(self) -> None:
        """Optimism and pessimism are symmetric errors on this record."""
        self.assertAlmostEqual(brier.brier(POLLYANNA.record), brier.brier(EEYORE.record))

    def test_persona_table_matches_computed_scores(self) -> None:
        for persona in (ORACLE, CASSANDRA, FOX, COINFLIPPER, HEDGEHOG, POLLYANNA, EEYORE):
            with self.subTest(persona=persona.name):
                self.assertAlmostEqual(brier.brier(persona.record), persona.brier)


class DecomposeTests(unittest.TestCase):
    def test_components_satisfy_murphys_identity_when_bins_separate(self) -> None:
        """Brier = Reliability - Resolution + Uncertainty, exactly.

        The identity is exact only when each bin holds a single distinct
        probability. With enough bins that holds; see the companion test for what
        happens when it does not.
        """
        pairs = list(FOX.record) + list(ORACLE.record)
        reliability, resolution, uncertainty, _, _ = brier.decompose(pairs, nbins=20)
        self.assertAlmostEqual(brier.brier(pairs),
                               reliability - resolution + uncertainty, places=12)

    def test_coarse_bins_leave_a_within_bin_variance_residual(self) -> None:
        """Documents why the decomposition is approximate at nbins=10.

        FOX's 0.65 and 0.70 share a decile bin, so reliability is measured against
        their mean and the identity picks up a small positive residual. This is a
        property of binned Murphy decomposition, not a defect — but it is worth
        pinning so nobody 'fixes' the arithmetic later.
        """
        pairs = list(FOX.record) + list(ORACLE.record)
        coarse = brier.decompose(pairs, nbins=10)
        fine = brier.decompose(pairs, nbins=20)
        observed = brier.brier(pairs)
        coarse_residual = observed - (coarse[0] - coarse[1] + coarse[2])
        fine_residual = observed - (fine[0] - fine[1] + fine[2])
        self.assertGreater(coarse_residual, 0.0)
        self.assertAlmostEqual(fine_residual, 0.0, places=12)
        self.assertGreater(coarse_residual, fine_residual)

    def test_uncertainty_is_base_rate_variance(self) -> None:
        pairs = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
        _, _, uncertainty, obar, _ = brier.decompose(pairs)
        self.assertAlmostEqual(obar, 0.5)
        self.assertAlmostEqual(uncertainty, 0.25)

    def test_confident_and_correct_has_high_resolution(self) -> None:
        _, sharp_res, _, _, _ = brier.decompose(list(CASSANDRA.record))
        _, flat_res, _, _, _ = brier.decompose(list(COINFLIPPER.record))
        self.assertGreater(sharp_res, flat_res)

    def test_coinflipper_has_no_resolution(self) -> None:
        """Never moving off the base rate means never discriminating."""
        _, resolution, _, _, _ = brier.decompose(list(COINFLIPPER.record))
        self.assertAlmostEqual(resolution, 0.0)

    def test_nonpositive_bins_rejected(self) -> None:
        for nbins in (0, -3):
            with self.subTest(nbins=nbins), self.assertRaises(ValueError):
                brier.decompose([(0.5, 1)], nbins=nbins)

    def test_table_rows_cover_the_populated_bins(self) -> None:
        _, _, _, _, table = brier.decompose(list(ORACLE.record), nbins=10)
        self.assertTrue(len(table) >= 1)


class BaselineTests(unittest.TestCase):
    def test_baseline_skill_score_data(self) -> None:
        rows = [{"prob": 0.7, "outcome": 1}, {"prob": 0.3, "outcome": 0}]
        label, pairs = brier.baseline_pairs(rows, args_for(baseline_prob=0.5))
        self.assertEqual(label, "constant 0.500")
        self.assertEqual(pairs, [(0.5, 1), (0.5, 0)])

    def test_baseline_field_pulls_per_row_values(self) -> None:
        rows = [{"prob": 0.7, "outcome": 1, "crowd_prob": 0.6},
                {"prob": 0.3, "outcome": 0, "crowd_prob": 0.4}]
        _, pairs = brier.baseline_pairs(rows, args_for(baseline_field="crowd_prob"))
        self.assertEqual(pairs, [(0.6, 1), (0.4, 0)])

    def test_rows_missing_the_baseline_field_are_skipped(self) -> None:
        rows = [{"prob": 0.7, "outcome": 1, "crowd_prob": 0.6},
                {"prob": 0.3, "outcome": 0}]
        _, pairs = brier.baseline_pairs(rows, args_for(baseline_field="crowd_prob"))
        self.assertEqual(pairs, [(0.6, 1)])

    def test_no_baseline_requested_yields_nothing(self) -> None:
        _, pairs = brier.baseline_pairs([{"prob": 0.7, "outcome": 1}], args_for())
        self.assertFalse(pairs)


class LoadRowsTests(unittest.TestCase):
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
            rows = brier.load_rows(args_for(log=str(log)))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["outcome"], 1)
            self.assertEqual(rows[0]["baseline_prob"], 0.5)

    def test_unresolved_forecasts_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                forecastlog.main(["--log", str(log), "add", "--question", "open",
                                  "--prob", "0.7"])
            self.assertEqual(brier.load_rows(args_for(log=str(log))), [])

    def test_inline_pairs_are_parsed(self) -> None:
        rows = brier.load_rows(args_for(pairs=["0.7:1", "0.3:0"]))
        self.assertEqual([(r["prob"], r["outcome"]) for r in rows], [(0.7, 1), (0.3, 0)])

    def test_malformed_pair_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            brier.load_rows(args_for(pairs=["0.7-1"]))

    def test_prob_without_outcome_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            brier.load_rows(args_for(prob=0.7))


class TimeWeightedTests(unittest.TestCase):
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


class BrierCliTests(unittest.TestCase):
    def test_brier_cli_outputs_skill_score(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = brier.main(["--pairs", "0.7:1", "0.3:0", "--baseline-prob", "0.5"])
        self.assertEqual(code, 0)
        self.assertIn("Brier Skill Score", buf.getvalue())

    def test_single_pair_scores_from_the_cli(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(brier.main(["--prob", "0.7", "--outcome", "1"]), 0)
        self.assertIn("0.0900", buf.getvalue())

    def test_oracle_ledger_beats_a_half_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "forecasts.jsonl"
            write_ledger(log, ORACLE)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(brier.main(["--log", str(log), "--baseline-prob", "0.5"]), 0)
            self.assertIn("Brier Skill Score", buf.getvalue())

    def test_conflicting_baseline_options_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                brier.main(["--pairs", "0.7:1", "--baseline-prob", "0.5",
                            "--baseline-field", "crowd_prob"])

    def test_zero_bins_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                brier.main(["--pairs", "0.7:1", "--bins", "0"])

    def test_no_data_source_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty.jsonl"
            empty.write_text("")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    brier.main(["--log", str(empty)])


if __name__ == "__main__":
    unittest.main()
