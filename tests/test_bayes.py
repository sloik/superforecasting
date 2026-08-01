"""Tests for scripts/bayes.py — the inside view: update a prior with evidence."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

import bayes  # noqa: E402


class OddsConversionTests(unittest.TestCase):
    def test_probability_odds_round_trip_and_open_validation(self) -> None:
        for probability in (0.1, 0.5, 0.9):
            with self.subTest(probability=probability):
                self.assertAlmostEqual(
                    bayes.odds_to_p(bayes.p_to_odds(probability)), probability, places=6
                )
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                bayes.p_to_odds(probability)

    def test_even_odds_at_one_half(self) -> None:
        self.assertAlmostEqual(bayes.p_to_odds(0.5), 1.0, places=9)
        self.assertAlmostEqual(bayes.odds_to_p(1.0), 0.5, places=9)

    def test_odds_increase_monotonically_with_probability(self) -> None:
        self.assertLess(bayes.p_to_odds(0.2), bayes.p_to_odds(0.5))
        self.assertLess(bayes.p_to_odds(0.5), bayes.p_to_odds(0.8))

    def test_out_of_range_probability_rejected(self) -> None:
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                bayes.p_to_odds(probability)


class UpdateTests(unittest.TestCase):
    def test_prior_must_be_open_probability(self) -> None:
        with self.assertRaises(ValueError):
            bayes.update(0.0, [2.0])

    def test_update_applies_multiple_likelihood_ratios(self) -> None:
        self.assertAlmostEqual(bayes.update(0.5, [2.0, 2.0, 0.5]), 2 / 3, places=4)

    def test_likelihood_ratio_of_one_is_a_no_op(self) -> None:
        self.assertAlmostEqual(bayes.update(0.37, [1.0, 1.0]), 0.37, places=9)

    def test_empty_evidence_returns_the_prior(self) -> None:
        self.assertAlmostEqual(bayes.update(0.42, []), 0.42, places=9)

    def test_order_of_likelihood_ratios_does_not_matter(self) -> None:
        self.assertAlmostEqual(bayes.update(0.3, [2.0, 0.5, 4.0]),
                               bayes.update(0.3, [4.0, 2.0, 0.5]), places=9)

    def test_supporting_evidence_raises_and_undercutting_lowers(self) -> None:
        self.assertGreater(bayes.update(0.3, [3.0]), 0.3)
        self.assertLess(bayes.update(0.3, [0.3]), 0.3)

    def test_non_positive_likelihood_ratio_rejected(self) -> None:
        for lr in (0.0, -2.0):
            with self.subTest(lr=lr), self.assertRaises(ValueError):
                bayes.update(0.5, [lr])

    def test_posterior_stays_in_the_open_interval(self) -> None:
        self.assertLess(bayes.update(0.5, [1e6]), 1.0)
        self.assertGreater(bayes.update(0.5, [1e-6]), 0.0)


class BayesCliTests(unittest.TestCase):
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

    def test_prior_at_the_endpoints_fails_cleanly(self) -> None:
        for prior in ("0", "1"):
            with self.subTest(prior=prior), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bayes.main(["--prior", prior, "--lr", "2"])

    def test_negative_likelihood_ratio_fails_cleanly(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bayes.main(["--prior", "0.5", "--lr", "-2"])


if __name__ == "__main__":
    unittest.main()
