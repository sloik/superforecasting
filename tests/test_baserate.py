"""Tests for scripts/baserate.py — the outside view: reference class → anchor + interval."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

import baserate  # noqa: E402


class WilsonIntervalTests(unittest.TestCase):
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

    def test_interval_narrows_as_the_sample_grows(self) -> None:
        narrow = baserate.wilson(300, 1100)
        wide = baserate.wilson(3, 11)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_interval_stays_inside_unit_range_at_the_extremes(self) -> None:
        for hits, total in ((0, 10), (10, 10)):
            with self.subTest(hits=hits, total=total):
                lo, hi = baserate.wilson(hits, total)
                self.assertGreaterEqual(lo, 0.0)
                self.assertLessEqual(hi, 1.0)
                self.assertLess(lo, hi)

    def test_interval_is_symmetric_under_outcome_flip(self) -> None:
        lo_a, hi_a = baserate.wilson(3, 11)
        lo_b, hi_b = baserate.wilson(8, 11)
        self.assertAlmostEqual(lo_a, 1 - hi_b, places=9)
        self.assertAlmostEqual(hi_a, 1 - lo_b, places=9)


class BaseRateCliTests(unittest.TestCase):
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

    def test_hits_without_total_falls_back_to_the_checklist(self) -> None:
        """Documents current behaviour: incomplete input is guidance, not an error.

        `--hits 3` with no `--total` does not raise — it prints the reference-class
        checklist and exits 0, same as no arguments at all. Defensible (the tool
        teaches before it computes) but easy to mistake for a silent no-op, so it
        is pinned here rather than left implicit.
        """
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(baserate.main(["--hits", "3"]), 0)
        self.assertIn("REFERENCE-CLASS CHECKLIST", out.getvalue())

    def test_non_integer_counts_fail_cleanly(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                baserate.main(["--hits", "not-a-number", "--total", "11"])

    def test_all_hits_reports_an_anchor_below_one(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(baserate.main(["--hits", "10", "--total", "10"]), 0)
        self.assertNotIn("1.0000", out.getvalue().split("\n")[0])


if __name__ == "__main__":
    unittest.main()
