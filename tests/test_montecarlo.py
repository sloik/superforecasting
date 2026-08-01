"""Tests for scripts/montecarlo.py — Fermi-ized simulation of range-valued quantities."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402

import montecarlo  # noqa: E402


def model(**overrides) -> dict:
    base = {
        "trials": 200,
        "vars": {"x": {"dist": "point", "value": 10}},
        "expr": "x",
    }
    base.update(overrides)
    return base


class RunTests(unittest.TestCase):
    def test_happy_path_is_reproducible(self) -> None:
        out = montecarlo.run({
            "trials": 100,
            "vars": {"x": {"dist": "point", "value": 10}},
            "expr": "x * 2",
            "threshold": 15,
            "direction": ">",
        })
        self.assertEqual(out["p50"], 20)
        self.assertEqual(out["prob"], 1.0)

    def test_threshold_no_sample_can_reach_gives_zero_probability(self) -> None:
        # x is always 10, so P(x < 5) == 0
        out = montecarlo.run(model(expr="x", threshold=5, direction="<"))
        self.assertEqual(out["prob"], 0.0)

    def test_direction_variants_are_all_honoured(self) -> None:
        for direction, threshold, expected in ((">", 5, 1.0), (">=", 10, 1.0),
                                               ("<", 5, 0.0), ("<=", 10, 1.0)):
            with self.subTest(direction=direction):
                out = montecarlo.run(model(threshold=threshold, direction=direction))
                self.assertEqual(out["prob"], expected)

    def test_percentiles_are_ordered(self) -> None:
        out = montecarlo.run({
            "trials": 2000,
            "vars": {"x": {"dist": "uniform", "lo": 0, "hi": 100}},
            "expr": "x",
        })
        for lower, upper in (("min", "p5"), ("p5", "p25"), ("p25", "p50"),
                             ("p50", "p75"), ("p75", "p95"), ("p95", "max")):
            with self.subTest(pair=(lower, upper)):
                self.assertLessEqual(out[lower], out[upper])

    def test_summary_reports_trials_mean_and_spread(self) -> None:
        out = montecarlo.run(model(trials=321))
        self.assertEqual(out["trials"], 321)
        self.assertEqual(out["mean"], 10)
        self.assertEqual(out["sd"], 0.0)

    def test_expression_can_combine_several_variables(self) -> None:
        out = montecarlo.run({
            "trials": 50,
            "vars": {"a": {"dist": "point", "value": 3},
                     "b": {"dist": "point", "value": 4}},
            "expr": "a * b",
        })
        self.assertEqual(out["p50"], 12)

    def test_zero_or_negative_trials_are_rejected(self) -> None:
        for trials in (0, -5):
            with self.subTest(trials=trials), self.assertRaises(ValueError):
                montecarlo.run(model(trials=trials))

    def test_empty_vars_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run({"trials": 10, "vars": {}, "expr": "1"})


class DistributionTests(unittest.TestCase):
    def test_every_supported_distribution_runs(self) -> None:
        specs = {
            "normal": {"dist": "normal", "mean": 10, "sd": 2},
            "uniform": {"dist": "uniform", "lo": 1, "hi": 5},
            "triangular": {"dist": "triangular", "lo": 1, "mode": 2, "hi": 3},
            "lognormal": {"dist": "lognormal", "mean": 100, "sd_factor": 1.5},
            "beta": {"dist": "beta", "a": 2, "b": 5},
            "bernoulli": {"dist": "bernoulli", "p": 0.4},
            "point": {"dist": "point", "value": 7},
        }
        for name, spec in specs.items():
            with self.subTest(dist=name):
                out = montecarlo.run({"trials": 100, "vars": {"x": spec}, "expr": "x"})
                self.assertIn("p50", out)

    def test_lognormal_accepts_both_parameterisations(self) -> None:
        for spec in ({"dist": "lognormal", "mean": 100, "sd_factor": 1.5},
                     {"dist": "lognormal", "mu": 4.6, "sigma": 0.4}):
            with self.subTest(spec=spec):
                out = montecarlo.run({"trials": 100, "vars": {"x": spec}, "expr": "x"})
                self.assertGreater(out["p50"], 0)

    def test_point_distribution_has_no_spread(self) -> None:
        out = montecarlo.run(model(vars={"x": {"dist": "point", "value": 42}}))
        self.assertEqual(out["p5"], 42)
        self.assertEqual(out["p95"], 42)
        self.assertEqual(out["sd"], 0.0)

    def test_bernoulli_stays_within_zero_and_one(self) -> None:
        out = montecarlo.run({
            "trials": 500, "vars": {"x": {"dist": "bernoulli", "p": 0.5}}, "expr": "x",
        })
        self.assertGreaterEqual(out["min"], 0)
        self.assertLessEqual(out["max"], 1)

    def test_bad_distribution_params_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run({
                "trials": 1, "vars": {"x": {"dist": "bernoulli", "p": 1.2}}, "expr": "x",
            })

    def test_unknown_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run(model(vars={"x": {"dist": "cauchy", "scale": 1}}))

    def test_triangular_with_mode_outside_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run(model(vars={"x": {"dist": "triangular", "lo": 1, "mode": 9, "hi": 3}}))

    def test_uniform_with_inverted_bounds_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run(model(vars={"x": {"dist": "uniform", "lo": 5, "hi": 1}}))


class ExpressionSafetyTests(unittest.TestCase):
    def test_unsafe_expression_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run({
                "trials": 1,
                "vars": {"x": {"dist": "point", "value": 10}},
                "expr": "__import__('os').system('echo bad')",
            })

    def test_attribute_access_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run(model(expr="x.__class__"))

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.run(model(expr="x + y"))

    def test_compile_expr_returns_evaluable_code_for_safe_input(self) -> None:
        code = montecarlo.compile_expr("x * 2", {"x"})
        self.assertEqual(eval(code, {"__builtins__": {}}, {"x": 21}), 42)

    def test_compile_expr_allows_whitelisted_math_functions(self) -> None:
        code = montecarlo.compile_expr("min(x, 5)", {"x"})
        self.assertEqual(eval(code, {"__builtins__": {}}, {"x": 9, **montecarlo._SAFE_FUNCS}), 5)

    def test_compile_expr_rejects_non_whitelisted_calls(self) -> None:
        with self.assertRaises(ValueError):
            montecarlo.compile_expr("open('x')", {"x"})


class MonteCarloCliTests(unittest.TestCase):
    def test_json_model_runs_from_the_cli(self) -> None:
        spec = json.dumps({
            "trials": 100,
            "vars": {"x": {"dist": "point", "value": 10}},
            "expr": "x",
            "threshold": 5,
            "direction": ">",
        })
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(montecarlo.main(["--json", spec]), 0)
        self.assertIn("p50", out.getvalue())

    def test_malformed_json_fails_cleanly(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                montecarlo.main(["--json", "{not json"])

    def test_unsafe_expression_from_cli_fails_cleanly(self) -> None:
        spec = json.dumps({
            "trials": 1,
            "vars": {"x": {"dist": "point", "value": 1}},
            "expr": "__import__('os')",
        })
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                montecarlo.main(["--json", spec])


if __name__ == "__main__":
    unittest.main()
