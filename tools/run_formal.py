#!/usr/bin/env python3
"""Run pinned, bounded TLC checks and required semantic counterexamples."""
import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TLC_SHA256 = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
ROOT = Path(__file__).resolve().parents[1]
CASES = (("lifecycle", None), ("terminal", "TerminalImmutable"),
         ("history", "HistoryComplete"), ("score", "LatestProbabilityScore"),
         ("ids", "UniqueIds"), ("unscored", "OnlyResolvedScored"))


def qualify(output, code, target=None):
    """Require native completion or exactly the intended semantic failure."""
    errors = re.findall(r"^Error: (.*)$", output, re.MULTILINE)
    if target is None:
        return (type(code) is int and code == 0 and not errors
                and "Model checking completed. No error has been found." in output
                and " distinct states found" in output)
    return (type(code) is int and code == 12
            and errors == ["Invariant " + target + " is violated.",
                           "The behavior up to this point is:"]
            and "State 1:" in output and " distinct states found" in output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--output", type=Path, help="Optional directory for fresh native logs")
    args = parser.parse_args(argv)
    jar = args.jar.resolve()
    if hashlib.sha256(jar.read_bytes()).hexdigest() != TLC_SHA256:
        parser.error("TLC jar does not match the pinned 1.7.4 SHA-256")
    java = shutil.which(args.java)
    if java is None:
        parser.error("Java executable required")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    for case, target in CASES:
        with tempfile.TemporaryDirectory(prefix="forecast-tlc-") as tmp:
            work = Path(tmp)
            for name in ("ForecastLifecycle.tla", case + ".cfg"):
                shutil.copyfile(ROOT / "formal" / name, work / name)
            result = subprocess.run(
                [java, "-Xmx512m", "-cp", str(jar), "tlc2.TLC", "-workers", "1",
                 "-seed", "1", "-metadir", "states", "-config", case + ".cfg",
                 "ForecastLifecycle.tla"],
                cwd=work, capture_output=True, text=True, timeout=120,
            )
            output = result.stdout + result.stderr
        if args.output:
            hashes = {name: hashlib.sha256((ROOT / "formal" / name).read_bytes()).hexdigest()
                      for name in ("ForecastLifecycle.tla", case + ".cfg")}
            header = repr({"case": case, "inputs": hashes, "exit": result.returncode})
            (args.output / (case + ".log")).write_text(header + "\n" + output)
        if not qualify(output, result.returncode, target):
            print(output)
            raise SystemExit("FAIL: " + case + " did not produce its required result")
        counts = re.search(r"[\d,]+ states generated, [\d,]+ distinct states found.*", output)
        print("PASS " + case + ": " + (counts.group(0) if counts else "qualified"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
