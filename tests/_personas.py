"""Forecaster personas — fixtures that encode the behaviour under test.

These are not decorative names. Each persona is a track record whose Brier score
is a documented, asserted property, so a test can say what it means:

    self.assertLess(brier_of(ORACLE), brier_of(COINFLIPPER))

reads as "a calibrated forecaster beats an uninformed one", which is the actual
claim. `forecaster_a`/`forecaster_b` would hide it.

Scores below are exact and verified by `test_suite_structure` /
`test_brier.PersonaCalibrationTests`; changing a record changes a score, and the
tests will say so.

    persona      Brier     character
    ---------    -------   ------------------------------------------------
    CASSANDRA    0.00250   right and extreme, but disbelieved (best Brier)
    ORACLE       0.00625   confident and right — the calibration target
    FOX          0.10625   many modest calls, mostly right (Tetlock's fox)
    COINFLIPPER  0.25000   always 0.50 — the uninformed baseline, exactly
    POLLYANNA    0.41000   optimism bias: everything is likely
    EEYORE       0.41000   pessimism bias: nothing is likely
    HEDGEHOG     0.49010   one big theory, stated at 0.99 — worse than chance
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """A forecaster with a fixed track record of (probability, outcome) pairs."""

    name: str
    character: str
    record: tuple[tuple[float, int], ...]

    @property
    def brier(self) -> float:
        return sum((p - o) ** 2 for p, o in self.record) / len(self.record)

    def probs(self) -> list[float]:
        return [p for p, _ in self.record]

    def outcomes(self) -> list[int]:
        return [o for _, o in self.record]


ORACLE = Persona(
    "oracle", "confident and right — the calibration target",
    ((0.95, 1), (0.05, 0), (0.90, 1), (0.10, 0)),
)

CASSANDRA = Persona(
    "cassandra", "right and extreme, and nobody listens",
    ((0.95, 1), (0.05, 0), (0.95, 1), (0.05, 0)),
)

FOX = Persona(
    "fox", "many modest calls, mostly right — Tetlock's fox",
    ((0.70, 1), (0.30, 0), (0.65, 1), (0.35, 0)),
)

COINFLIPPER = Persona(
    "coinflipper", "always 0.50 — the uninformed baseline",
    ((0.50, 1), (0.50, 0), (0.50, 1), (0.50, 0)),
)

HEDGEHOG = Persona(
    "hedgehog", "one big theory, always at 0.99 — worse than chance",
    ((0.99, 1), (0.99, 0), (0.01, 0), (0.01, 1)),
)

POLLYANNA = Persona(
    "pollyanna", "optimism bias — everything is likely",
    ((0.90, 1), (0.90, 0), (0.90, 1), (0.90, 0)),
)

EEYORE = Persona(
    "eeyore", "pessimism bias — nothing is likely",
    ((0.10, 1), (0.10, 0), (0.10, 1), (0.10, 0)),
)

ALL_PERSONAS = (ORACLE, CASSANDRA, FOX, COINFLIPPER, HEDGEHOG, POLLYANNA, EEYORE)


def ledger_records(*personas: Persona, start_id: int = 1) -> list[dict]:
    """Build resolved forecast+resolve event rows for the given personas.

    Produces the same shape `forecastlog.py` writes, so it folds correctly and can
    be consumed by `brier.py` and `aggregate.track_record_weights`.
    """
    rows: list[dict] = []
    next_id = start_id
    for persona in personas:
        for index, (prob, outcome) in enumerate(persona.record):
            rows.append({
                "event": "forecast",
                "schema_version": 2,
                "id": next_id,
                "question": f"{persona.name} call #{index + 1}",
                "prob": prob,
                "forecaster": persona.name,
                "created": f"2026-01-{index + 1:02d}T00:00:00Z",
                "resolve_by": "2026-06-30",
                "status": "open",
                "outcome": None,
                "tags": [],
            })
            rows.append({
                "event": "resolve",
                "schema_version": 2,
                "id": next_id,
                "outcome": outcome,
                "prob": prob,
                "resolved": f"2026-02-{index + 1:02d}T00:00:00Z",
            })
            next_id += 1
    return rows


def write_ledger(path, *personas: Persona) -> None:
    """Write a JSONL ledger of the personas' resolved track records to `path`."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n"
                            for row in ledger_records(*personas)))
