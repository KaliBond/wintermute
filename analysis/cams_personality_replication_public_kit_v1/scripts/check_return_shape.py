#!/usr/bin/env python3
"""Check the shape of an Arm A, B, or C return.

This does not score a return and it does not contain an answer key.
A malformed hash field is reported and left unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ARM_C_FIELDS = [
    "arm",
    "model",
    "entity",
    "era_id",
    "year_start",
    "year_end",
    "arm_a_sha256",
    "q3_grade",
    "mood_levels",
    "evidence_note",
    "sources",
    "timestamp_aedt",
]
ENTITIES = {"Russia", "Australia", "USA", "China", "UK", "Norway", "Denmark"}
GRADES = {"close", "broad", "partial", "mixed", "direct_disagreement"}
MOOD = {"agree", "partial", "disagree", "unclear"}
MOOD_KEYS = {"popular", "institutional", "official"}
ARM_A_FIELDS = [
    "arm",
    "model",
    "entity",
    "era_id",
    "year_start",
    "year_end",
    "leaders",
    "constraints",
    "pocket",
    "notes",
    "sha256_input_csv",
    "sha256_this_output",
    "timestamp_aedt",
]
ARM_B_FIELDS = [
    "blind_id",
    "entity",
    "chosen_era_id",
    "chosen_year_start",
    "chosen_year_end",
    "confidence",
    "brief_reason",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    raw = path.read_bytes()
    if raw == b"":
        return rows
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: malformed JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no}: row is not a JSON object")
        row["_line"] = line_no
        rows.append(row)
    return rows


def check_arm_c(rows: list[dict]) -> list[str]:
    problems = []
    seen = set()
    for row in rows:
        line = row["_line"]
        for field in ARM_C_FIELDS:
            if field not in row:
                problems.append(f"line {line}: missing {field}")
        if row.get("arm") != "C":
            problems.append(f"line {line}: arm is {row.get('arm')!r}, expected 'C'")
        entity = row.get("entity")
        era = row.get("era_id")
        if entity not in ENTITIES:
            problems.append(f"line {line}: entity {entity!r} is outside the packet list")
        if entity == "Denmark" and era == "IX":
            problems.append(f"line {line}: Denmark IX is unobserved and must be omitted")
        if row.get("q3_grade") not in GRADES:
            problems.append(f"line {line}: q3_grade {row.get('q3_grade')!r} is outside the five grades")
        mood = row.get("mood_levels")
        if not isinstance(mood, dict) or set(mood) != MOOD_KEYS:
            problems.append(f"line {line}: mood_levels must be popular, institutional, and official")
        elif any(mood.get(key) not in MOOD for key in MOOD_KEYS):
            problems.append(f"line {line}: a mood label is outside agree, partial, disagree, unclear")
        sources = row.get("sources")
        if not isinstance(sources, list) or not sources:
            problems.append(f"line {line}: sources needs at least one item")
        else:
            for source in sources:
                if not isinstance(source, dict) or not source.get("title") or not source.get("url"):
                    problems.append(f"line {line}: a source is missing title or url")
                    break
        digest = row.get("arm_a_sha256")
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            problems.append(
                f"line {line}: arm_a_sha256 is not 64 lowercase hex characters "
                f"(length {len(digest) if isinstance(digest, str) else 'n/a'}); left unchanged"
            )
        key = (entity, era)
        if key in seen:
            problems.append(f"line {line}: duplicate {entity} {era}")
        seen.add(key)
    return problems


def check_arm_a(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        line = row["_line"]
        for field in ARM_A_FIELDS:
            if field not in row:
                problems.append(f"line {line}: missing observed field {field}")
        if row.get("arm") != "A":
            problems.append(f"line {line}: arm is {row.get('arm')!r}, expected 'A'")
        if row.get("entity") == "Denmark" and row.get("era_id") == "IX":
            problems.append(f"line {line}: Denmark IX must be omitted")
        digest = row.get("sha256_this_output")
        if digest in ("", None):
            problems.append(
                f"line {line}: sha256_this_output is empty. "
                "Hash the finished file with sha256_file.py. Do not type a digest."
            )
        elif not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            problems.append(f"line {line}: sha256_this_output is not 64 lowercase hex characters")
    return problems


def check_arm_b(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        line = row["_line"]
        for field in ARM_B_FIELDS:
            if field not in row:
                problems.append(f"line {line}: missing observed field {field}")
    return problems


def fixture_row() -> dict:
    """Schema fixture only. Not a historical era and not an answer."""
    return {
        "arm": "C",
        "model": "schema-fixture",
        "entity": "Australia",
        "era_id": "NOT_A_RESULT",
        "year_start": 0,
        "year_end": 0,
        "arm_a_sha256": "ab" * 32,
        "q3_grade": "close",
        "mood_levels": {"popular": "agree", "institutional": "partial", "official": "unclear"},
        "evidence_note": "Schema fixture. Not a historical grading.",
        "sources": [{"title": "Schema fixture", "url": "https://example.invalid/schema-fixture", "accessed_aedt": "2026-11-07"}],
        "timestamp_aedt": "2026-11-07 00:00 AEDT",
        "_line": 1,
    }


def self_test() -> int:
    good = check_arm_c([fixture_row()])
    bad = dict(fixture_row())
    bad["q3_grade"] = "wrong"
    bad["entity"] = "Denmark"
    bad["era_id"] = "IX"
    bad["arm_a_sha256"] = "not-a-hash"
    caught = check_arm_c([bad])
    if good:
        print("self-test failed: fixture was rejected")
        for item in good:
            print(item)
        return 1
    needed = ("outside the five grades", "Denmark IX", "not 64 lowercase hex")
    if not all(any(piece in item for item in caught) for piece in needed):
        print("self-test failed: expected problems were not reported")
        for item in caught:
            print(item)
        return 1
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check return shape. Does not score.")
    parser.add_argument("--arm", choices=("A", "B", "C"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.arm is None or args.path is None:
        parser.error("pass --arm and a JSONL path, or --self-test")
    rows = load_jsonl(args.path)
    if args.arm == "A":
        problems = check_arm_a(rows)
    elif args.arm == "B":
        problems = check_arm_b(rows)
    else:
        problems = check_arm_c(rows)
    print(f"rows={len(rows)} problems={len(problems)}")
    for item in problems:
        print(item)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
