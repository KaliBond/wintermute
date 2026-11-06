#!/usr/bin/env python3
"""Reconcile this deposit to its sealed score rows and manifests.

This script does not rescore against a HOLD key, call a model, or edit seals.
It checks file hashes, recomputes Arm B and Arm C counts from the supplied
scored JSONL under the READY adjacency rule, and writes tables next to itself.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
TABLES = ROOT / "tables"

READY = {
    frozenset({"close", "broad"}),
    frozenset({"broad", "partial"}),
    frozenset({"partial", "mixed"}),
}
OLD_SCALE = READY | {frozenset({"mixed", "direct_disagreement"})}
GRADES = ["close", "broad", "partial", "mixed", "direct_disagreement"]

# The Arm B score-pack manifest records a hash of itself. That line does not
# match the manifest bytes in the attachment. Content lines do match.
# This check locks the inconsistency in place so a later edit is visible.
ARMB_MANIFEST_REL = "seals/armB_scores_2026-11-06/MANIFEST.sha256"
ARMB_SELF_LINE_RECORDED = "e390b23a60542e74a8c022c26050a902fcee8ba28252b7995602dcbc9b681374"
ARMB_SELF_LINE_ACTUAL = "e81107966246070772fca7f84b295dd4e145e9ab8c371dbb180db73266f357bb"

EXPECTED_HEADLINES = {
    "gemini_core": (56, 27, 46, 0, 0),
    "claude_core": (56, 27, 48, 0, 0),
    "claude_ext": (51, 31, 41, 0, 0),
    "claude_dk": (19, 4, 10, 0, 4),
    "gemini_dk_descriptive": (19, 7, 14, 0, 2),
    "gemini_dk_validated": (18, 6, 13, 0, 2),
}

ARM_B_EXPECTED = {
    "Manus": {"n": 126, "correct": 87, "CORE": (56, 56), "LIMIT": (19, 15), "EXTENSION": (51, 16)},
    "Codex": {"n": 126, "correct": 7, "CORE": (56, 3), "LIMIT": (19, 1), "EXTENSION": (51, 3)},
    "Grok": {"n": 56, "correct": 19, "CORE": (56, 19)},
    "Kimi": {"n": 126, "correct": 72, "CORE": (56, 52), "LIMIT": (19, 12), "EXTENSION": (51, 8)},
    "Gemini": {"n": 75, "correct": 73, "CORE": (56, 56), "LIMIT": (19, 17)},
    "Claude": {"n": 126, "correct": 114, "CORE": (56, 56), "LIMIT": (19, 13), "EXTENSION": (51, 45)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw == b"":
        return []
    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        rows.append(json.loads(line))
        rows[-1]["_line"] = line_no
    return rows


def classify(returned: str, sealed: str) -> dict:
    exact = returned == sealed
    ready_adj = (not exact) and frozenset({returned, sealed}) in READY
    old_adj = (not exact) and frozenset({returned, sealed}) in OLD_SCALE
    return {
        "exact": exact,
        "ready_adjacent": ready_adj,
        "ready_eoa": exact or ready_adj,
        "old_eoa": exact or old_adj,
        "overturn": returned == "direct_disagreement" and sealed in {"close", "broad"},
        "sealed_dd_reversal": sealed == "direct_disagreement" and returned in {"close", "broad"},
    }


def score_arm_c(rows: list[dict]) -> dict:
    totals = collections.Counter()
    by_entity = collections.defaultdict(collections.Counter)
    confusion = collections.Counter()
    returned_grades = collections.Counter()
    flag_mismatches = 0
    old_differs = 0
    reversals = []
    for row in rows:
        flags = classify(row["q3_grade"], row["sealed_q3_grade"])
        totals["n"] += 1
        totals["exact"] += int(flags["exact"])
        totals["adjacent_only"] += int(flags["ready_adjacent"])
        totals["eoa"] += int(flags["ready_eoa"])
        totals["non_adjacent"] += int(not flags["ready_eoa"])
        totals["overturn"] += int(flags["overturn"])
        totals["sealed_dd_reversal"] += int(flags["sealed_dd_reversal"])
        if flags["old_eoa"] != flags["ready_eoa"]:
            old_differs += 1
        if "exact_match" in row and bool(row["exact_match"]) != flags["exact"]:
            flag_mismatches += 1
        if "adjacent" in row and bool(row["adjacent"]) != flags["ready_adjacent"]:
            flag_mismatches += 1
        if "exact_or_adjacent" in row and bool(row["exact_or_adjacent"]) != flags["ready_eoa"]:
            flag_mismatches += 1
        entity = row["entity"]
        bucket = by_entity[entity]
        bucket["n"] += 1
        bucket["exact"] += int(flags["exact"])
        bucket["adjacent_only"] += int(flags["ready_adjacent"])
        bucket["eoa"] += int(flags["ready_eoa"])
        bucket["non_adjacent"] += int(not flags["ready_eoa"])
        confusion[(row["sealed_q3_grade"], row["q3_grade"])] += 1
        returned_grades[row["q3_grade"]] += 1
        if flags["sealed_dd_reversal"] or not flags["ready_eoa"] or not flags["exact"]:
            if not flags["exact"]:
                reversals.append(
                    {
                        "entity": entity,
                        "era_id": row["era_id"],
                        "year_start": row["year_start"],
                        "year_end": row["year_end"],
                        "sealed_q3_grade": row["sealed_q3_grade"],
                        "returned_q3_grade": row["q3_grade"],
                        "ready_relation": (
                            "exact"
                            if flags["exact"]
                            else "adjacent"
                            if flags["ready_adjacent"]
                            else "non_adjacent"
                        ),
                        "sealed_dd_reversal": flags["sealed_dd_reversal"],
                        "overturning_disagreement": flags["overturn"],
                    }
                )
    return {
        "totals": totals,
        "by_entity": by_entity,
        "confusion": confusion,
        "returned_grades": returned_grades,
        "flag_mismatches": flag_mismatches,
        "old_scale_eoa_differs": old_differs,
        "non_exact": reversals,
    }


def check_manifests() -> list[str]:
    problems = []
    for manifest in sorted(EVIDENCE.rglob("MANIFEST.sha256")):
        rel_manifest = manifest.relative_to(EVIDENCE).as_posix()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            recorded, name = line.split(None, 1)
            name = name.lstrip("*").strip()
            target = manifest.parent / name
            if not target.exists():
                problems.append(f"missing manifest member {rel_manifest}: {name}")
                continue
            actual = sha256(target)
            if rel_manifest == ARMB_MANIFEST_REL and name == "MANIFEST.sha256":
                if recorded != ARMB_SELF_LINE_RECORDED or actual != ARMB_SELF_LINE_ACTUAL:
                    problems.append(
                        "Arm B manifest self-hash changed "
                        f"(recorded {recorded}, actual {actual})"
                    )
                continue
            if actual != recorded:
                problems.append(
                    f"hash mismatch {rel_manifest}: {name} recorded {recorded} actual {actual}"
                )
    return problems


def write_evidence_manifest() -> str:
    lines = []
    files = sorted(path for path in EVIDENCE.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(EVIDENCE).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    text = "\n".join(lines) + "\n"
    destination = ROOT / "EVIDENCE_MANIFEST.sha256"
    destination.write_text(text, encoding="utf-8")
    return sha256(destination)


def main() -> int:
    TABLES.mkdir(exist_ok=True)
    problems = check_manifests()
    files = [path for path in EVIDENCE.rglob("*") if path.is_file()]
    if len(files) != 106:
        problems.append(f"expected 106 evidence files, found {len(files)}")

    arm_c_specs = [
        ("gemini_core", "Gemini", "CORE", "gemini-2.5-pro", EVIDENCE / "seals/armC_scores_2026-11-07/scored_Gemini.jsonl"),
        ("claude_core", "Claude", "CORE", "claude-sonnet-4-6", EVIDENCE / "seals/armC_scores_2026-11-07/scored_Claude.jsonl"),
        ("claude_ext", "Claude", "EXTENSION", "claude-sonnet-4-6", EVIDENCE / "seals/armC_scores_Claude_EXT_DK_2026-11-07/scored_Claude_EXTENSION.jsonl"),
        ("claude_dk", "Claude", "LIMIT", "claude-sonnet-4-6", EVIDENCE / "seals/armC_scores_Claude_EXT_DK_2026-11-07/scored_Claude_LIMIT.jsonl"),
        ("gemini_dk_descriptive", "Gemini", "LIMIT_descriptive", "gemini-2.5-pro", EVIDENCE / "seals/armC_LIMIT_scores_2026-11-07/scored_Gemini_Denmark_descriptive.jsonl"),
        ("gemini_dk_validated", "Gemini", "LIMIT_validated_excl_XV", "gemini-2.5-pro", EVIDENCE / "seals/armC_LIMIT_scores_2026-11-07/scored_Gemini_Denmark_validated.jsonl"),
    ]

    arm_c_rows_out = []
    non_exact_out = []
    confusion_out = []
    arm_c_summary = {}
    for key, model, entity_set, model_id, path in arm_c_specs:
        scored = score_arm_c(load_jsonl(path))
        totals = scored["totals"]
        got = (
            totals["n"],
            totals["exact"],
            totals["eoa"],
            totals["overturn"],
            totals["sealed_dd_reversal"],
        )
        if got != EXPECTED_HEADLINES[key]:
            problems.append(f"{key} headline {got} != {EXPECTED_HEADLINES[key]}")
        if scored["flag_mismatches"]:
            problems.append(f"{key} stored flags disagree with READY recomputation ({scored['flag_mismatches']})")
        if scored["old_scale_eoa_differs"]:
            problems.append(f"{key} old mixed/DD scale changes an exact-or-adjacent count")
        arm_c_summary[key] = {
            "model": model_id,
            "set": entity_set,
            "n": totals["n"],
            "exact": totals["exact"],
            "adjacent_only": totals["adjacent_only"],
            "exact_or_adjacent": totals["eoa"],
            "non_adjacent": totals["non_adjacent"],
            "overturning_disagreements": totals["overturn"],
            "sealed_dd_reversals": totals["sealed_dd_reversal"],
            "returned_grades": dict(scored["returned_grades"]),
            "flag_mismatches": scored["flag_mismatches"],
            "old_scale_eoa_differs": scored["old_scale_eoa_differs"],
            "scored_sha256": sha256(path),
        }
        arm_c_rows_out.append(
            {
                "model": model,
                "model_id": model_id,
                "set": entity_set,
                "n": totals["n"],
                "exact": totals["exact"],
                "adjacent_only": totals["adjacent_only"],
                "exact_or_adjacent": totals["eoa"],
                "non_adjacent": totals["non_adjacent"],
                "overturning_disagreements": totals["overturn"],
                "sealed_dd_reversals_to_close_or_broad": totals["sealed_dd_reversal"],
            }
        )
        for entity, bucket in sorted(scored["by_entity"].items()):
            arm_c_rows_out.append(
                {
                    "model": model,
                    "model_id": model_id,
                    "set": f"{entity_set}:{entity}",
                    "n": bucket["n"],
                    "exact": bucket["exact"],
                    "adjacent_only": bucket["adjacent_only"],
                    "exact_or_adjacent": bucket["eoa"],
                    "non_adjacent": bucket["non_adjacent"],
                    "overturning_disagreements": "",
                    "sealed_dd_reversals_to_close_or_broad": "",
                }
            )
        for sealed, returned in ((s, r) for s in GRADES for r in GRADES):
            confusion_out.append(
                {
                    "model": model,
                    "set": entity_set,
                    "sealed_q3_grade": sealed,
                    "returned_q3_grade": returned,
                    "n": scored["confusion"][(sealed, returned)],
                }
            )
        for item in scored["non_exact"]:
            item = dict(item)
            item["model"] = model
            item["set"] = entity_set
            non_exact_out.append(item)

    # Specific Denmark rows named by the audit addendum and the Claude score pack.
    dk_desc = {
        row["era_id"]: row
        for row in load_jsonl(EVIDENCE / "seals/armC_LIMIT_scores_2026-11-07/scored_Gemini_Denmark_descriptive.jsonl")
    }
    dk_claude = {
        row["era_id"]: row
        for row in load_jsonl(EVIDENCE / "seals/armC_scores_Claude_EXT_DK_2026-11-07/scored_Claude_LIMIT.jsonl")
    }
    expected_rows = {
        ("gemini", "VI"): ("direct_disagreement", "broad"),
        ("gemini", "X"): ("direct_disagreement", "close"),
        ("gemini", "XII"): ("partial", "broad"),
        ("gemini", "XIX"): ("mixed", "close"),
        ("gemini", "XV"): ("broad", "broad"),
        ("claude", "III"): ("direct_disagreement", "broad"),
        ("claude", "V"): ("direct_disagreement", "close"),
        ("claude", "VI"): ("direct_disagreement", "close"),
        ("claude", "X"): ("direct_disagreement", "close"),
    }
    for (who, era), (sealed, returned) in expected_rows.items():
        row = dk_desc[era] if who == "gemini" else dk_claude[era]
        if row["sealed_q3_grade"] != sealed or row["q3_grade"] != returned:
            problems.append(f"{who} Denmark {era} grades {row['sealed_q3_grade']}->{row['q3_grade']} != {sealed}->{returned}")

    xv_raw = next(
        row
        for row in load_jsonl(EVIDENCE / "seals/gemini_armC_LIMIT_2026-11-07/armC_LIMIT.jsonl")
        if row["era_id"] == "XV"
    )
    if len(xv_raw["arm_a_sha256"]) != 77:
        problems.append(f"Gemini Denmark XV arm_a_sha256 length {len(xv_raw['arm_a_sha256'])} != 77")
    if any(row["era_id"] == "XV" for row in load_jsonl(EVIDENCE / "seals/armC_LIMIT_scores_2026-11-07/scored_Gemini_Denmark_validated.jsonl")):
        problems.append("validated Denmark subset still contains XV")
    if any(row["era_id"] == "IX" for row in list(dk_desc.values()) + list(dk_claude.values())):
        problems.append("Denmark IX is present in a scored LIMIT file")

    sealed_core = {}
    for label, path in (
        ("gemini", EVIDENCE / "seals/armC_scores_2026-11-07/scored_Gemini.jsonl"),
        ("claude", EVIDENCE / "seals/armC_scores_2026-11-07/scored_Claude.jsonl"),
    ):
        for row in load_jsonl(path):
            sealed_core.setdefault((row["entity"], row["era_id"], row["year_start"], row["year_end"]), {})[label] = row["sealed_q3_grade"]
    for key, grades in sealed_core.items():
        if len(grades) == 2 and grades["gemini"] != grades["claude"]:
            problems.append(f"CORE sealed grade disagreement {key}: {grades}")
    for era, grow in dk_desc.items():
        crow = dk_claude[era]
        if grow["sealed_q3_grade"] != crow["sealed_q3_grade"] or (grow["year_start"], grow["year_end"]) != (crow["year_start"], crow["year_end"]):
            problems.append(f"Denmark sealed disagreement {era}")

    byte_pairs = [
        ("seals/claude_armC_2026-11-07/armC_CORE.jsonl", "seals/claude_armC_2026-11-07/source_claude_armC_CORE.jsonl"),
        ("seals/claude_armC_EXT_DK_2026-11-07/armC_EXTENSION.jsonl", "seals/claude_armC_EXT_DK_2026-11-07/source_claude_armC_EXTENSION.jsonl"),
        ("seals/claude_armC_EXT_DK_2026-11-07/armC_LIMIT.jsonl", "seals/claude_armC_EXT_DK_2026-11-07/source_claude_armC_LIMIT.jsonl"),
    ]
    for left, right in byte_pairs:
        if (EVIDENCE / left).read_bytes() != (EVIDENCE / right).read_bytes():
            problems.append(f"byte mismatch {left} vs {right}")

    empty_ext = EVIDENCE / "seals/gemini_armC_LIMIT_2026-11-07/armC_EXTENSION.jsonl"
    if empty_ext.read_bytes() != b"" or sha256(empty_ext) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        problems.append("Gemini EXTENSION file is not the sealed empty file")

    arm_b_files = {
        "Manus": EVIDENCE / "seals/armB_scores_2026-11-06/scored_Manus.jsonl",
        "Codex": EVIDENCE / "seals/armB_scores_2026-11-06/scored_Codex.jsonl",
        "Grok": EVIDENCE / "seals/armB_scores_2026-11-06/scored_Grok.jsonl",
        "Kimi": EVIDENCE / "seals/armB_scores_2026-11-06/scored_Kimi.jsonl",
        "Gemini": EVIDENCE / "seals/armB_scores_2026-11-06/scored_Gemini.jsonl",
        "Claude": EVIDENCE / "seals/armB_scores_2026-11-06/scored_Claude.jsonl",
    }
    arm_b_out = []
    entity_out = []
    for model, path in arm_b_files.items():
        rows = load_jsonl(path)
        expected = ARM_B_EXPECTED[model]
        correct = sum(int(row["correct"]) for row in rows)
        chances = [float(row["chance"]) for row in rows]
        mean_chance = sum(chances) / len(chances)
        by_set = collections.defaultdict(lambda: [0, 0])
        by_entity = collections.defaultdict(lambda: [0, 0])
        for row in rows:
            by_set[row["set"]][0] += 1
            by_set[row["set"]][1] += int(row["correct"])
            by_entity[row["entity"]][0] += 1
            by_entity[row["entity"]][1] += int(row["correct"])
        if len(rows) != expected["n"] or correct != expected["correct"]:
            problems.append(f"Arm B {model} {correct}/{len(rows)} != {expected['correct']}/{expected['n']}")
        for set_name, pair in expected.items():
            if set_name in {"n", "correct"}:
                continue
            got = (by_set[set_name][0], by_set[set_name][1])
            if got != pair:
                problems.append(f"Arm B {model} {set_name} {got} != {pair}")
        core_n, core_correct = by_set["CORE"]
        core_chance_rows = [float(row["chance"]) for row in rows if row["set"] == "CORE"]
        core_mean = sum(core_chance_rows) / len(core_chance_rows)
        passes = (core_correct / core_n) >= (2 * core_mean)
        if model == "Codex" and passes:
            problems.append("Codex CORE unexpectedly meets 2x")
        if model != "Codex" and not passes:
            problems.append(f"{model} CORE unexpectedly misses 2x")
        arm_b_out.append(
            {
                "model": model,
                "n": len(rows),
                "correct": correct,
                "accuracy": round(correct / len(rows), 6),
                "mean_row_chance": round(mean_chance, 6),
                "core_n": core_n,
                "core_correct": core_correct,
                "core_mean_row_chance": round(core_mean, 6),
                "core_meets_2x_chance": passes,
                "limit_correct": by_set["LIMIT"][1] if "LIMIT" in by_set else "",
                "limit_n": by_set["LIMIT"][0] if "LIMIT" in by_set else "",
                "extension_correct": by_set["EXTENSION"][1] if "EXTENSION" in by_set else "",
                "extension_n": by_set["EXTENSION"][0] if "EXTENSION" in by_set else "",
                "scored_sha256": sha256(path),
            }
        )
        for entity, (n_ent, correct_ent) in sorted(by_entity.items()):
            entity_out.append(
                {
                    "model": model,
                    "entity": entity,
                    "n": n_ent,
                    "correct": correct_ent,
                }
            )

    coverage = [
        ("Claude", "claude-sonnet-4-6", "A", "CORE+EXT+LIMIT claimed", "hash claimed; bytes absent", "520005f20d08096a266ca20c46cf5939b382c6026c1b7ffa903f4a235c2cddf7"),
        ("Claude", "claude-sonnet-4-6", "B", "CORE 56/56; DK 13/19; EXT 45/51", "scored", "ed54c0a5420b93c68764ddeb44ad970a15f58ac8f8ab6536021a42dacdaeea36"),
        ("Claude", "claude-sonnet-4-6", "C", "CORE 27/56 exact, 48/56 EoA; EXT 31/51 exact, 41/51 EoA; DK 4/19 exact, 10/19 EoA", "numeric majority; Arm A bytes unverified", ""),
        ("Gemini", "gemini-2.5-pro", "A", "no Arm A file in deposit", "claimed hashes admitted non-cryptographic", ""),
        ("Gemini", "gemini-2.5-pro", "B", "CORE 56/56; DK 17/19; EXT not in drop", "scored; not auto-excluded by A/C admission", "c135c2d4a5f1704cc1ec2ab883224ecdae850be20b220713c18b00369f68d66e"),
        ("Gemini", "gemini-2.5-pro", "C", "CORE 27/56 exact, 46/56 EoA; DK 7/19 exact, 14/19 EoA; validated excl. XV 6/18 exact, 13/18 EoA; EXT 0/51", "descriptive only; not protocol-clean", ""),
        ("Kimi", "Kimi K2.5", "A", "CORE 56 rows and Denmark 19 rows sealed", "file hashes match ledger; no central leadership score in deposit", ""),
        ("Kimi", "Kimi K2.5", "B", "CORE 52/56; DK 12/19; EXT 8/51", "scored", "c23f52b0a1fa1f72289e523cf96668aac0b5eb4fec7f688323c30903ec966704"),
        ("Kimi", "Kimi K2.5", "C", "not started", "phone inaccessible; not a failed test", ""),
        ("Grok", "Grok-4.7", "A", "CORE 56 rows sealed", "file hash matches ledger; no central leadership score in deposit", "cbd4b64cdbd947b1a9bf4d16ed1ebe96e3dc6fc45d993afe0eefadaa7042ce0d"),
        ("Grok", "Grok-4.7", "B", "CORE 19/56; DK and EXT absent", "scored; meets 2x on CORE", "c09b54823fd55e4a3a554a1b1d98ab180c7d67d8fb8ed6a87b9ce9ffa2eeee5b"),
        ("Grok", "Grok-4.7", "C", "not returned", "not a failed test", ""),
        ("Codex", "Codex (GPT-6, exact revision unavailable)", "A", "no Arm A file in deposit", "no Arm A metric recorded", ""),
        ("Codex", "Codex (GPT-6, exact revision unavailable)", "B", "CORE 3/56; DK 1/19; EXT 3/51", "scored; fails 2x on CORE", "28e25db21af300cf28bde11848c433c0a6ab06ce468edbcc7d8c32d7f0196a22"),
        ("Codex", "Codex (GPT-6, exact revision unavailable)", "C", "not returned", "not a failed test", ""),
        ("Manus", "Manus agent (LLM version not exposed)", "A", "no Arm A file in deposit", "same-session contamination self-reported on Arm B", ""),
        ("Manus", "Manus agent (LLM version not exposed)", "B", "CORE 56/56; DK 15/19; EXT 16/51", "scored with contamination flag", ""),
        ("Manus", "Manus agent (LLM version not exposed)", "C", "not returned", "credits exhausted; not a failed test", ""),
    ]

    def write_csv(name: str, rows: list[dict]) -> None:
        path = TABLES / name
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv("arm_c_scores.csv", arm_c_rows_out)
    write_csv("arm_c_confusion.csv", confusion_out)
    write_csv("arm_c_nonexact.csv", non_exact_out)
    write_csv("arm_b_scores.csv", arm_b_out)
    write_csv("arm_b_by_entity.csv", entity_out)
    write_csv(
        "coverage.csv",
        [
            {
                "model": model,
                "model_id_as_recorded": model_id,
                "arm": arm,
                "recorded_result": result,
                "status": status,
                "referenced_sha256": referenced,
            }
            for model, model_id, arm, result, status, referenced in coverage
        ],
    )

    manifest_sha = write_evidence_manifest()
    summary = {
        "evidence_files": len(files),
        "evidence_bytes": sum(path.stat().st_size for path in files),
        "evidence_manifest_sha256": manifest_sha,
        "arm_b_manifest_self_hash_matches": False,
        "arm_b_manifest_self_hash_recorded": ARMB_SELF_LINE_RECORDED,
        "arm_b_manifest_self_hash_actual": ARMB_SELF_LINE_ACTUAL,
        "arm_c": arm_c_summary,
        "problems": problems,
        "hold_key_files_present": False,
    }
    (ROOT / "reconciliation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "CAMS personality replication evidence reconciliation",
        f"evidence_files={len(files)}",
        f"evidence_bytes={summary['evidence_bytes']}",
        f"problems={len(problems)}",
    ]
    report_lines.extend(problems)
    report_lines.append("ARM_C")
    for key, item in arm_c_summary.items():
        report_lines.append(
            f"{key} {item['exact']}/{item['n']} exact {item['exact_or_adjacent']}/{item['n']} eoa "
            f"overturns={item['overturning_disagreements']} sealed_dd_reversals={item['sealed_dd_reversals']}"
        )
    report_lines.append("ARM_B")
    for row in arm_b_out:
        report_lines.append(
            f"{row['model']} {row['correct']}/{row['n']} CORE {row['core_correct']}/{row['core_n']} "
            f"2x={row['core_meets_2x_chance']}"
        )
    text = "\n".join(report_lines) + "\n"
    (ROOT / "validation_output.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
