#!/usr/bin/env python3
"""Build the participant input zip from genuine issued files.

The archive is deterministic: sorted names, fixed timestamps, deflate.
It does not include scored outputs, the results report, or answer keys.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "CAMS_personality_participant_inputs_v1.zip"
PREFIX = "CAMS_personality_participant_inputs_v1"

# Genuine issued files, plus the two new warnings that belong with them.
RELATIVE = [
    "START_HERE.md",
    "inputs/CONTAMINATION_AND_SPOILERS.md",
    "inputs/arm_a_standard/MANIFEST.sha256",
    "inputs/arm_a_standard/README.md",
    "inputs/arm_a_standard/algebra_and_rules.md",
    "inputs/arm_a_standard/era_means.csv",
    "inputs/arm_a_standard/era_means_denmark.csv",
    "inputs/arm_a_standard/era_means_extension.csv",
    "inputs/arm_a_standard/experimental_plan_summary.md",
    "inputs/arm_a_standard/node_definitions.md",
    "inputs/arm_a_standard/output_schema.md",
    "inputs/arm_b_blind_historical/COUNTS.txt",
    "inputs/arm_b_blind_historical/MANIFEST.sha256",
    "inputs/arm_b_blind_historical/README_ArmB.md",
    "inputs/arm_b_blind_historical/SPOILER_READ_FIRST.md",
    "inputs/arm_b_blind_historical/era_shortlist.csv",
    "inputs/arm_b_blind_historical/output_schema_ArmB.md",
    "inputs/arm_b_blind_historical/pockets_blind.jsonl",
    "inputs/arm_c_mood/MANIFEST.sha256",
    "inputs/arm_c_mood/README_ArmC.md",
    "inputs/arm_c_mood/SHORT_PASTE.md",
    "inputs/arm_c_mood/output_schema_ArmC.md",
    "inputs/issued_archives/CAMS_ArmB_blind_v1.zip",
    "inputs/issued_archives/CAMS_ArmC_mood_v1.zip",
    "inputs/issued_archives/CAMS_Gemini_ArmA_CSVs_v1.zip",
    "inputs/issued_archives/CAMS_personality_AI_tester_packet_v1.zip",
]


def build_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for relative in RELATIVE:
            path = ROOT / relative
            if not path.is_file():
                raise SystemExit(f"missing {relative}")
            info = zipfile.ZipInfo(f"{PREFIX}/{relative}")
            info.date_time = (2026, 11, 7, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def main() -> int:
    payload = build_bytes()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(payload)
    print(f"{hashlib.sha256(payload).hexdigest()}  {OUT.relative_to(ROOT).as_posix()}  {len(payload)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
