#!/usr/bin/env python3
"""Check this public kit against the sealed Arm C packet and its own scripts.

Reads the sealed packet to compare bytes. Does not modify it and does not score.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEAL = REPO / "analysis/cams_personality_independent_ai_replication_2026-11-07/evidence/packets/arm_c_mood"
COPY = ROOT / "packet/arm_c_mood"
PACKET_FILES = ("README_ArmC.md", "output_schema_ArmC.md", "SHORT_PASTE.md", "MANIFEST.sha256")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems = []
    if not SEAL.is_dir():
        problems.append(f"sealed Arm C packet missing at {SEAL}")
    else:
        manifest = (SEAL / "MANIFEST.sha256").read_text(encoding="utf-8")
        for name in PACKET_FILES:
            left = SEAL / name
            right = COPY / name
            if not left.is_file() or not right.is_file():
                problems.append(f"missing packet file {name}")
                continue
            if left.read_bytes() != right.read_bytes():
                problems.append(f"kit copy differs from sealed packet: {name}")
            if name == "MANIFEST.sha256":
                continue
            recorded = None
            for line in manifest.splitlines():
                if line.endswith("  " + name) or line.endswith(" " + name):
                    recorded = line.split()[0]
            actual = sha256(left)
            if recorded != actual:
                problems.append(f"sealed manifest mismatch for {name}")
    script = ROOT / "scripts/sha256_file.py"
    empty = ROOT / "scripts/.empty_hash_probe"
    empty.write_bytes(b"")
    try:
        probe = subprocess.run(
            [sys.executable, str(script), str(empty)],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        empty.unlink(missing_ok=True)
    if probe.returncode != 0 or not probe.stdout.startswith(
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ):
        problems.append("sha256_file.py did not hash empty bytes to the SHA-256 of empty input")
    shape = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_return_shape.py"), "--self-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    if shape.returncode != 0:
        problems.append("check_return_shape.py self-test failed")
        problems.append(shape.stdout.strip())
        problems.append(shape.stderr.strip())

    inputs = ROOT / "inputs"
    fresh = inputs / "arm_c_mood"
    for name in PACKET_FILES:
        if (COPY / name).read_bytes() != (fresh / name).read_bytes():
            problems.append(f"inputs Arm C copy differs from packet copy: {name}")
    issued = {
        "CAMS_personality_AI_tester_packet_v1.zip": "da94f1c3f405b4c8f4229178509bf400f5661c694cfb38b909bdc1a70b5d5d1e",
        "CAMS_ArmB_blind_v1.zip": "dceee12e89c1b07b9f63e4ce26836f51fb231089b621afc4f9d8f819ab73aae9",
        "CAMS_ArmC_mood_v1.zip": "23fce155fa687d5c9b88e41c25cc9611537dbb8072086bf13a7741e2b24f6f26",
        "CAMS_Gemini_ArmA_CSVs_v1.zip": "a9d36f5f5273c50359e3f9a45b04490dd9ffda48953836f10734972177781849",
    }
    for name, expected in issued.items():
        path = inputs / "issued_archives" / name
        if not path.is_file() or sha256(path) != expected:
            problems.append(f"issued archive hash mismatch: {name}")
    for manifest_path in (
        inputs / "MANIFEST.sha256",
        inputs / "arm_a_standard" / "MANIFEST.sha256",
        inputs / "arm_b_blind_historical" / "MANIFEST.sha256",
    ):
        base = manifest_path.parent
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            recorded, name = line.split(None, 1)
            name = name.removeprefix("./")
            target = base / name
            if not target.is_file() or sha256(target) != recorded:
                problems.append(f"manifest mismatch {manifest_path.name}: {name}")
    counts = (inputs / "arm_b_blind_historical" / "COUNTS.txt").read_text(encoding="utf-8")
    if "58ff7c4166d0293b2f5ad6cfee33db5d3d46a3b5b9932c3a84594e507c8b6ea6" not in counts:
        problems.append("COUNTS.txt no longer records the stale Arm B zip hash")
    if sha256(inputs / "issued_archives" / "CAMS_ArmB_blind_v1.zip").startswith("58ff7c41"):
        problems.append("issued Arm B zip unexpectedly matches the stale COUNTS hash")
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name in {"arm_b_key.csv", "arm_c_key.csv"}:
            problems.append(f"withheld key file present: {path}")
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_participant_zip

    built = build_participant_zip.build_bytes()
    committed = (ROOT / "CAMS_personality_participant_inputs_v1.zip").read_bytes()
    if built != committed:
        problems.append("participant zip does not match the deterministic build")
    print(f"problems={len(problems)}")
    for item in problems:
        if item:
            print(item)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
