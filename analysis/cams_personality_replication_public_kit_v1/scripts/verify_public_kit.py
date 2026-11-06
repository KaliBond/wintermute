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
    print(f"problems={len(problems)}")
    for item in problems:
        if item:
            print(item)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
