#!/usr/bin/env python3
"""SHA-256 of file bytes, computed here with hashlib.

This does not accept a typed digest. If a session cannot run this script,
stop. Do not invent a 64-character hex string and call it a hash.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clock_stamp() -> str:
    if ZoneInfo is not None:
        try:
            now = datetime.now(ZoneInfo("Australia/Sydney"))
            return now.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            pass
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M %Z")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash files with SHA-256 over their bytes.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--ledger",
        action="store_true",
        help="Also print one ledger line using this computer's clock.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--arm", default="")
    parser.add_argument("--entity-set", default="")
    args = parser.parse_args()
    missing = [str(path) for path in args.paths if not path.is_file()]
    if missing:
        print("not a file:", ", ".join(missing), file=sys.stderr)
        return 2
    rows = []
    for path in args.paths:
        digest = sha256(path)
        rows.append((digest, path))
        print(f"{digest}  {path.as_posix()}")
    if args.ledger:
        stamp = clock_stamp()
        joined = "; ".join(f"{path.name}={digest}" for digest, path in rows)
        print(
            f"model={args.model or 'UNSTATED'}; arm={args.arm or 'UNSTATED'}; "
            f"entity_set={args.entity_set or 'UNSTATED'}; sha256_output={joined}; "
            f"timestamp={stamp}; hash_method=hashlib.sha256_over_file_bytes"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
