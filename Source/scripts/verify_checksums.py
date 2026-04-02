#!/usr/bin/env python3
"""
verify_checksums.py — Verify SHA256 hashes of model checkpoints and datasets.

Usage:
    python scripts/verify_checksums.py

Reads checksums.json in the project root and verifies every listed file.
Prints PASS/FAIL per file and an overall result.
Exit code 0 on all-pass, 1 on any failure.
"""

import hashlib
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKSUMS_FILE = os.path.join(PROJECT_ROOT, "checksums.json")


def sha256(filepath: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not os.path.exists(CHECKSUMS_FILE):
        print(f"FAIL: checksums.json not found at {CHECKSUMS_FILE}")
        return 1

    with open(CHECKSUMS_FILE, "r") as f:
        checksums = json.load(f)

    if not checksums:
        print("FAIL: checksums.json is empty")
        return 1

    all_pass = True
    print(f"Verifying {len(checksums)} file(s) …\n")

    for rel_path, info in checksums.items():
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        expected_hash = info["sha256"]
        expected_size = info.get("size_bytes")

        if not os.path.exists(abs_path):
            print(f"  FAIL  {rel_path}  — file not found")
            all_pass = False
            continue

        actual_size = os.path.getsize(abs_path)
        if expected_size is not None and actual_size != expected_size:
            print(f"  FAIL  {rel_path}  — size mismatch "
                  f"(expected {expected_size:,}, got {actual_size:,})")
            all_pass = False
            continue

        actual_hash = sha256(abs_path)
        if actual_hash == expected_hash:
            print(f"  PASS  {rel_path}  ({actual_size:,} bytes)")
        else:
            print(f"  FAIL  {rel_path}  — SHA256 mismatch")
            print(f"         expected: {expected_hash}")
            print(f"         actual:   {actual_hash}")
            all_pass = False

    print()
    if all_pass:
        print("OVERALL: PASS — all checksums verified ✓")
        return 0
    else:
        print("OVERALL: FAIL — one or more checksums did not match")
        return 1


if __name__ == "__main__":
    sys.exit(main())
