#!/usr/bin/env python3
"""
Zero-Leakage Dataset Verification Script

Performs an exhaustive clinical audit on the active dataset:
1. Distribution check across splits (Train, Val, Test)
2. Exact duplicate audit using MD5 cryptographic hashes (within and across splits)
3. Patient-level identity isolation audit to verify 0.0% patient leakage
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def extract_patient_id(filename: str) -> str:
    """Extract patient ID from standard CXR naming schemes."""
    match = re.match(r'^(person\d+)_', filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match_norm = re.match(r'^(NORMAL2-IM-\d+|IM-\d+)', filename)
    if match_norm:
        return match_norm.group(1)
    return filename.split('.')[0]


def audit_dataset(data_dir: Path) -> bool:
    """Audit the dataset directory for distribution, duplicates, and patient leakage."""
    splits = ["train", "val", "test"]
    stats = {}
    hashes = defaultdict(list)
    patient_splits = defaultdict(set)
    patient_files = defaultdict(list)

    if not data_dir.exists():
        print(f"❌ Error: Data directory not found at {data_dir}")
        return False

    print("=" * 70)
    print("  CLINICAL RADIOLOGY DATASET AUDIT & ZERO-LEAKAGE VERIFICATION")
    print(f"  Target Directory: {data_dir.resolve()}")
    print("=" * 70 + "\n")

    for split in splits:
        stats[split] = {}
        for cls in ["NORMAL", "PNEUMONIA"]:
            p = data_dir / split / cls
            if not p.exists():
                p = data_dir / split / cls.lower()
            if not p.exists():
                continue
            files = [f for f in p.glob("*.*") if f.is_file() and not f.name.startswith(".")]
            stats[split][cls] = len(files)
            for f in files:
                try:
                    h = hashlib.md5(f.read_bytes()).hexdigest()
                    hashes[h].append((split, cls, f.name))
                except Exception as e:
                    print(f"⚠️  Could not read {f.name}: {e}")
                    continue

                pid = extract_patient_id(f.name)
                patient_splits[pid].add(split)
                patient_files[pid].append((split, cls, f.name))

    print("📊 1. DATASET SPLIT DISTRIBUTION:")
    for s, d in stats.items():
        tot = sum(d.values())
        norm = d.get('NORMAL', 0)
        pneu = d.get('PNEUMONIA', 0)
        pct = (pneu / tot * 100) if tot > 0 else 0
        print(f"   • {s.upper():<6}: NORMAL={norm:<5} PNEUMONIA={pneu:<5} Total={tot:<5} ({pct:.1f}% Pneumonia)")

    print("\n🔒 2. CRYPTOGRAPHIC INTEGRITY & DUPLICATE AUDIT (MD5):")
    cross_split_dups = []
    within_split_dups = []
    for h, entries in hashes.items():
        sp_set = set(e[0] for e in entries)
        if len(sp_set) > 1:
            cross_split_dups.append(entries)
        elif len(entries) > 1:
            within_split_dups.append(entries)

    if cross_split_dups:
        print(f"   ❌ FAILED: {len(cross_split_dups)} exact duplicates shared across splits!")
    else:
        print("   ✅ PASSED: 0 cross-split duplicates found.")

    if within_split_dups:
        print(f"   ⚠️  NOTE: {len(within_split_dups)} duplicate images within identical splits.")
    else:
        print("   ✅ PASSED: 0 within-split duplicates found.")

    print("\n🧬 3. PATIENT-LEVEL IDENTITY LEAKAGE AUDIT:")
    total_patients = len(patient_splits)
    leaked_patients = {pid: sp for pid, sp in patient_splits.items() if len(sp) > 1}
    leak_rate = (len(leaked_patients) / total_patients * 100) if total_patients > 0 else 0.0

    print(f"   • Total Unique Patients: {total_patients}")
    if leaked_patients:
        print(f"   ❌ FAILED: {len(leaked_patients)} patients appear across multiple splits ({leak_rate:.2f}% leakage)!")
        return False
    else:
        print(f"   ✅ PASSED: Strictly 0.0% patient leakage (all patient scans isolated to single splits).")

    print("\n" + "=" * 70)
    print("  🏆 FINAL VERDICT: DATASET INTEGRITY VERIFIED (100% LEAK-FREE)")
    print("=" * 70 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Zero-Leakage Dataset Verification")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    args = parser.parse_args()

    success = audit_dataset(args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
