"""Build the synthetic tax benchmark dataset.

This is the single entry point for generating the full benchmark.  It
orchestrates all three tiers, writes the combined JSON output, and prints
a summary table.

Usage
-----
    python data_syn/build_dataset.py
    python data_syn/build_dataset.py --tiers 1 2
    python data_syn/build_dataset.py --output data_syn/output/my_dataset.json
    python data_syn/build_dataset.py --dry-run

Options
-------
--tiers       Space-separated list of tiers to build (default: 1 2 3).
--output      Output JSON file path (default: data_syn/output/benchmark.json).
--dry-run     Validate the pipeline without writing files.
--verbose     Print each generated case ID and ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the project root or from within data_syn/
sys.path.insert(0, str(Path(__file__).parent))

from cases.tier1 import build_tier1_cases
from cases.tier2 import build_tier2_cases
from cases.tier3 import build_tier3_cases
from config import OUTPUT_DIR
from schema import BenchmarkCase, save_dataset


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the synthetic federal tax benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Which tiers to build (default: 1 2 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "benchmark.json",
        help="Destination JSON file (default: data_syn/output/benchmark.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline and validate without writing output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each case ID and ground truth as it is generated.",
    )
    return parser.parse_args()


def _print_summary(cases: list[BenchmarkCase]) -> None:
    """Print a breakdown table to stdout."""
    from collections import Counter

    tier_counts   = Counter(c.tier for c in cases)
    style_counts  = Counter(c.style for c in cases)
    domain_counts = Counter(c.domain for c in cases)
    diff_counts   = Counter(c.difficulty for c in cases)

    print()
    print("=" * 60)
    print("Synthetic Tax Benchmark — Build Summary")
    print("=" * 60)
    print(f"Total cases: {len(cases)}")
    print()

    print("By tier:")
    for tier in sorted(tier_counts):
        print(f"  Tier {tier}: {tier_counts[tier]}")

    print()
    print("By style:")
    for style in sorted(style_counts):
        print(f"  {style:<18}: {style_counts[style]}")

    print()
    print("By domain:")
    for domain in sorted(domain_counts):
        print(f"  {domain:<25}: {domain_counts[domain]}")

    print()
    print("By difficulty:")
    for diff in ["basic", "intermediate", "advanced"]:
        print(f"  {diff:<14}: {diff_counts.get(diff, 0)}")

    # Ground truth type breakdown
    gt_types = Counter(c.ground_truth_type for c in cases)
    print()
    print("By ground truth type:")
    for gt in sorted(gt_types):
        print(f"  {gt:<18}: {gt_types[gt]}")

    # Flag any cases needing external validation
    needs_validation = [c for c in cases if c.explanation and "Flag" in c.explanation]
    if needs_validation:
        print()
        print(f"Cases flagged for external validation: {len(needs_validation)}")
        print("  (AMT review or QBI W-2 wage limitations — use PolicyEngine-US)")

    print("=" * 60)


def _validate(cases: list[BenchmarkCase]) -> list[str]:
    """Run basic integrity checks on the generated cases.

    Returns a list of error messages (empty if all checks pass).
    """
    errors: list[str] = []
    seen_ids: set[str] = set()

    for case in cases:
        # Unique ID
        if case.id in seen_ids:
            errors.append(f"Duplicate case ID: {case.id}")
        seen_ids.add(case.id)

        # Required fields
        if not case.question.strip():
            errors.append(f"{case.id}: empty question")

        if case.ground_truth is None:
            errors.append(f"{case.id}: ground_truth is None")

        # Numeric cases must have a float ground truth
        if case.style == "numeric" and not isinstance(case.ground_truth, (int, float)):
            errors.append(
                f"{case.id}: numeric case has non-numeric ground_truth "
                f"({case.ground_truth!r})"
            )

        # Entailment cases must be "Yes" or "No"
        if case.style == "entailment" and case.ground_truth not in ("Yes", "No"):
            errors.append(
                f"{case.id}: entailment case has invalid ground_truth "
                f"({case.ground_truth!r}) — expected 'Yes' or 'No'"
            )

        # MCQ cases must have choices and a valid key
        if case.style == "mcq":
            if not case.choices:
                errors.append(f"{case.id}: MCQ case missing choices")
            elif case.ground_truth not in ("A", "B", "C", "D"):
                errors.append(
                    f"{case.id}: MCQ ground_truth must be A/B/C/D, "
                    f"got {case.ground_truth!r}"
                )

        # Tier 1 cases must have non-zero reasoning steps
        if case.tier == 1 and not case.reasoning_steps:
            errors.append(f"{case.id}: Tier 1 case has no reasoning_steps")

    return errors


def main() -> int:
    args = _build_args()
    all_cases: list[BenchmarkCase] = []
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Tier 1 — Numeric (deterministic engine)
    # ------------------------------------------------------------------
    if 1 in args.tiers:
        print("Building Tier 1 (numeric) cases...", end=" ", flush=True)
        t1_cases = build_tier1_cases()
        all_cases.extend(t1_cases)
        print(f"{len(t1_cases)} cases.")
        if args.verbose:
            for c in t1_cases:
                print(f"  [{c.id}] {c.question[:60]!r} → {c.ground_truth}")

    # ------------------------------------------------------------------
    # Tier 2 — Entailment (statute application)
    # ------------------------------------------------------------------
    if 2 in args.tiers:
        print("Building Tier 2 (entailment) cases...", end=" ", flush=True)
        t2_cases = build_tier2_cases()
        all_cases.extend(t2_cases)
        print(f"{len(t2_cases)} cases.")
        if args.verbose:
            for c in t2_cases:
                print(f"  [{c.id}] → {c.ground_truth}")

    # ------------------------------------------------------------------
    # Tier 3 — Q&A / MCQ / Scenario
    # ------------------------------------------------------------------
    if 3 in args.tiers:
        print("Building Tier 3 (Q&A / MCQ / scenario) cases...", end=" ", flush=True)
        t3_cases = build_tier3_cases()
        all_cases.extend(t3_cases)
        print(f"{len(t3_cases)} cases.")
        if args.verbose:
            for c in t3_cases:
                label = (
                    c.ground_truth[:60]
                    if isinstance(c.ground_truth, str)
                    else c.ground_truth
                )
                print(f"  [{c.id}] {c.style} → {label!r}")

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    print("Validating...", end=" ", flush=True)
    errors = _validate(all_cases)
    if errors:
        print(f"FAILED ({len(errors)} errors).")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("OK.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(all_cases)
    elapsed = time.monotonic() - t0
    print(f"\nBuild time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[dry-run] Output not written.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(all_cases, args.output)
    print(f"\nDataset written to: {args.output}")
    print(f"Total cases: {len(all_cases)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
