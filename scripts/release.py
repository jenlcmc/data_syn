"""Release gate: validate, annotate, and package the benchmark.

Runs in order:
  1. Strict schema + logic validation
  2. IRS locked harness validation
  3. Property-based invariant checks
  4. Differential validation against Tax-Calculator
  5. Confidence-tier annotation + dataset save
  6. Release card generation
  7. Profile-safe train/dev/test split generation

Usage
-----
    python data_syn/scripts/release.py
    python data_syn/scripts/release.py --skip-differential --skip-irs-harness
    python data_syn/scripts/release.py --version v2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import BenchmarkCase, load_dataset, save_dataset
from scripts.checks import run_invariant_checks
from scripts.harness import run_harness
from scripts.validate import validate


_OUT_OF_SCOPE = [
    "Alternative Minimum Tax (IRC §55-59)",
    "Passive activity loss rules",
    "Net operating loss carryovers",
    "Foreign tax credit",
    "QBI W-2 wage/UBIA limits above threshold",
    "SSTB phase-outs",
    "Earned Income Tax Credit",
    "Premium Tax Credit",
]


# ---------------------------------------------------------------------------
# Differential validation (Tax-Calculator)
# ---------------------------------------------------------------------------

def _run_differential(cases, random_rate, threshold_usd, seed):
    """Import and run the differential validator; return its report dict."""
    from scripts.differential import run_differential_validation
    return run_differential_validation(
        cases=cases,
        random_rate=random_rate,
        threshold_usd=threshold_usd,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Release card
# ---------------------------------------------------------------------------

def _income_band(total: float) -> str:
    if total < 50_000:    return "<50k"
    if total < 100_000:   return "50k-100k"
    if total < 200_000:   return "100k-200k"
    if total < 500_000:   return "200k-500k"
    return "500k+"


def _fmt_counter(counter: Counter) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- {k}: {v}" for k, v in counter.most_common())


def _generate_release_card(
    cases: list[BenchmarkCase],
    differential_report: dict | None,
    output_path: Path,
    version: str,
) -> None:
    source_mix = Counter(c.source for c in cases)
    confidence_mix = Counter(c.confidence_tier for c in cases)
    filing_status: Counter = Counter()
    income_bands: Counter = Counter()
    topics: Counter = Counter()

    for case in cases:
        for tag in case.tags:
            topics[tag] += 1
        if case.facts is not None:
            filing_status[case.facts.filing_status] += 1
            total = float(case.facts.wages + case.facts.spouse_wages + case.facts.business_income)
            income_bands[_income_band(total)] += 1

    diff_lines = ["- unavailable (run scripts/differential.py)"]
    if differential_report:
        r = differential_report.get("results", {})
        diff_lines = [
            f"- comparable_cases: {r.get('comparable_cases')}",
            f"- passed: {r.get('passed')}",
            f"- failed: {r.get('failed')}",
            f"- agreement_rate: {r.get('agreement_rate')}",
        ]

    sections = [
        f"# data_syn Release Card ({version})",
        "",
        "## Source Mix",
        _fmt_counter(source_mix),
        "",
        "## Confidence Tiers",
        _fmt_counter(confidence_mix),
        "",
        "## Coverage",
        "### Filing Status",
        _fmt_counter(filing_status),
        "### Income Bands",
        _fmt_counter(income_bands),
        "### Top Topics",
        _fmt_counter(Counter(dict(topics.most_common(20)))),
        "",
        "## Disagreement Rates vs External Engine",
        *diff_lines,
        "",
        "## Known Out-of-Scope Areas",
        *[f"- {item}" for item in _OUT_OF_SCOPE],
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


# ---------------------------------------------------------------------------
# Profile-safe splits
# ---------------------------------------------------------------------------

def _profile_key(case: BenchmarkCase) -> str:
    return case.profile_id or case.id


def _create_splits(
    cases: list[BenchmarkCase],
    output_dir: Path,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    by_profile: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        by_profile[_profile_key(case)].append(case)

    ids = list(by_profile.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_set = set(ids[:n_train])
    dev_set = set(ids[n_train:n_train + n_dev])
    test_set = set(ids[n_train + n_dev:])

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for case in cases:
        key = _profile_key(case)
        split = "train" if key in train_set else ("dev" if key in dev_set else "test")
        buckets[split].append(case.to_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in buckets.items():
        with open(output_dir / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    manifest = {
        "seed": seed,
        "profiles": {"total": n, "train": len(train_set), "dev": len(dev_set), "test": len(test_set)},
        "cases": {split: len(b) for split, b in buckets.items()},
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump([manifest], fh, indent=2, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release gate for data_syn.")
    parser.add_argument("--dataset", type=Path, default=Path("data_syn/output/benchmark.json"))
    parser.add_argument("--harness", type=Path, default=Path("data_syn/fixtures/irs_locked_examples.json"))
    parser.add_argument("--differential-report", type=Path,
                        default=Path("data_syn/output/differential_taxcalc_report.json"))
    parser.add_argument("--release-card", type=Path, default=Path("data_syn/output/release_card.md"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data_syn/output/splits"))
    parser.add_argument("--random-rate", type=float, default=0.20)
    parser.add_argument("--threshold-usd", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--skip-irs-harness", action="store_true")
    parser.add_argument("--skip-invariants", action="store_true")
    parser.add_argument("--skip-differential", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()

    print("Loading dataset...")
    cases = load_dataset(args.dataset)

    print("Strict validation...")
    errors = validate(cases, strict=True)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors[:30]:
            print(f"  - {e}")
        return 1
    print(f"  OK ({len(cases)} cases)")

    if not args.skip_irs_harness:
        print("IRS harness...")
        ok, report = run_harness(args.harness)
        if not ok:
            print(f"  FAILED: {report['failure_count']} failures")
            for item in report["failures"][:20]:
                print(f"    - {item}")
            return 1
        print(f"  OK ({report['locked_examples']} locked)")

    if not args.skip_invariants:
        print("Invariant checks...")
        failures = run_invariant_checks(seed=args.seed)
        if failures:
            print(f"  FAILED: {len(failures)}")
            for f in failures[:20]:
                print(f"    - {f}")
            return 1
        print("  OK")

    diff_report: dict | None = None
    if not args.skip_differential:
        print("Differential validation (Tax-Calculator)...")
        try:
            diff_report = _run_differential(
                cases, args.random_rate, args.threshold_usd, args.seed
            )
            args.differential_report.parent.mkdir(parents=True, exist_ok=True)
            with open(args.differential_report, "w", encoding="utf-8") as fh:
                json.dump(diff_report, fh, indent=2, ensure_ascii=False)
            r = diff_report.get("results", {})
            print(f"  comparable={r.get('comparable_cases')}, "
                  f"failed={r.get('failed')}, "
                  f"agreement={r.get('agreement_rate')}")
        except ImportError:
            print("  SKIPPED (taxcalc not installed)")

    print("Saving dataset with confidence metadata...")
    save_dataset(cases, args.dataset)

    print("Generating release card...")
    _generate_release_card(cases, diff_report, args.release_card, args.version)
    print(f"  {args.release_card}")

    if not args.skip_splits:
        print("Creating profile-safe splits...")
        manifest = _create_splits(cases, args.splits_dir, seed=args.seed)
        p = manifest["profiles"]
        print(f"  profiles: train={p['train']}, dev={p['dev']}, test={p['test']}")

    print("Release gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
