"""Differential validation against Tax-Calculator.

Implements a layered cross-engine check for Tier 1 numeric cases:
  - Mandatory sample: all advanced + all externally flagged cases
  - Random sample: 20% of remaining cases (configurable)
  - Auto-flag disagreement when abs(delta) > threshold (default $25)

This script can also update confidence tiers in the dataset:
  - Tier A: engine + external engine agree
  - Tier B: engine only
  - Tier C: text/scenario judged
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.taxcalc import TaxCalcResult, compute_taxcalc
from schema import BenchmarkCase, load_dataset, save_dataset


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run differential validation against Tax-Calculator."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data_syn/output/benchmark.json"),
        help="Input dataset JSON path.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("data_syn/output/differential_taxcalc_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Optional output dataset path with confidence tiers and check metadata.",
    )
    parser.add_argument(
        "--random-rate",
        type=float,
        default=0.20,
        help="Random sample fraction from non-mandatory cases (default: 0.20).",
    )
    parser.add_argument(
        "--threshold-usd",
        type=float,
        default=25.0,
        help="Absolute-difference threshold for agreement (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling.",
    )
    parser.add_argument(
        "--engine",
        choices=["taxcalc", "taxsim"],
        default="taxcalc",
        help="External engine adapter (taxsim currently not implemented).",
    )
    return parser.parse_args()



def _target_metric(case: BenchmarkCase) -> str:
    """Infer comparable metric from the case question text."""
    question = case.question.lower()

    if "adjusted gross income" in question or "(agi)" in question:
        return "agi"
    if "total deduction" in question and "taxable income" in question:
        return "deduction_used"
    if "federal taxable income" in question or "taxable income" in question:
        return "taxable_income"
    if "net federal income tax liability" in question or "net tax liability" in question:
        return "net_tax"
    if "self-employment (se) tax" in question or "self-employment tax" in question:
        return "se_tax"
    if "child tax credit" in question:
        return "child_tax_credit"
    return "unknown"



def _metric_value(metric: str, external: TaxCalcResult) -> float | None:
    """Map metric name to comparable Tax-Calculator value."""
    if metric == "agi":
        return external.agi
    if metric == "taxable_income":
        return external.taxable_income
    if metric == "net_tax":
        return external.net_tax_proxy
    if metric == "se_tax":
        return external.se_tax
    if metric == "deduction_used":
        return external.deduction_used
    if metric == "child_tax_credit":
        return external.child_tax_credit
    return None



def run_differential_validation(
    cases: list[BenchmarkCase],
    random_rate: float = 0.20,
    threshold_usd: float = 25.0,
    seed: int = 42,
) -> dict:
    """Run differential validation and update in-memory cases metadata."""
    rng = random.Random(seed)

    # Baseline confidence tiers before differential promotion.
    for case in cases:
        if case.tier == 3:
            case.confidence_tier = "C"
        else:
            case.confidence_tier = "B"
        case.external_engine = ""
        case.external_delta_usd = None
        case.external_check_passed = None

    tier1_cases = [
        c
        for c in cases
        if c.tier == 1 and c.style == "numeric" and c.facts is not None
    ]
    tier1_by_profile: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in tier1_cases:
        tier1_by_profile[case.profile_id or case.id].append(case)

    mandatory = [
        c
        for c in tier1_cases
        if c.difficulty == "advanced" or ("Flagged" in (c.explanation or ""))
    ]
    mandatory_ids = {c.id for c in mandatory}

    optional = [c for c in tier1_cases if c.id not in mandatory_ids]
    optional_n = math.ceil(len(optional) * max(0.0, min(1.0, random_rate)))
    sampled_optional = rng.sample(optional, k=min(optional_n, len(optional)))

    selected = mandatory + sampled_optional

    external_cache: dict[str, TaxCalcResult] = {}
    profile_checks: dict[str, list[bool]] = defaultdict(list)

    per_case_results: list[dict] = []
    comparable_count = 0
    pass_count = 0
    fail_count = 0
    skipped_count = 0

    for case in selected:
        metric = _target_metric(case)
        profile_id = case.profile_id or case.id

        if metric == "unknown":
            skipped_count += 1
            per_case_results.append(
                {
                    "case_id": case.id,
                    "profile_id": profile_id,
                    "metric": metric,
                    "status": "skipped_no_mapping",
                }
            )
            continue

        if profile_id not in external_cache:
            external_cache[profile_id] = compute_taxcalc(case.facts, tax_year=case.tax_year)

        external = external_cache[profile_id]
        external_value = _metric_value(metric, external)
        engine_value = float(case.ground_truth)

        if external_value is None:
            skipped_count += 1
            per_case_results.append(
                {
                    "case_id": case.id,
                    "profile_id": profile_id,
                    "metric": metric,
                    "status": "skipped_no_mapping",
                }
            )
            continue

        abs_diff = abs(engine_value - external_value)
        passed = abs_diff <= threshold_usd

        comparable_count += 1
        if passed:
            pass_count += 1
        else:
            fail_count += 1

        case.external_engine = "tax_calculator"
        case.external_delta_usd = round(abs_diff, 2)
        case.external_check_passed = passed

        profile_checks[profile_id].append(passed)

        per_case_results.append(
            {
                "case_id": case.id,
                "profile_id": profile_id,
                "metric": metric,
                "status": "passed" if passed else "failed",
                "engine_value": round(engine_value, 2),
                "external_value": round(float(external_value), 2),
                "abs_diff": round(abs_diff, 2),
                "threshold_usd": threshold_usd,
            }
        )

    # Promote Tier 1 profiles with full comparable agreement to confidence tier A.
    checked_profiles = 0
    agreed_profiles = 0
    for profile_id, profile_cases in tier1_by_profile.items():
        checks = profile_checks.get(profile_id, [])
        if checks:
            checked_profiles += 1
            if all(checks):
                agreed_profiles += 1
                for case in profile_cases:
                    case.confidence_tier = "A"
                    case.verified_by = "tax_calculator"
                    case.external_engine = "tax_calculator"
                    if case.external_check_passed is None:
                        case.external_check_passed = True

    # Tier 2 always remains B; Tier 3 always C.
    for case in cases:
        if case.tier == 2:
            case.confidence_tier = "B"
        elif case.tier == 3:
            case.confidence_tier = "C"

    confidence_counts = Counter(c.confidence_tier for c in cases)

    return {
        "engine": "tax_calculator",
        "sampling": {
            "seed": seed,
            "random_rate": random_rate,
            "threshold_usd": threshold_usd,
            "selected_cases": len(selected),
            "mandatory_cases": len(mandatory),
            "optional_pool": len(optional),
            "optional_sampled": len(sampled_optional),
        },
        "results": {
            "comparable_cases": comparable_count,
            "passed": pass_count,
            "failed": fail_count,
            "skipped": skipped_count,
            "agreement_rate": (pass_count / comparable_count) if comparable_count else None,
            "checked_profiles": checked_profiles,
            "agreed_profiles": agreed_profiles,
            "profile_agreement_rate": (
                agreed_profiles / checked_profiles if checked_profiles else None
            ),
        },
        "confidence_tier_counts": dict(confidence_counts),
        "case_results": per_case_results,
    }



def main() -> int:
    args = _args()
    if args.engine == "taxsim":
        raise NotImplementedError(
            "taxsim adapter is not implemented in this repository yet. "
            "Use --engine taxcalc for differential validation."
        )

    dataset_path = args.dataset
    output_report = args.output_report
    output_dataset = args.output_dataset or dataset_path

    cases = load_dataset(dataset_path)
    report = run_differential_validation(
        cases=cases,
        random_rate=args.random_rate,
        threshold_usd=args.threshold_usd,
        seed=args.seed,
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(cases, output_dataset)

    results = report["results"]
    print("Differential validation complete.")
    print(f"  comparable cases : {results['comparable_cases']}")
    print(f"  passed           : {results['passed']}")
    print(f"  failed           : {results['failed']}")
    print(f"  skipped          : {results['skipped']}")
    print(f"  agreement rate   : {results['agreement_rate']}")
    print(f"  profile agreement: {results['profile_agreement_rate']}")
    print(f"Report: {output_report}")
    print(f"Dataset: {output_dataset}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
