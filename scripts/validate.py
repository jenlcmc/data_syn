"""Validate a generated benchmark JSON file.

Checks structural integrity, ground truth types, duplicate IDs, and
cross-tier coverage.  Exits with code 0 on success, 1 on any failure.

Usage
-----
    python data_syn/scripts/validate.py data_syn/output/benchmark.json
    python data_syn/scripts/validate.py data_syn/output/benchmark.json --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import BenchmarkCase, load_dataset


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a synthetic tax benchmark JSON file."
    )
    parser.add_argument("dataset", type=Path, help="Path to benchmark.json.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if any case lacks reasoning_steps (Tier 1) "
            "or required references/context (Tier 2 and Tier 3)."
        ),
    )
    return parser.parse_args()


def _validate_facts(prefix: str, facts) -> list[str]:
    """Validate cross-field logical consistency for TaxpayerFacts."""
    errors: list[str] = []

    if facts.qualified_dividends > facts.ordinary_dividends:
        errors.append(
            f"{prefix} qualified_dividends exceeds ordinary_dividends."
        )

    if facts.child_care_expenses > 0 and facts.num_qualifying_children <= 0:
        errors.append(
            f"{prefix} child_care_expenses present without qualifying children."
        )

    if facts.self_employed_health_ins > 0 and facts.business_income <= 0:
        errors.append(
            f"{prefix} self_employed_health_ins present without business_income."
        )

    if facts.business_income > 0 and facts.self_employed_health_ins > facts.business_income:
        errors.append(
            f"{prefix} self_employed_health_ins exceeds business_income."
        )

    if facts.sep_simple_ira_deduction > 0 and facts.business_income <= 0:
        errors.append(
            f"{prefix} sep_simple_ira_deduction present without business_income."
        )

    if facts.business_income > 0 and facts.sep_simple_ira_deduction > (facts.business_income * 0.25 + 1):
        errors.append(
            f"{prefix} sep_simple_ira_deduction exceeds 25% of business_income."
        )

    if facts.filing_status != "mfj":
        if facts.spouse_wages > 0:
            errors.append(f"{prefix} spouse_wages present for non-MFJ filing status.")
        if facts.age_spouse not in (None, 0):
            errors.append(f"{prefix} age_spouse present for non-MFJ filing status.")

    if facts.filing_status == "mfj" and facts.spouse_wages > 0 and facts.age_spouse is None:
        errors.append(f"{prefix} MFJ with spouse_wages is missing age_spouse.")

    if facts.filing_status == "qss" and facts.num_qualifying_children <= 0:
        errors.append(
            f"{prefix} qss filing status should include at least one qualifying child."
        )

    if facts.social_security_benefits > 0 and facts.age_primary < 60:
        errors.append(
            f"{prefix} social_security_benefits present for taxpayer younger than 60."
        )

    if facts.pension_income > 0 and facts.age_primary < 50:
        errors.append(
            f"{prefix} pension_income present for taxpayer younger than 50."
        )

    if facts.ira_distributions > 0 and facts.age_primary < 50:
        errors.append(
            f"{prefix} ira_distributions present for taxpayer younger than 50."
        )

    if facts.rental_income_net > 0 and (facts.real_estate_tax + facts.mortgage_interest) == 0:
        errors.append(
            f"{prefix} rental_income_net present without housing cost indicators."
        )

    if facts.covered_by_workplace_plan and (facts.wages + facts.spouse_wages) <= 0:
        errors.append(
            f"{prefix} covered_by_workplace_plan set with no W-2 wage income."
        )

    earned_compensation = facts.wages + facts.spouse_wages + facts.business_income
    if facts.ira_contribution > earned_compensation:
        errors.append(
            f"{prefix} ira_contribution exceeds earned compensation."
        )

    return errors


def validate(cases: list[BenchmarkCase], strict: bool = False) -> list[str]:
    """Return a list of error strings.  Empty list means fully valid."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    valid_styles      = {"numeric", "entailment", "qa", "mcq", "scenario"}
    valid_domains     = {"federal_income_tax", "self_employment", "both"}
    valid_difficulties = {"basic", "intermediate", "advanced"}
    valid_gt_types    = {"numeric_usd", "boolean_str", "choice_key", "text"}

    for c in cases:
        prefix = f"[{c.id}]"

        # Unique IDs
        if c.id in seen_ids:
            errors.append(f"{prefix} Duplicate case ID.")
        seen_ids.add(c.id)

        # Enum fields
        if c.style not in valid_styles:
            errors.append(f"{prefix} Unknown style: {c.style!r}.")
        if c.domain not in valid_domains:
            errors.append(f"{prefix} Unknown domain: {c.domain!r}.")
        if c.difficulty not in valid_difficulties:
            errors.append(f"{prefix} Unknown difficulty: {c.difficulty!r}.")
        if c.ground_truth_type not in valid_gt_types:
            errors.append(f"{prefix} Unknown ground_truth_type: {c.ground_truth_type!r}.")
        if c.tier not in (1, 2, 3):
            errors.append(f"{prefix} Invalid tier: {c.tier}.")

        # Non-empty required text
        if not c.question.strip():
            errors.append(f"{prefix} Empty question.")
        if c.ground_truth is None:
            errors.append(f"{prefix} ground_truth is None.")

        # Style-specific checks
        if c.style == "numeric":
            if not isinstance(c.ground_truth, (int, float)):
                errors.append(
                    f"{prefix} Numeric case has non-numeric ground_truth "
                    f"({type(c.ground_truth).__name__})."
                )
            if c.ground_truth_type != "numeric_usd":
                errors.append(
                    f"{prefix} Numeric case should have ground_truth_type='numeric_usd'."
                )

        elif c.style == "entailment":
            if c.ground_truth not in ("Yes", "No"):
                errors.append(
                    f"{prefix} Entailment ground_truth must be 'Yes' or 'No', "
                    f"got {c.ground_truth!r}."
                )

        elif c.style == "mcq":
            if not c.choices:
                errors.append(f"{prefix} MCQ case missing choices dict.")
            else:
                missing_keys = set("ABCD") - set(c.choices.keys())
                if missing_keys:
                    errors.append(
                        f"{prefix} MCQ choices missing keys: {sorted(missing_keys)}."
                    )
            if c.ground_truth not in ("A", "B", "C", "D"):
                errors.append(
                    f"{prefix} MCQ ground_truth must be A/B/C/D, "
                    f"got {c.ground_truth!r}."
                )

        # Facts consistency checks
        if c.facts:
            errors.extend(_validate_facts(prefix, c.facts))

        if c.tier == 1 and not c.profile_id.strip():
            errors.append(f"{prefix} Tier 1 case missing profile_id.")

        if c.confidence_tier == "A" and c.external_check_passed is not True:
            errors.append(
                f"{prefix} confidence_tier A requires external_check_passed=True."
            )

        if c.tier == 3 and c.confidence_tier != "C":
            errors.append(f"{prefix} Tier 3 case should use confidence_tier='C'.")

        # Strict checks
        if strict:
            if c.tier == 1 and not c.reasoning_steps:
                errors.append(f"{prefix} Tier 1 case missing reasoning_steps (--strict).")
            if c.tier == 2 and not c.statutory_refs:
                errors.append(f"{prefix} Tier 2 case missing statutory_refs (--strict).")
            if c.tier == 2 and not c.facts_narrative.strip():
                errors.append(f"{prefix} Tier 2 case missing facts_narrative (--strict).")
            if c.tier == 3 and not c.statutory_refs:
                errors.append(f"{prefix} Tier 3 case missing statutory_refs (--strict).")

    return errors


def print_stats(cases: list[BenchmarkCase]) -> None:
    """Print a concise distribution table."""
    tier_counts  = Counter(c.tier for c in cases)
    style_counts = Counter(c.style for c in cases)
    diff_counts  = Counter(c.difficulty for c in cases)
    domain_counts = Counter(c.domain for c in cases)
    yes_no = Counter(
        c.ground_truth for c in cases if c.style == "entailment"
    )

    print(f"Total cases : {len(cases)}")
    print()
    print("Tier    Count")
    for t in sorted(tier_counts):
        print(f"  {t}     {tier_counts[t]}")
    print()
    print("Style             Count")
    for s in sorted(style_counts):
        print(f"  {s:<18} {style_counts[s]}")
    print()
    print("Difficulty     Count")
    for d in ["basic", "intermediate", "advanced"]:
        print(f"  {d:<14} {diff_counts.get(d, 0)}")
    print()
    print("Domain                    Count")
    for dom in sorted(domain_counts):
        print(f"  {dom:<26} {domain_counts[dom]}")
    if yes_no:
        print()
        print(f"Entailment labels — Yes: {yes_no.get('Yes', 0)}, "
              f"No: {yes_no.get('No', 0)}")


def main() -> int:
    args = _args()

    if not args.dataset.exists():
        print(f"ERROR: File not found: {args.dataset}", file=sys.stderr)
        return 1

    print(f"Loading {args.dataset} ...")
    try:
        cases = load_dataset(args.dataset)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        print(f"ERROR: Failed to parse dataset: {exc}", file=sys.stderr)
        return 1

    print_stats(cases)
    print()

    errors = validate(cases, strict=args.strict)
    if errors:
        print(f"Validation FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        return 1

    mode = "(strict)" if args.strict else ""
    print(f"Validation PASSED {mode} — {len(cases)} cases, 0 errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
