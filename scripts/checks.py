"""Sanity checks and property-based invariants for the tax engine.

Two check suites are combined here:

1. Targeted checks — high-risk components that are easy to regress:
   NIIT (3.8%), Additional Medicare Tax (0.9%), IRA phase-out.

2. Invariant checks — properties that must hold across all generated records:
   - taxable_income <= AGI
   - increasing wages does not decrease regular_tax
   - QBI deduction stays within 20% statutory caps

Usage
-----
    python data_syn/scripts/checks.py
    python data_syn/scripts/checks.py --n-lca 200 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.ground_truth import compute
from sources.profiles import SEED_RECORDS, generate_lca_grounded_records
from schema import TaxpayerFacts


# ---------------------------------------------------------------------------
# Targeted checks
# ---------------------------------------------------------------------------

def _assert_close(label: str, actual: float, expected: float, tol: float = 0.01) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{label}: expected {expected:.2f}, got {actual:.2f}"
        )


def _check_niit() -> None:
    facts = TaxpayerFacts(
        filing_status="single",
        wages=180_000,
        capital_gains_net=30_000,
        ordinary_dividends=30_000,
        qualified_dividends=30_000,
    )
    result = compute(facts)
    # AGI ~240k; NII = 60k; excess over 200k threshold = 40k -> NIIT base 40k.
    _assert_close("NIIT", result.niit, 40_000 * 0.038)


def _check_addl_medicare_w2() -> None:
    facts = TaxpayerFacts(filing_status="single", wages=260_000)
    result = compute(facts)
    # Additional Medicare base = 260k - 200k = 60k.
    _assert_close("Additional Medicare Tax (W-2)", result.addl_medicare_tax, 60_000 * 0.009)


def _check_addl_medicare_se() -> None:
    facts = TaxpayerFacts(filing_status="single", wages=200_000, business_income=80_000)
    result = compute(facts)
    # SE earnings base for Medicare = 80,000 * 0.9235 = 73,880.
    # Combined earned = 200,000 + 73,880 = 273,880.
    expected = (273_880 * 0.009) - (200_000 * 0.009)
    _assert_close("Additional Medicare Tax (SE)", result.addl_medicare_tax, expected)


def _check_ira_phaseout() -> None:
    facts = TaxpayerFacts(
        filing_status="single",
        wages=84_000,
        ira_contribution=7_000,
        covered_by_workplace_plan=True,
    )
    result = compute(facts)
    # Single covered phaseout 77k..87k. At AGI 84k, deductible fraction = 3/10.
    _assert_close("IRA deduction phase-out", result.ira_deduction, 7_000 * 0.30)


def run_targeted_checks() -> list[str]:
    """Return a list of failure messages; empty means all pass."""
    checks = [
        ("niit", _check_niit),
        ("addl_medicare_w2", _check_addl_medicare_w2),
        ("addl_medicare_se", _check_addl_medicare_se),
        ("ira_phaseout", _check_ira_phaseout),
    ]
    failures = []
    for name, fn in checks:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"targeted/{name}: {exc}")
    return failures


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def _qbi_pre_qbi_taxable(result) -> float:
    return max(0.0, result.agi - result.deduction_used)


def run_invariant_checks(
    n_lca: int = 120,
    seed: int = 42,
    wage_bump: int = 1_000,
) -> list[str]:
    """Return a list of invariant failures; empty means pass."""
    failures: list[str] = []

    lca_records = generate_lca_grounded_records(n=n_lca, random_seed=seed)
    seed_records = [TaxpayerFacts(**raw) for raw in SEED_RECORDS]
    records = lca_records + seed_records

    for idx, facts in enumerate(records, start=1):
        base = compute(facts)

        if base.taxable_income > base.agi + 0.01:
            failures.append(
                f"record[{idx}] taxable_income > AGI "
                f"({base.taxable_income:.2f} > {base.agi:.2f})"
            )

        bumped = replace(facts, wages=facts.wages + wage_bump)
        if compute(bumped).regular_tax + 0.01 < base.regular_tax:
            failures.append(
                f"record[{idx}] regular_tax decreased after wage bump "
                f"({base.regular_tax:.2f} -> {compute(bumped).regular_tax:.2f})"
            )

        if facts.business_income > 0:
            qbi_cap = 0.20 * float(facts.business_income)
            pre_qbi_cap = 0.20 * _qbi_pre_qbi_taxable(base)
            upper = max(0.0, min(qbi_cap, pre_qbi_cap))
            if base.qbi_deduction > upper + 0.01:
                failures.append(
                    f"record[{idx}] qbi_deduction exceeds cap "
                    f"({base.qbi_deduction:.2f} > {upper:.2f})"
                )

    return failures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tax engine checks and invariants.")
    parser.add_argument("--n-lca", type=int, default=120,
                        help="LCA profiles for invariant checks (default: 120).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wage-bump", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = _args()

    targeted = run_targeted_checks()
    invariants = run_invariant_checks(n_lca=args.n_lca, seed=args.seed, wage_bump=args.wage_bump)

    all_failures = targeted + invariants
    total_records = args.n_lca + len(SEED_RECORDS)

    print(f"Targeted checks : {'PASS' if not targeted else f'FAIL ({len(targeted)})'}")
    print(f"Invariant checks: records={total_records}, "
          f"{'PASS' if not invariants else f'FAIL ({len(invariants)})'}")

    if all_failures:
        print("\nFailures:")
        for f in all_failures[:30]:
            print(f"  - {f}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
