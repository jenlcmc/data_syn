"""Preview benchmark cases in a human-readable format.

Useful for spot-checking ground truth, reading mined IRS scenarios,
and verifying reasoning steps before a full evaluation run.

Usage
-----
    # Show 5 random cases
    python data_syn/scripts/preview.py data_syn/output/benchmark.json

    # Show all Tier 1 cases
    python data_syn/scripts/preview.py data_syn/output/benchmark.json --tier 1

    # Show MCQ cases tagged 'capital_gains'
    python data_syn/scripts/preview.py data_syn/output/benchmark.json --style mcq --tag capital_gains

    # Show a specific case by ID
    python data_syn/scripts/preview.py data_syn/output/benchmark.json --id t1_numeric_0003

    # Show N cases and include full reasoning steps
    python data_syn/scripts/preview.py data_syn/output/benchmark.json --n 10 --reasoning
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import BenchmarkCase, load_dataset


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview benchmark cases from a JSON dataset."
    )
    parser.add_argument("dataset", type=Path, help="Path to benchmark.json.")
    parser.add_argument("--tier",  type=int, choices=[1, 2, 3],
                        help="Filter by tier.")
    parser.add_argument("--style",
                        choices=["numeric", "entailment", "qa", "mcq", "scenario"],
                        help="Filter by style.")
    parser.add_argument("--tag", type=str,
                        help="Filter to cases containing this tag.")
    parser.add_argument("--difficulty",
                        choices=["basic", "intermediate", "advanced"],
                        help="Filter by difficulty.")
    parser.add_argument("--domain",
                        choices=["federal_income_tax", "self_employment", "both"],
                        help="Filter by domain.")
    parser.add_argument("--id", type=str, dest="case_id",
                        help="Show a specific case by ID.")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of cases to preview (default: 5, 0 = all).")
    parser.add_argument("--reasoning", action="store_true",
                        help="Show full reasoning steps.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling (default: 0).")
    return parser.parse_args()


def _divider(title: str = "", width: int = 70) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return "-" * pad + f" {title} " + "-" * pad
    return "-" * width


def _fmt_facts(facts_dict: dict) -> str:
    """Format a TaxpayerFacts dict as indented key: value lines."""
    if not facts_dict:
        return "  (no structured facts)"
    lines = []
    for k, v in facts_dict.items():
        if v and v != 0 and v != [] and v != {} and v is not None:
            if isinstance(v, int) and k not in (
                "age_primary", "age_spouse", "num_qualifying_children"
            ):
                lines.append(f"  {k}: ${v:,}")
            else:
                lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "  (all zero / default values)"


def preview_case(case: BenchmarkCase, show_reasoning: bool = False) -> str:
    """Format one case as a readable string block."""
    lines = [
        _divider(f"{case.id} | {case.style} | Tier {case.tier} | {case.difficulty}"),
        f"Domain     : {case.domain}",
        f"Tax year   : {case.tax_year}",
        f"Source     : {case.source}  (verified_by={case.verified_by})",
        f"Tags       : {', '.join(case.tags) if case.tags else 'none'}",
        f"Statutory  : {', '.join(case.statutory_refs) if case.statutory_refs else 'none'}",
        "",
    ]

    if case.facts_narrative:
        lines.append("Scenario:")
        for line in case.facts_narrative.split(". "):
            if line.strip():
                lines.append(f"  {line.strip()}.")
        lines.append("")
    elif case.facts:
        lines.append("Taxpayer facts:")
        facts_dict = {}
        if hasattr(case.facts, "__dict__"):
            facts_dict = vars(case.facts)
        elif isinstance(case.facts, dict):
            facts_dict = case.facts
        lines.append(_fmt_facts(facts_dict))
        lines.append("")

    lines.append("Question:")
    # Wrap long questions
    q = case.question
    while len(q) > 72:
        split_at = q[:72].rfind(" ")
        if split_at == -1:
            split_at = 72
        lines.append(f"  {q[:split_at]}")
        q = q[split_at:].strip()
    lines.append(f"  {q}")
    lines.append("")

    if case.choices:
        lines.append("Choices:")
        for key in sorted(case.choices):
            lines.append(f"  {key}) {case.choices[key]}")
        lines.append("")

    gt = case.ground_truth
    if isinstance(gt, float):
        gt_display = f"${gt:,.2f}"
    else:
        gt_display = str(gt)

    lines.append(f"Ground truth ({case.ground_truth_type}): {gt_display}")

    if case.tolerance_lenient_usd > 0:
        lines.append(f"Tolerances  : strict=±$0, lenient=±${case.tolerance_lenient_usd:.0f}, pct=±{case.tolerance_pct:.0%}")

    if case.explanation:
        lines.append(f"Explanation : {case.explanation[:200]}")

    if show_reasoning and case.reasoning_steps:
        lines.append("")
        lines.append("Reasoning steps:")
        for i, step in enumerate(case.reasoning_steps, 1):
            lines.append(f"  {i}. {step}")

    return "\n".join(lines)


def main() -> int:
    args = _args()

    if not args.dataset.exists():
        print(f"ERROR: File not found: {args.dataset}", file=sys.stderr)
        return 1

    cases = load_dataset(args.dataset)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------
    if args.case_id:
        cases = [c for c in cases if c.id == args.case_id]
        if not cases:
            print(f"No case found with id={args.case_id!r}.", file=sys.stderr)
            return 1

    if args.tier is not None:
        cases = [c for c in cases if c.tier == args.tier]
    if args.style:
        cases = [c for c in cases if c.style == args.style]
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.difficulty:
        cases = [c for c in cases if c.difficulty == args.difficulty]
    if args.domain:
        cases = [c for c in cases if c.domain == args.domain]

    if not cases:
        print("No cases match the given filters.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    if args.n > 0 and len(cases) > args.n:
        random.seed(args.seed)
        cases = random.sample(cases, args.n)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print(f"Previewing {len(cases)} case(s) from {args.dataset.name}:\n")
    for case in cases:
        print(preview_case(case, show_reasoning=args.reasoning))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
