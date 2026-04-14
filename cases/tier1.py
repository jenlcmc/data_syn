"""Build Tier 1 numeric benchmark cases.

Tier 1 cases present a structured taxpayer profile and ask the LLM to
compute a specific numeric tax value (AGI, taxable income, net liability,
SE tax, etc.).  Ground truth comes from the built-in deterministic Python
tax engine, making these the highest-confidence cases in the dataset.

Record sources (in order of preference):
  1. LCA-grounded records from ``sources/profile_builder.py`` — wages
     drawn from DOL H-1B filings and BLS OEWS; all benefits rule-derived.
  2. Hand-crafted seed records — round-number anchors for regression tests.

Each record generates multiple cases (one per quantity), so 25 + 25
records yield ~100–110 Tier 1 cases.

Case structure
--------------
  style        : "numeric"
  tier         : 1
  facts        : structured TaxpayerFacts
  question     : "What is the taxpayer's [quantity] for Tax Year 2024?"
  ground_truth : numeric USD value
  tolerance_strict_usd : 0
  tolerance_lenient_usd : 5
  reasoning_steps : full computation trace
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TAX_YEAR
from engine.ground_truth import TaxResult, compute
from sources.profiles import SEED_RECORDS, generate_lca_grounded_records
from schema import BenchmarkCase, TaxpayerFacts

# ------------------------------------------------------------------
# Target quantities produced per record
# ------------------------------------------------------------------
# Each entry is (field_on_TaxResult, question_template, difficulty, tags)
# The question template receives one positional format argument: TAX_YEAR.

_QUANTITY_SPECS: list[tuple[str, str, str, list[str]]] = [
    (
        "agi",
        "What is this taxpayer's adjusted gross income (AGI) for Tax Year {}?",
        "basic",
        ["agi"],
    ),
    (
        "taxable_income",
        "What is this taxpayer's federal taxable income for Tax Year {}?",
        "basic",
        ["taxable_income"],
    ),
    (
        "net_tax",
        "What is this taxpayer's net federal income tax liability for Tax Year {}?",
        "intermediate",
        ["net_tax", "federal_income_tax"],
    ),
    (
        "se_tax",
        "What is the self-employment (SE) tax owed by this taxpayer for Tax Year {}?",
        "intermediate",
        ["se_tax", "self_employment"],
    ),
    (
        "child_tax_credit",
        "What is the Child Tax Credit this taxpayer may claim for Tax Year {}?",
        "basic",
        ["child_tax_credit"],
    ),
    (
        "deduction_used",
        (
            "What is the total deduction (standard or itemized, whichever is greater) "
            "this taxpayer uses when computing taxable income for Tax Year {}?"
        ),
        "basic",
        ["standard_deduction", "itemized_deductions"],
    ),
]


def _narrative(facts: TaxpayerFacts) -> str:
    """Build a plain-English description of the taxpayer's financial situation."""
    lines = []

    status_labels = {
        "single": "single",
        "mfj":    "married filing jointly",
        "mfs":    "married filing separately",
        "hoh":    "head of household",
        "qss":    "qualifying surviving spouse",
    }
    lines.append(
        f"The taxpayer is filing as {status_labels.get(facts.filing_status, facts.filing_status)} "
        f"for Tax Year {TAX_YEAR}."
    )

    if facts.wages:
        lines.append(f"Wages and salaries: ${facts.wages:,}.")
    if facts.spouse_wages:
        lines.append(f"Spouse's wages: ${facts.spouse_wages:,}.")
    if facts.business_income:
        lines.append(
            f"Net self-employment (Schedule C) income: ${facts.business_income:,}."
        )
    if facts.taxable_interest:
        lines.append(f"Taxable interest income: ${facts.taxable_interest:,}.")
    if facts.ordinary_dividends:
        lines.append(
            f"Ordinary dividends: ${facts.ordinary_dividends:,} "
            f"(of which ${facts.qualified_dividends:,} are qualified dividends)."
        )
    if facts.capital_gains_net:
        lines.append(f"Net long-term capital gains: ${facts.capital_gains_net:,}.")
    if facts.pension_income:
        lines.append(f"Pension income: ${facts.pension_income:,}.")
    if facts.social_security_benefits:
        lines.append(f"Social Security benefits received: ${facts.social_security_benefits:,}.")
    if facts.rental_income_net:
        lines.append(f"Net rental income (Schedule E): ${facts.rental_income_net:,}.")
    if facts.ira_distributions:
        lines.append(f"IRA distributions: ${facts.ira_distributions:,}.")

    # Above-the-line deductions
    if facts.educator_expenses:
        lines.append(f"Educator expenses paid: ${facts.educator_expenses:,}.")
    if facts.student_loan_interest:
        lines.append(f"Student loan interest paid: ${facts.student_loan_interest:,}.")
    if facts.ira_contribution:
        if facts.covered_by_workplace_plan:
            lines.append(
                f"Traditional IRA contribution: ${facts.ira_contribution:,} "
                "(taxpayer is covered by a workplace retirement plan; "
                "deductibility may be phased out)."
            )
        else:
            lines.append(
                f"Traditional IRA contribution: ${facts.ira_contribution:,} "
                "(assume fully deductible — not covered by a workplace retirement plan)."
            )
    if facts.self_employed_health_ins:
        lines.append(
            f"Self-employed health insurance premiums paid: ${facts.self_employed_health_ins:,}."
        )
    if facts.sep_simple_ira_deduction:
        lines.append(
            f"SEP-IRA or SIMPLE IRA contribution deduction: ${facts.sep_simple_ira_deduction:,}."
        )

    # Itemized deduction inputs
    deduction_parts = []
    if facts.state_income_tax or facts.real_estate_tax:
        salt = facts.state_income_tax + facts.real_estate_tax
        deduction_parts.append(
            f"state and local taxes paid (SALT): ${salt:,} "
            f"(subject to the $10,000 cap)"
        )
    if facts.mortgage_interest:
        deduction_parts.append(f"mortgage interest: ${facts.mortgage_interest:,}")
    if facts.charitable_cash:
        deduction_parts.append(f"cash charitable contributions: ${facts.charitable_cash:,}")
    if facts.medical_expenses:
        deduction_parts.append(
            f"unreimbursed medical expenses: ${facts.medical_expenses:,} "
            f"(subject to 7.5% AGI floor)"
        )
    if deduction_parts:
        lines.append(
            "Potential itemized deductions include: " + "; ".join(deduction_parts) + "."
        )

    if facts.num_qualifying_children:
        n = facts.num_qualifying_children
        lines.append(
            f"The taxpayer has {n} qualifying child{'ren' if n > 1 else ''} "
            "for the Child Tax Credit."
        )

    return " ".join(lines)


def _domain(facts: TaxpayerFacts) -> str:
    """Determine the domain tag based on the taxpayer profile."""
    if facts.business_income > 0 and (facts.wages + facts.spouse_wages) > 0:
        return "both"
    if facts.business_income > 0:
        return "self_employment"
    return "federal_income_tax"


def _tags_from_facts(facts: TaxpayerFacts) -> list[str]:
    """Return a base set of topic tags derived from the facts profile."""
    tags = []
    if facts.wages:
        tags.append("wage_income")
    if facts.business_income:
        tags.append("self_employment")
        tags.append("se_tax")
    if facts.capital_gains_net or facts.qualified_dividends:
        tags.append("capital_gains")
    if facts.pension_income or facts.ira_distributions:
        tags.append("retirement_income")
    if facts.social_security_benefits:
        tags.append("social_security")
    if facts.rental_income_net:
        tags.append("rental_income")
    if facts.mortgage_interest:
        tags.append("itemized_deductions")
        tags.append("mortgage_interest")
    if facts.charitable_cash or facts.charitable_noncash:
        tags.append("charitable")
    if facts.num_qualifying_children:
        tags.append("child_tax_credit")
    if facts.ira_contribution:
        tags.append("ira")
    if facts.educator_expenses:
        tags.append("educator_expenses")
    if facts.student_loan_interest:
        tags.append("student_loan_interest")
    return tags


def build_tier1_cases() -> list[BenchmarkCase]:
    """Build all Tier 1 numeric cases.

    Uses two record sources:
      - LCA-grounded records (primary): 25 profiles derived from DOL LCA
        wage data and BLS OEWS percentiles.  Non-round, occupation-realistic
        wages and rule-derived benefits.
      - Hand-crafted seed records (secondary): 25 round-number profiles used
        as deterministic regression anchors.

    Each record produces one case per applicable tax quantity.

    Returns
    -------
    list[BenchmarkCase]
        Tier 1 numeric benchmark cases ordered by record source then quantity.
    """
    cases: list[BenchmarkCase] = []
    seq = 1

    # Primary source: LCA-grounded profiles
    # Use a larger sample so scenario families and wage bands are represented
    # more evenly across the generated set.
    lca_records = generate_lca_grounded_records(n=60, random_seed=42)
    all_record_sources = [
        (facts, "lca_oews") for facts in lca_records
    ] + [
        (TaxpayerFacts(**raw), "hand_crafted") for raw in SEED_RECORDS
    ]

    for profile_seq, (facts, source_label) in enumerate(all_record_sources, start=1):
        result: TaxResult = compute(facts)
        profile_id = f"t1p_{source_label}_{profile_seq:04d}"

        narrative = _narrative(facts)
        domain = _domain(facts)
        base_tags = _tags_from_facts(facts)

        for field_name, question_tmpl, difficulty, quantity_tags in _QUANTITY_SPECS:
            value: float = getattr(result, field_name)

            # Skip quantities that are zero and not the primary focus of the record
            if value == 0.0 and field_name not in ("agi", "taxable_income", "net_tax"):
                continue

            # Skip SE tax for records with no self-employment income
            if field_name == "se_tax" and facts.business_income == 0:
                continue

            # Skip CTC if no qualifying children
            if field_name == "child_tax_credit" and facts.num_qualifying_children == 0:
                continue

            case_id = f"t1_numeric_{seq:04d}"
            question = question_tmpl.format(TAX_YEAR)

            # Enrich difficulty for records with complex profiles
            effective_difficulty = difficulty
            has_complexity = (
                facts.capital_gains_net > 0
                or facts.social_security_benefits > 0
                or result.needs_amt_review
            )
            if has_complexity and difficulty == "basic":
                effective_difficulty = "intermediate"
            if result.needs_amt_review:
                effective_difficulty = "advanced"

            case = BenchmarkCase(
                id=case_id,
                tier=1,
                domain=domain,
                style="numeric",
                difficulty=effective_difficulty,
                tax_year=TAX_YEAR,
                source=source_label,
                verified_by="python_engine",
                profile_id=profile_id,
                facts=facts,
                facts_narrative=narrative,
                question=question,
                choices=None,
                ground_truth=round(value, 2),
                ground_truth_type="numeric_usd",
                tolerance_strict_usd=0.0,
                tolerance_lenient_usd=5.0,
                tolerance_pct=0.10,
                reasoning_steps=list(result.reasoning_steps),
                statutory_refs=_statutory_refs_for(field_name),
                explanation=(
                    f"Computed by the 2024 Python tax engine. "
                    f"{'Flagged for external validation (AMT possible). ' if result.needs_amt_review else ''}"
                    f"{'Flagged: QBI W-2 wage limitations not computed. ' if result.needs_external_validation else ''}"
                ),
                tags=list(set(base_tags + quantity_tags)),
            )
            cases.append(case)
            seq += 1

    return cases


def _statutory_refs_for(field_name: str) -> list[str]:
    """Return the primary IRC sections governing a given computed quantity."""
    refs: dict[str, list[str]] = {
        "agi":               ["26 USC §62"],
        "taxable_income":    ["26 USC §63"],
        "net_tax":           ["26 USC §1", "26 USC §63"],
        "se_tax":            ["26 USC §1401", "26 USC §164(f)"],
        "child_tax_credit":  ["26 USC §24"],
        "deduction_used":    ["26 USC §63", "26 USC §164", "26 USC §170"],
    }
    return refs.get(field_name, ["26 USC §1"])
