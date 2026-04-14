"""Build Tier 2 statutory entailment cases.

Tier 2 cases present a taxpayer situation and ask a Yes/No question about
whether a specific IRC rule applies.  Ground truth is derived by evaluating
the rule against the taxpayer's computed profile — no LLM is involved.

Question format
---------------
  "Under [IRC section], does [rule condition] apply to this taxpayer?"
  Expected answer: "Yes" or "No"

These are modeled after the SARA (Statutory Article Reasoning Assessment)
entailment format used in LegalBench.

All cases are hand-crafted to ensure the statute text, taxpayer facts, and
correct label align precisely.  A human tax professional should review a
representative sample (recommended: 20% of the set).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ADDL_MEDICARE_THRESHOLD,
    CTC_PHASEOUT_THRESHOLD,
    IRA_CONTRIBUTION_LIMIT,
    LTCG_0_THRESHOLD,
    QBI_THRESHOLD,
    SALT_CAP,
    STANDARD_DEDUCTION,
    TAX_YEAR,
)
from engine.ground_truth import compute
from schema import BenchmarkCase, TaxpayerFacts


@dataclass
class EntailmentSpec:
    """Specification for one Tier 2 entailment case."""

    facts: TaxpayerFacts
    question: str
    statute: str
    label: Callable[[TaxpayerFacts], bool]   # returns True if answer is "Yes"
    explanation: Callable[[TaxpayerFacts], str]
    difficulty: str
    tags: list[str]
    statutory_refs: list[str]


# ---------------------------------------------------------------------------
# Entailment specifications
# Each spec defines a taxpayer situation and a Yes/No rule question.
# The label function is evaluated at build time to produce the ground truth.
# ---------------------------------------------------------------------------

def _build_specs() -> list[EntailmentSpec]:
    specs: list[EntailmentSpec] = []

    # ------------------------------------------------------------------
    # 1. Standard deduction vs. itemizing
    # ------------------------------------------------------------------
    for fs, wages, salt, mortgage, charitable, std_label in [
        ("single", 75_000, 5_500, 0, 1_200, "single"),
        ("single", 95_000, 9_000, 18_000, 4_000, "single"),   # itemized wins
        ("mfj",    150_000, 12_000, 22_000, 6_000, "mfj"),    # itemized wins
        ("mfj",    80_000, 5_000, 0, 2_000, "mfj"),           # std wins
        ("hoh",    65_000, 7_500, 12_000, 3_000, "hoh"),      # itemized wins
    ]:
        facts = TaxpayerFacts(
            filing_status=fs,
            wages=wages,
            state_income_tax=min(salt, 10_000),
            mortgage_interest=mortgage,
            charitable_cash=charitable,
        )
        itemized = min(salt, SALT_CAP) + mortgage + charitable
        std = STANDARD_DEDUCTION[fs]

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §63, does this taxpayer benefit from itemizing "
                f"deductions rather than taking the standard deduction for "
                f"Tax Year {TAX_YEAR}?"
            ),
            statute=(
                "IRC §63 provides that 'taxable income' means gross income "
                "minus deductions.  A taxpayer may elect to itemize deductions "
                "under §63(e) only if the total of allowable itemized deductions "
                "exceeds the standard deduction."
            ),
            label=lambda f, i=itemized, s=std: i > s,
            explanation=lambda f, i=itemized, s=std, fs=fs: (
                f"Itemized deductions total ${i:,} "
                f"(SALT capped at ${SALT_CAP:,}, mortgage interest, charitable). "
                f"Standard deduction for {fs} = ${s:,}. "
                f"Itemizing {'is' if i > s else 'is not'} beneficial."
            ),
            difficulty="basic",
            tags=["standard_deduction", "itemized_deductions"],
            statutory_refs=["26 USC §63"],
        ))

    # ------------------------------------------------------------------
    # 2. SALT cap applicability
    # ------------------------------------------------------------------
    for fs, wages, salt_paid in [
        ("single", 80_000, 7_500),    # below cap — cap does not bind
        ("single", 120_000, 14_000),  # above cap — cap binds
        ("mfj", 180_000, 22_000),     # above cap — cap binds
        ("mfj", 140_000, 9_000),      # below cap — cap does not bind
    ]:
        facts = TaxpayerFacts(filing_status=fs, wages=wages, state_income_tax=salt_paid)
        cap_binds = salt_paid > SALT_CAP

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §164(b)(6), does the $10,000 SALT deduction cap "
                f"limit this taxpayer's state and local tax deduction for "
                f"Tax Year {TAX_YEAR}?"
            ),
            statute=(
                "IRC §164(b)(6) limits the deduction for state and local taxes "
                "paid (income, sales, and property taxes combined) to $10,000 "
                "($5,000 for married filing separately) per taxable year."
            ),
            label=lambda f, c=cap_binds: c,
            explanation=lambda f, s=salt_paid, c=cap_binds: (
                f"SALT paid = ${s:,}. Cap = ${SALT_CAP:,}. "
                f"The cap {'does' if c else 'does not'} reduce the deduction."
            ),
            difficulty="basic",
            tags=["salt_cap", "itemized_deductions"],
            statutory_refs=["26 USC §164(b)(6)"],
        ))

    # ------------------------------------------------------------------
    # 3. QBI deduction eligibility (income below threshold)
    # ------------------------------------------------------------------
    for fs, wages, se_income, threshold_label in [
        ("single", 0, 85_000, "single"),     # below threshold — eligible
        ("single", 0, 220_000, "single"),    # above threshold — W-2 limit applies
        ("mfj", 60_000, 110_000, "mfj"),     # below threshold — eligible
        ("mfj", 100_000, 320_000, "mfj"),    # above threshold — limitation applies
    ]:
        facts = TaxpayerFacts(filing_status=fs, wages=wages, business_income=se_income)
        result = compute(facts)
        threshold = QBI_THRESHOLD[fs]
        below_threshold = result.agi <= threshold

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §199A, is this self-employed taxpayer eligible "
                f"to claim the full 20% QBI deduction without the W-2 wage "
                f"limitation for Tax Year {TAX_YEAR}?"
            ),
            statute=(
                "IRC §199A(b)(2) allows a deduction equal to 20% of qualified "
                "business income.  For taxpayers with taxable income above the "
                "threshold ($191,950 single / $383,900 MFJ for 2024), the "
                "deduction is limited to the greater of: (A) 50% of W-2 wages, "
                "or (B) 25% of W-2 wages plus 2.5% of the unadjusted basis of "
                "qualified property.  Below the threshold, no W-2 wage limitation "
                "applies."
            ),
            label=lambda f, b=below_threshold: b,
            explanation=lambda f, agi=result.agi, thr=threshold, b=below_threshold: (
                f"AGI = ${agi:,.2f}. §199A threshold = ${thr:,}. "
                f"The W-2 wage limitation {'does not apply' if b else 'applies'} — "
                f"taxpayer {'qualifies for' if b else 'does not qualify for'} "
                f"the full 20% QBI deduction."
            ),
            difficulty="intermediate",
            tags=["qbi", "self_employment"],
            statutory_refs=["26 USC §199A"],
        ))

    # ------------------------------------------------------------------
    # 4. Self-employment tax obligation
    # ------------------------------------------------------------------
    for fs, wages, se_income in [
        ("single", 0, 48_000),   # SE income → owes SE tax
        ("single", 60_000, 0),   # no SE income → no SE tax
        ("mfj", 50_000, 35_000), # both W-2 and SE → owes SE tax
    ]:
        facts = TaxpayerFacts(filing_status=fs, wages=wages, business_income=se_income)
        owes_se = se_income > 400   # IRC §1402(b): $400 threshold

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §1401, is this taxpayer required to pay "
                f"self-employment tax for Tax Year {TAX_YEAR}?"
            ),
            statute=(
                "IRC §1401 imposes a tax on self-employment income at a rate "
                "of 15.3% (12.4% SS + 2.9% Medicare) on net earnings from "
                "self-employment.  IRC §1402(b) defines 'net earnings from "
                "self-employment' as net profit from a trade or business.  "
                "Taxpayers with net self-employment income below $400 are not "
                "subject to SE tax."
            ),
            label=lambda f, o=owes_se: o,
            explanation=lambda f, s=se_income, o=owes_se: (
                f"Net self-employment income = ${s:,}. "
                f"{'Exceeds' if o else 'Does not exceed'} the $400 threshold "
                f"(IRC §1402(b)). SE tax {'applies' if o else 'does not apply'}."
            ),
            difficulty="basic",
            tags=["se_tax", "self_employment"],
            statutory_refs=["26 USC §1401", "26 USC §1402"],
        ))

    # ------------------------------------------------------------------
    # 5. Traditional IRA deductibility
    # ------------------------------------------------------------------
    for fs, wages, ira_contrib, covered_by_plan, description in [
        ("single", 45_000, 7_000, False,
         "Not covered by workplace plan — full deduction"),
        ("single", 82_000, 7_000, True,
         "Covered by plan, AGI above phase-out range — no deduction"),
        ("single", 55_000, 7_000, True,
         "Covered by plan, AGI within phase-out range — partial deduction"),
        ("mfj", 100_000, 7_000, False,
         "Neither spouse covered — full deduction"),
    ]:
        facts = TaxpayerFacts(
            filing_status=fs,
            wages=wages,
            ira_contribution=ira_contrib,
            covered_by_workplace_plan=covered_by_plan,
        )
        result = compute(facts)
        agi = result.agi

        if not covered_by_plan:
            deductible = True
        elif fs == "single":
            deductible = agi < 87_000  # 2024 single, covered-by-plan phase-out end
        else:
            deductible = agi < 143_000  # 2024 MFJ phase-out end

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §219, can this taxpayer deduct their Traditional IRA "
                f"contribution for Tax Year {TAX_YEAR}? ({description})"
            ),
            statute=(
                "IRC §219(a) allows a deduction for contributions to an "
                "individual retirement account.  IRC §219(g) phases out the "
                "deduction for taxpayers covered by a workplace retirement plan "
                "with AGI above $77,000 (single, 2024) and $123,000 (MFJ, 2024), "
                "eliminating it at $87,000 and $143,000 respectively."
            ),
            label=lambda f, d=deductible: d,
            explanation=lambda f, agi=agi, d=deductible, cp=covered_by_plan: (
                f"AGI = ${agi:,.2f}. "
                f"{'Not covered by workplace plan — full deduction allowed.' if not cp else ''}"
                f"{'Covered by plan; ' if cp else ''}"
                f"{'IRA deduction is' if d else 'IRA deduction is not'} available."
            ),
            difficulty="intermediate",
            tags=["ira", "above_the_line_deduction"],
            statutory_refs=["26 USC §219"],
        ))

    # ------------------------------------------------------------------
    # 6. 0% long-term capital gains rate
    # ------------------------------------------------------------------
    for fs, wages, ltcg in [
        ("single", 28_000, 12_000),   # stacked income below 0% threshold
        ("single", 50_000, 10_000),   # stacked income above 0% threshold
        ("mfj", 60_000, 30_000),      # MFJ, within 0% bracket
        ("mfj", 90_000, 25_000),      # MFJ, above 0% bracket
    ]:
        facts = TaxpayerFacts(
            filing_status=fs,
            wages=wages,
            capital_gains_net=ltcg,
            qualified_dividends=ltcg,
            ordinary_dividends=ltcg,
        )
        result = compute(facts)
        ordinary_ti = result.ordinary_taxable_income
        threshold_0 = LTCG_0_THRESHOLD[fs]
        qualifies = ordinary_ti < threshold_0

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §1(h), does any portion of this taxpayer's "
                f"long-term capital gains qualify for the 0% preferential "
                f"tax rate for Tax Year {TAX_YEAR}?"
            ),
            statute=(
                "IRC §1(h) taxes net long-term capital gains at 0% to the "
                "extent the taxpayer's taxable income (including LTCG) does "
                "not exceed the applicable threshold ($47,025 single, "
                "$94,050 MFJ for 2024)."
            ),
            label=lambda f, q=qualifies: q,
            explanation=lambda f, ti=ordinary_ti, thr=threshold_0, q=qualifies: (
                f"Ordinary taxable income = ${ti:,.2f}. "
                f"0% LTCG threshold = ${thr:,}. "
                f"{'Some LTCG qualifies' if q else 'No LTCG qualifies'} for 0% rate."
            ),
            difficulty="intermediate",
            tags=["capital_gains", "ltcg_rate"],
            statutory_refs=["26 USC §1(h)"],
        ))

    # ------------------------------------------------------------------
    # 7. Net Investment Income Tax (NIIT)
    # ------------------------------------------------------------------
    for fs, wages, nii, description in [
        ("single", 180_000, 30_000, "High wages + investment income"),
        ("single", 60_000, 10_000, "Low wages, NII below threshold"),
        ("mfj", 240_000, 20_000, "MFJ above $250k threshold"),
        ("mfj", 200_000, 15_000, "MFJ below $250k threshold"),
    ]:
        facts = TaxpayerFacts(
            filing_status=fs,
            wages=wages,
            capital_gains_net=nii,
            ordinary_dividends=nii,
            qualified_dividends=nii,
        )
        result = compute(facts)
        threshold = ADDL_MEDICARE_THRESHOLD[fs]
        owes_niit = result.agi > threshold and nii > 0

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §1411, is this taxpayer subject to the 3.8% "
                f"Net Investment Income Tax (NIIT) for Tax Year {TAX_YEAR}? "
                f"({description})"
            ),
            statute=(
                "IRC §1411 imposes a 3.8% tax on the lesser of (1) net "
                "investment income or (2) the excess of modified AGI over "
                "$200,000 (single) or $250,000 (MFJ)."
            ),
            label=lambda f, o=owes_niit: o,
            explanation=lambda f, agi=result.agi, thr=threshold, o=owes_niit: (
                f"AGI = ${agi:,.2f}. NIIT threshold = ${thr:,}. "
                f"NIIT {'applies' if o else 'does not apply'}."
            ),
            difficulty="intermediate",
            tags=["niit", "net_investment_income"],
            statutory_refs=["26 USC §1411"],
        ))

    # ------------------------------------------------------------------
    # 8. Child Tax Credit phase-out
    # ------------------------------------------------------------------
    for fs, wages, n_children, description in [
        ("single", 175_000, 2, "Below $200k threshold — full CTC"),
        ("single", 220_000, 2, "Above $200k threshold — reduced CTC"),
        ("mfj", 380_000, 3, "Below $400k threshold — full CTC"),
        ("mfj", 420_000, 2, "Above $400k threshold — reduced CTC"),
    ]:
        facts = TaxpayerFacts(
            filing_status=fs,
            wages=wages,
            num_qualifying_children=n_children,
        )
        result = compute(facts)
        full_ctc = n_children * 2_000
        ctc_reduced = result.child_tax_credit < full_ctc and result.child_tax_credit >= 0
        ctc_phased_out = result.child_tax_credit < full_ctc

        specs.append(EntailmentSpec(
            facts=facts,
            question=(
                f"Under IRC §24(b), is this taxpayer's Child Tax Credit reduced "
                f"by the income phase-out for Tax Year {TAX_YEAR}? ({description})"
            ),
            statute=(
                "IRC §24(b) reduces the Child Tax Credit by $50 for each "
                "$1,000 (or fraction thereof) of modified AGI above $200,000 "
                "(single) or $400,000 (MFJ)."
            ),
            label=lambda f, r=ctc_phased_out: r,
            explanation=lambda f, agi=result.agi, ctc=result.child_tax_credit,
                          full=full_ctc, r=ctc_phased_out: (
                f"AGI = ${agi:,.2f}. Full CTC = ${full:,}. "
                f"Actual CTC = ${ctc:,.2f}. "
                f"Phase-out {'reduces' if r else 'does not reduce'} the CTC."
            ),
            difficulty="intermediate",
            tags=["child_tax_credit", "phase_out"],
            statutory_refs=["26 USC §24"],
        ))

    return specs


def _facts_narrative(facts: TaxpayerFacts) -> str:
    """Build a compact taxpayer narrative for entailment prompts."""
    status_labels = {
        "single": "single",
        "mfj": "married filing jointly",
        "mfs": "married filing separately",
        "hoh": "head of household",
        "qss": "qualifying surviving spouse",
    }

    lines = [
        f"The taxpayer is filing as {status_labels.get(facts.filing_status, facts.filing_status)} "
        f"for Tax Year {TAX_YEAR}."
    ]
    if facts.wages:
        lines.append(f"Wages: ${facts.wages:,}.")
    if facts.spouse_wages:
        lines.append(f"Spouse wages: ${facts.spouse_wages:,}.")
    if facts.business_income:
        lines.append(f"Net self-employment income: ${facts.business_income:,}.")
    if facts.capital_gains_net:
        lines.append(f"Net long-term capital gains: ${facts.capital_gains_net:,}.")
    if facts.ordinary_dividends:
        lines.append(f"Ordinary dividends: ${facts.ordinary_dividends:,}.")
    if facts.state_income_tax:
        lines.append(f"State income taxes paid: ${facts.state_income_tax:,}.")
    if facts.real_estate_tax:
        lines.append(f"Real estate taxes paid: ${facts.real_estate_tax:,}.")
    if facts.mortgage_interest:
        lines.append(f"Mortgage interest paid: ${facts.mortgage_interest:,}.")
    if facts.charitable_cash:
        lines.append(f"Cash charitable contributions: ${facts.charitable_cash:,}.")
    if facts.ira_contribution:
        lines.append(
            f"Traditional IRA contribution: ${facts.ira_contribution:,}. "
            f"Covered by workplace plan: {'Yes' if facts.covered_by_workplace_plan else 'No'}."
        )
    if facts.num_qualifying_children:
        lines.append(f"Number of qualifying children: {facts.num_qualifying_children}.")
    return " ".join(lines)


def _domain_from_facts(facts: TaxpayerFacts) -> str:
    """Map taxpayer facts to benchmark domain label."""
    has_w2 = (facts.wages + facts.spouse_wages) > 0
    has_se = facts.business_income > 0
    if has_w2 and has_se:
        return "both"
    if has_se:
        return "self_employment"
    return "federal_income_tax"


def build_tier2_cases() -> list[BenchmarkCase]:
    """Build all Tier 2 statutory entailment cases.

    Returns
    -------
    list[BenchmarkCase]
        Entailment cases with "Yes" or "No" ground truth, ordered by IRC
        section coverage.
    """
    specs = _build_specs()
    cases: list[BenchmarkCase] = []

    for seq, spec in enumerate(specs, start=1):
        label = spec.label(spec.facts)
        explanation = spec.explanation(spec.facts)

        case = BenchmarkCase(
            id=f"t2_entail_{seq:04d}",
            tier=2,
            domain=_domain_from_facts(spec.facts),
            style="entailment",
            difficulty=spec.difficulty,
            tax_year=TAX_YEAR,
            source="hand_crafted",
            verified_by="python_engine",
            profile_id=f"t2p_{seq:04d}",
            facts=spec.facts,
            facts_narrative=_facts_narrative(spec.facts),
            question=spec.question,
            choices=None,
            ground_truth="Yes" if label else "No",
            ground_truth_type="boolean_str",
            tolerance_strict_usd=0.0,
            tolerance_lenient_usd=0.0,
            tolerance_pct=0.0,
            reasoning_steps=[spec.statute, explanation],
            statutory_refs=spec.statutory_refs,
            explanation=explanation,
            tags=spec.tags,
        )
        cases.append(case)

    return cases
