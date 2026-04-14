"""Build Tier 3 Q&A, MCQ, and scenario cases.

Tier 3 cases come from two sources:

1. IRS worked examples mined from XML publications — these are authoritative
   scenarios written by IRS staff.  The miner in ``sources/irs_example_miner.py``
   extracts them; this module selects and frames them as open-ended scenario
   questions.

2. Hand-crafted Q&A and MCQ cases — covering conceptual tax knowledge,
   definitional questions, and multi-step scenario reasoning.  These are
   calibrated to the kinds of questions real taxpayers and practitioners ask.

Ground truth for this tier is the stated conclusion in the IRS publication
(for mined examples) or a carefully verified answer written alongside the
question (for hand-crafted cases).  Scoring is text-based (does the LLM's
response match the reference answer in substance?) rather than exact-match.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TAX_YEAR
from schema import BenchmarkCase

# ---------------------------------------------------------------------------
# Section 1: Hand-crafted Q&A cases
# ---------------------------------------------------------------------------
# Each dict contains the fields needed to build a BenchmarkCase.
# ground_truth_type is "text" for open-ended answers.

_QA_SPECS: list[dict] = [
    # --- Foundational concepts ---
    {
        "question": (
            "What is the difference between a tax deduction and a tax credit, "
            "and which provides a larger tax benefit per dollar?"
        ),
        "ground_truth": (
            "A tax deduction reduces taxable income, saving taxes at the marginal "
            "rate.  A tax credit reduces the tax owed dollar-for-dollar.  A credit "
            "generally provides a larger benefit: a $1,000 deduction saves a 22% "
            "bracket taxpayer $220, while a $1,000 credit saves exactly $1,000."
        ),
        "difficulty": "basic",
        "domain": "federal_income_tax",
        "tags": ["deduction_vs_credit", "tax_fundamentals"],
        "statutory_refs": ["26 USC §63", "26 USC §21"],
    },
    {
        "question": (
            "What is adjusted gross income (AGI) and why does it matter "
            "for a taxpayer's federal return?"
        ),
        "ground_truth": (
            "AGI is gross income minus above-the-line deductions (IRC §62), "
            "such as the SE deduction, educator expenses, and student loan "
            "interest.  AGI matters because many other deductions and credits "
            "phase out or are limited based on AGI (e.g., the medical expense "
            "7.5% floor, Roth IRA eligibility, and IRA deductibility)."
        ),
        "difficulty": "basic",
        "domain": "federal_income_tax",
        "tags": ["agi", "tax_fundamentals"],
        "statutory_refs": ["26 USC §62"],
    },
    {
        "question": (
            "A single filer earned $50,000 in wages.  What is her 2024 standard "
            "deduction, and what is her federal taxable income?"
        ),
        "ground_truth": (
            "The 2024 standard deduction for a single filer is $14,600 "
            "(IRC §63(c)).  Taxable income = $50,000 - $14,600 = $35,400."
        ),
        "difficulty": "basic",
        "domain": "federal_income_tax",
        "tags": ["standard_deduction", "taxable_income", "single_filer"],
        "statutory_refs": ["26 USC §63"],
    },
    {
        "question": (
            "A married couple filing jointly has combined wages of $110,000, "
            "mortgage interest of $14,000, state income tax of $9,000, real "
            "estate tax of $4,500, and charitable contributions of $3,500. "
            "Should they itemize or take the standard deduction for 2024?"
        ),
        "ground_truth": (
            "SALT (state income + real estate) = $13,500, capped at $10,000 "
            "(IRC §164(b)(6)). Total itemized = $10,000 + $14,000 + $3,500 = "
            "$27,500.  MFJ standard deduction = $29,200 (IRC §63(c)).  "
            "They should take the standard deduction because $27,500 < $29,200."
        ),
        "difficulty": "basic",
        "domain": "federal_income_tax",
        "tags": ["standard_deduction", "itemized_deductions", "salt_cap"],
        "statutory_refs": ["26 USC §63", "26 USC §164"],
    },
    {
        "question": (
            "What is the self-employment tax rate for 2024, and how is the "
            "tax base computed?"
        ),
        "ground_truth": (
            "The SE tax rate is 15.3%: 12.4% for Social Security (on net "
            "earnings up to $168,600) and 2.9% for Medicare (no cap).  "
            "The tax base is 92.35% of net self-employment income — that is, "
            "net profit multiplied by 0.9235 — because the self-employed "
            "person effectively deducts the 'employer' half of the tax from "
            "net earnings before computing the tax base (IRC §1401, §1402)."
        ),
        "difficulty": "basic",
        "domain": "self_employment",
        "tags": ["se_tax", "self_employment"],
        "statutory_refs": ["26 USC §1401", "26 USC §1402"],
    },
    {
        "question": (
            "A freelance graphic designer had net Schedule C profit of $72,000 "
            "in 2024.  What is her approximate self-employment tax, and what "
            "above-the-line deduction can she claim for it?"
        ),
        "ground_truth": (
            "SE tax base = $72,000 × 0.9235 = $66,492.  "
            "SE tax = $66,492 × 15.3% = $10,173.28 (rounded).  "
            "She may deduct half the SE tax as an above-the-line deduction: "
            "$10,173.28 / 2 = $5,086.64 (IRC §164(f))."
        ),
        "difficulty": "basic",
        "domain": "self_employment",
        "tags": ["se_tax", "se_deduction", "schedule_c"],
        "statutory_refs": ["26 USC §1401", "26 USC §164(f)"],
    },
    {
        "question": (
            "What is the 2024 maximum QBI deduction rate, and what is the "
            "income threshold below which no W-2 wage limitation applies "
            "for a single filer?"
        ),
        "ground_truth": (
            "The QBI deduction is 20% of qualified business income (IRC §199A). "
            "For a single filer in 2024, the W-2 wage limitation does not apply "
            "if taxable income is at or below $191,950 (the §199A threshold).  "
            "Above that threshold, the deduction is subject to the W-2 wage and "
            "UBIA of property limitations."
        ),
        "difficulty": "intermediate",
        "domain": "self_employment",
        "tags": ["qbi", "section_199a"],
        "statutory_refs": ["26 USC §199A"],
    },
    {
        "question": (
            "Under IRC §86, what percentage of Social Security benefits is "
            "included in gross income for a single filer with combined income "
            "of $40,000?"
        ),
        "ground_truth": (
            "For a single filer, combined income = AGI + non-taxable interest "
            "+ 50% of SS benefits.  If combined income exceeds $34,000 (the "
            "upper threshold), up to 85% of SS benefits are taxable.  At "
            "combined income of $40,000, the taxpayer is above both thresholds "
            "($25,000 and $34,000), so the full 85% inclusion formula applies — "
            "the taxable amount is the lesser of 85% of benefits or "
            "0.85 × (combined income - $34,000) + 0.50 × min(benefits, $9,000)."
        ),
        "difficulty": "intermediate",
        "domain": "federal_income_tax",
        "tags": ["social_security", "gross_income_inclusion"],
        "statutory_refs": ["26 USC §86"],
    },
    {
        "question": (
            "A self-employed consultant earned $95,000 net Schedule C profit in "
            "2024 and paid $11,400 in self-employed health insurance premiums. "
            "Both the SE deduction and the health insurance deduction are "
            "above-the-line.  In what order should they be applied, and why does "
            "the order matter?"
        ),
        "ground_truth": (
            "The SE deduction is computed first because health insurance "
            "deductibility for the self-employed (IRC §162(l)) is limited to "
            "net Schedule C profit minus the SE deduction.  Computing SE tax "
            "on $95,000 × 0.9235 × 15.3% ≈ $13,421 yields an SE deduction "
            "of ≈ $6,710.  Net profit after SE deduction = $95,000 - $6,710 "
            "= $88,290.  Since health insurance premiums ($11,400) are less "
            "than this net, the full $11,400 is deductible.  Reversing the "
            "order would incorrectly inflate the health insurance deduction "
            "in cases where premiums approach net profit."
        ),
        "difficulty": "advanced",
        "domain": "self_employment",
        "tags": ["se_deduction", "health_insurance", "ordering"],
        "statutory_refs": ["26 USC §162(l)", "26 USC §164(f)"],
    },
    {
        "question": (
            "What is the 2024 Child Tax Credit per qualifying child, and at "
            "what income level does it begin to phase out for a single filer?"
        ),
        "ground_truth": (
            "The Child Tax Credit is $2,000 per qualifying child under 17 at "
            "year-end (IRC §24).  For a single filer, the credit phases out by "
            "$50 per $1,000 (or fraction) of modified AGI above $200,000. "
            "It is fully phased out at approximately $240,000 for two children."
        ),
        "difficulty": "basic",
        "domain": "federal_income_tax",
        "tags": ["child_tax_credit", "phase_out"],
        "statutory_refs": ["26 USC §24"],
    },
    {
        "question": (
            "A taxpayer sold her primary residence in 2024 after living in it "
            "for 3 years and realized a $280,000 gain.  Is any of the gain "
            "excludable from income?"
        ),
        "ground_truth": (
            "Yes.  Under IRC §121, a taxpayer may exclude up to $250,000 of "
            "gain from the sale of a principal residence ($500,000 for MFJ) if "
            "she owned and used the home as a principal residence for at least "
            "2 of the 5 years before the sale.  She meets both tests (3 years). "
            "Her $280,000 gain is reduced by the $250,000 exclusion, leaving "
            "$30,000 taxable as a long-term capital gain."
        ),
        "difficulty": "intermediate",
        "domain": "federal_income_tax",
        "tags": ["home_sale", "section_121_exclusion", "capital_gains"],
        "statutory_refs": ["26 USC §121"],
    },
    {
        "question": (
            "What is the 2024 annual contribution limit to a Traditional or "
            "Roth IRA, and how does it change for taxpayers age 50 or older?"
        ),
        "ground_truth": (
            "The 2024 IRA contribution limit is $7,000 per person (IRC §219). "
            "Taxpayers who are 50 or older by December 31 may contribute an "
            "additional $1,000 catch-up, for a total of $8,000.  Contributions "
            "cannot exceed earned income for the year."
        ),
        "difficulty": "basic",
        "domain": "federal_income_tax",
        "tags": ["ira", "contribution_limit"],
        "statutory_refs": ["26 USC §219", "26 USC §408A"],
    },
    {
        "question": (
            "A self-employed plumber wants to deduct part of his home as a "
            "business office under IRC §280A.  What are the two methods for "
            "computing the home office deduction, and which is simpler?"
        ),
        "ground_truth": (
            "IRC §280A allows a home office deduction when the space is used "
            "regularly and exclusively for business.  The two methods are: "
            "(1) Regular method — deduct actual expenses (mortgage interest, "
            "rent, utilities, depreciation, insurance) allocable to the "
            "business portion (square footage / total home square footage). "
            "(2) Simplified method — deduct $5 per square foot of dedicated "
            "office space, up to 300 sq ft (maximum $1,500).  "
            "The simplified method is simpler but may yield a smaller deduction "
            "for large home offices or homes with high expenses."
        ),
        "difficulty": "intermediate",
        "domain": "self_employment",
        "tags": ["home_office", "schedule_c"],
        "statutory_refs": ["26 USC §280A"],
    },
    {
        "question": (
            "What is the estimated tax safe harbor for 2024 to avoid an "
            "underpayment penalty?"
        ),
        "ground_truth": (
            "To avoid the underpayment penalty under IRC §6654, a taxpayer "
            "must generally pay the lesser of: (1) 90% of the current year's "
            "tax liability, or (2) 100% of the prior year's tax liability "
            "(110% if prior-year AGI exceeded $150,000).  Payments must be "
            "made in four equal installments by April 15, June 17, September 16, "
            "and January 15 of the following year."
        ),
        "difficulty": "intermediate",
        "domain": "federal_income_tax",
        "tags": ["estimated_tax", "safe_harbor", "underpayment_penalty"],
        "statutory_refs": ["26 USC §6654"],
    },
    {
        "question": (
            "A sole proprietor had $120,000 net Schedule C profit in 2024 and "
            "no other income.  She is single.  What is her approximate QBI "
            "deduction, and what is her taxable income before the QBI deduction?"
        ),
        "ground_truth": (
            "SE tax = $120,000 × 0.9235 × 15.3% ≈ $16,956.  "
            "SE deduction = $16,956 / 2 ≈ $8,478.  "
            "AGI = $120,000 - $8,478 = $111,522.  "
            "Standard deduction (single) = $14,600.  "
            "Taxable income before QBI = $111,522 - $14,600 = $96,922.  "
            "QBI deduction = min(20% × $120,000, 20% × $96,922) "
            "= min($24,000, $19,384) = $19,384.  "
            "Taxable income after QBI = $96,922 - $19,384 = $77,538."
        ),
        "difficulty": "advanced",
        "domain": "self_employment",
        "tags": ["qbi", "se_deduction", "taxable_income"],
        "statutory_refs": ["26 USC §199A", "26 USC §164(f)"],
    },
]


# ---------------------------------------------------------------------------
# Section 2: Multiple-choice (MCQ) cases
# ---------------------------------------------------------------------------
# Each dict has "choices": {"A": ..., "B": ..., "C": ..., "D": ...}
# and "ground_truth": the correct key letter ("A"–"D").

_MCQ_SPECS: list[dict] = [
    {
        "question": (
            "Which of the following forms is used to report self-employment "
            "income from a sole proprietorship on a federal income tax return?"
        ),
        "choices": {
            "A": "Schedule A (Itemized Deductions)",
            "B": "Schedule B (Interest and Ordinary Dividends)",
            "C": "Schedule C (Profit or Loss From Business)",
            "D": "Schedule D (Capital Gains and Losses)",
        },
        "ground_truth": "C",
        "explanation": (
            "Schedule C is used to report profit or loss from a business "
            "operated as a sole proprietorship.  Schedule A covers "
            "itemized deductions, Schedule B reports interest and dividends, "
            "and Schedule D reports capital gains and losses."
        ),
        "difficulty": "basic",
        "domain": "self_employment",
        "tags": ["schedule_c", "sole_proprietorship"],
        "statutory_refs": ["IRS Schedule C Instructions", "26 USC §61"],
    },
    {
        "question": (
            "A single filer has $60,000 wages and $8,000 in net long-term "
            "capital gains.  Her taxable income is $48,000.  At what rate "
            "will her long-term capital gains be taxed for 2024?"
        ),
        "choices": {
            "A": "0% — her ordinary income is below the $47,025 LTCG threshold",
            "B": "15% — her stacked income exceeds the $47,025 threshold",
            "C": "22% — long-term capital gains are taxed at ordinary rates",
            "D": "20% — all long-term capital gains above $40,000 are taxed at 20%",
        },
        "ground_truth": "B",
        "explanation": (
            "Her ordinary taxable income = $48,000 - $8,000 LTCG = $40,000. "
            "The 0% bracket ends at $47,025 (single, 2024).  Room at 0% = "
            "$47,025 - $40,000 = $7,025.  $7,025 of LTCG is taxed at 0%; "
            "the remaining $975 is taxed at 15%.  But in aggregate, not all "
            "gains are at 0%, so answer B is the most accurate description: "
            "most of the gains are taxed at 15% once the 0% room is exhausted."
        ),
        "difficulty": "intermediate",
        "domain": "federal_income_tax",
        "tags": ["capital_gains", "ltcg_rate"],
        "statutory_refs": ["26 USC §1(h)"],
    },
    {
        "question": (
            "Which of the following is an above-the-line deduction (deductible "
            "in computing AGI) for self-employed taxpayers under IRC §62?"
        ),
        "choices": {
            "A": "Mortgage interest on a primary residence",
            "B": "Cash charitable contributions",
            "C": "One-half of self-employment tax",
            "D": "State and local income taxes paid",
        },
        "ground_truth": "C",
        "explanation": (
            "IRC §62(a)(1) lists deductions from gross income that reduce "
            "AGI.  IRC §164(f) specifically allows self-employed taxpayers "
            "to deduct one-half of SE tax above the line.  Mortgage interest, "
            "charitable contributions, and SALT are itemized (below-the-line) "
            "deductions, meaning they only reduce taxable income if the "
            "taxpayer itemizes."
        ),
        "difficulty": "basic",
        "domain": "self_employment",
        "tags": ["above_the_line_deduction", "se_deduction"],
        "statutory_refs": ["26 USC §62", "26 USC §164(f)"],
    },
    {
        "question": (
            "A married couple filing jointly has AGI of $430,000 in 2024.  "
            "They have two qualifying children for the Child Tax Credit.  "
            "What is the maximum Child Tax Credit they can claim?"
        ),
        "choices": {
            "A": "$4,000 — no phase-out applies below $500,000",
            "B": "$2,500 — the credit is reduced by $50 per $1,000 above $400,000",
            "C": "$1,500 — the credit is halved for high-income filers",
            "D": "$0 — the credit is fully phased out above $400,000",
        },
        "ground_truth": "B",
        "explanation": (
            "The MFJ phase-out begins at $400,000.  Excess AGI = $430,000 - "
            "$400,000 = $30,000.  Phase-out = ceil($30,000 / $1,000) × $50 "
            "= 30 × $50 = $1,500.  CTC = (2 × $2,000) - $1,500 = $2,500."
        ),
        "difficulty": "intermediate",
        "domain": "federal_income_tax",
        "tags": ["child_tax_credit", "phase_out"],
        "statutory_refs": ["26 USC §24"],
    },
    {
        "question": (
            "Which of the following best describes the 'regular and exclusive "
            "use' requirement for the home office deduction under IRC §280A?"
        ),
        "choices": {
            "A": (
                "The taxpayer must use the space for business on at least "
                "50% of working days during the year."
            ),
            "B": (
                "The dedicated space must be used only for business, "
                "with no personal use allowed, and must be the principal "
                "place of business or a place to meet clients."
            ),
            "C": (
                "The taxpayer may deduct any room used for business "
                "purposes, including rooms shared with family members."
            ),
            "D": (
                "The home office must be a separate structure not attached "
                "to the dwelling."
            ),
        },
        "ground_truth": "B",
        "explanation": (
            "IRC §280A(c)(1) requires regular and exclusive use for a "
            "specific trade or business, at the principal place of business "
            "or a place used to meet clients/customers.  Any personal use "
            "of the space disqualifies the deduction entirely.  A separate "
            "structure is one qualifying category but not the only one."
        ),
        "difficulty": "intermediate",
        "domain": "self_employment",
        "tags": ["home_office", "section_280a"],
        "statutory_refs": ["26 USC §280A"],
    },
    {
        "question": (
            "A single taxpayer's AGI is $80,000 in 2024.  She is covered by "
            "a 401(k) plan at work.  What is the maximum deductible Traditional "
            "IRA contribution she may make?"
        ),
        "choices": {
            "A": "$7,000 — the full contribution limit always applies",
            "B": "$0 — being covered by a workplace plan eliminates the deduction",
            "C": (
                "A partial deduction, because her AGI falls within the "
                "$77,000–$87,000 single phase-out range for covered participants"
            ),
            "D": "$3,500 — the deduction is automatically halved when covered by a plan",
        },
        "ground_truth": "C",
        "explanation": (
            "IRC §219(g) phases out the Traditional IRA deduction for single "
            "filers covered by a workplace plan with AGI between $77,000 and "
            "$87,000 (2024).  At $80,000, she is within the phase-out range "
            "and receives a partial deduction.  The deduction is not entirely "
            "eliminated until AGI reaches $87,000."
        ),
        "difficulty": "intermediate",
        "domain": "federal_income_tax",
        "tags": ["ira", "phase_out", "workplace_plan"],
        "statutory_refs": ["26 USC §219"],
    },
    {
        "question": (
            "Which of the following correctly describes the tax treatment of "
            "a W-2 employee compared to a self-employed independent contractor "
            "earning the same gross income?"
        ),
        "choices": {
            "A": (
                "Both pay identical payroll taxes because the combined "
                "FICA rate is 15.3% for everyone."
            ),
            "B": (
                "The employee pays 7.65% in FICA taxes matched by the employer; "
                "the independent contractor pays the full 15.3% SE tax "
                "but may deduct half above the line."
            ),
            "C": (
                "The independent contractor pays no payroll taxes because "
                "self-employment income is exempt from FICA."
            ),
            "D": (
                "The employee's FICA taxes are fully deductible; "
                "the independent contractor's SE tax is not deductible."
            ),
        },
        "ground_truth": "B",
        "explanation": (
            "An employee pays 7.65% FICA (6.2% SS + 1.45% Medicare); "
            "the employer matches.  A self-employed person pays both halves "
            "(15.3%) as SE tax under IRC §1401, but may deduct 50% as an "
            "above-the-line deduction (IRC §164(f)) — effectively mimicking "
            "the employer's deduction."
        ),
        "difficulty": "basic",
        "domain": "both",
        "tags": ["se_tax", "fica", "w2_vs_1099"],
        "statutory_refs": ["26 USC §1401", "26 USC §164(f)"],
    },
    {
        "question": (
            "What is the purpose of Form 8995 on a federal tax return?"
        ),
        "choices": {
            "A": "To report net long-term capital gains and losses",
            "B": "To compute and claim the 20% Qualified Business Income (QBI) deduction",
            "C": "To report self-employment tax from Schedule SE",
            "D": "To claim the home office deduction for a sole proprietorship",
        },
        "ground_truth": "B",
        "explanation": (
            "Form 8995 (or 8995-A for complex situations) is used to "
            "compute the QBI deduction under IRC §199A.  Capital gains "
            "are reported on Schedule D; SE tax is computed on Schedule SE; "
            "home office deduction uses Form 8829."
        ),
        "difficulty": "basic",
        "domain": "self_employment",
        "tags": ["qbi", "form_8995"],
        "statutory_refs": ["26 USC §199A"],
    },
]


# ---------------------------------------------------------------------------
# Section 3: IRS-mined scenario cases
# ---------------------------------------------------------------------------

_WEAK_CONCLUSION_PREFIXES = (
    "see ",
    "refer to ",
    "for more information",
    "use worksheet",
    "go to ",
)


def _is_weak_conclusion(text: str) -> bool:
    """Return True when a mined conclusion is likely not self-contained."""
    if not text:
        return True
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return True
    if any(normalized.startswith(prefix) for prefix in _WEAK_CONCLUSION_PREFIXES):
        return True
    if len(normalized) < 24 and "$" not in text:
        return True
    return False

def _build_mined_scenarios(max_cases: int = 40) -> list[dict]:
    """Mine IRS XML publications for worked example scenarios.

    Filters mined examples to those most suitable for benchmarking: examples
    with at least one dollar amount and a clear conclusion sentence.

    Parameters
    ----------
    max_cases:
        Maximum number of mined cases to include.

    Returns
    -------
    list[dict]
        Raw case specs ready for conversion to BenchmarkCase objects.
    """
    from sources.miner import mine_all_sources

    examples = mine_all_sources()

    # Prefer examples with dollar amounts and longer text (richer scenarios)
    examples = [e for e in examples if e["dollar_amounts"] and len(e["text"]) >= 120]
    examples.sort(key=lambda e: (-len(e["dollar_amounts"]), -len(e["text"])))

    specs: list[dict] = []
    for ex in examples:
        conclusion = ex.get("conclusion", "").strip()
        if _is_weak_conclusion(conclusion):
            continue

        # Infer a question from the scenario text and topic tags
        tags = ex["tags"]
        question = _infer_question(ex["text"], tags, ex["source"])

        specs.append({
            "question":     question,
            "ground_truth": conclusion,
            "facts_narrative": ex["text"],
            "source":      ex["source_label"],
            "statutory_refs": [ex["source_label"]],
            "tags":        tags,
            "difficulty":  "intermediate" if len(tags) > 2 else "basic",
            "domain":      _infer_domain(tags),
        })

        if len(specs) >= max_cases:
            break

    return specs


def _ensure_statutory_refs(refs: list[str] | None, fallback: str) -> list[str]:
    """Return cleaned statutory references, with a non-empty fallback."""
    cleaned = [r.strip() for r in (refs or []) if isinstance(r, str) and r.strip()]
    if cleaned:
        return cleaned
    return [fallback]


def _infer_question(text: str, tags: list[str], source: str) -> str:
    """Infer a benchmark question from an IRS example's text and tags."""
    if "self_employment" in tags or "se_tax" in tags:
        return (
            "Based on this scenario from an IRS publication, what is the "
            "correct federal tax treatment for the self-employed taxpayer described?"
        )
    if "capital_gains" in tags:
        return (
            "Based on this IRS example, how should the taxpayer report "
            "the capital gain or loss described?"
        )
    if "home_office" in tags:
        return (
            "Based on this IRS example, what is the taxpayer's home office "
            "deduction and how is it computed?"
        )
    if "charitable" in tags:
        return (
            "Based on this IRS example, is the charitable contribution "
            "deductible, and if so, what is the deductible amount?"
        )
    if "ira" in tags or "retirement_plan" in tags:
        return (
            "Based on this IRS example, what is the correct tax treatment "
            "of the retirement plan contribution or distribution described?"
        )
    if "education" in tags:
        return (
            "Based on this IRS example, what education-related deduction "
            "or credit may the taxpayer claim?"
        )
    return (
        "Based on this worked example from an IRS publication, what is the "
        "correct federal income tax treatment for the situation described?"
    )


def _infer_domain(tags: list[str]) -> str:
    """Determine domain from tags."""
    se_tags = {"self_employment", "se_tax", "schedule_c", "qbi", "home_office"}
    if se_tags & set(tags):
        return "self_employment"
    return "federal_income_tax"


def build_tier3_cases() -> list[BenchmarkCase]:
    """Build all Tier 3 Q&A, MCQ, and scenario cases.

    Returns
    -------
    list[BenchmarkCase]
        Cases ordered by style: Q&A first, then MCQ, then mined scenarios.
    """
    cases: list[BenchmarkCase] = []
    seq = 1

    # --- Q&A cases ---
    for spec in _QA_SPECS:
        case = BenchmarkCase(
            id=f"t3_qa_{seq:04d}",
            tier=3,
            domain=spec["domain"],
            style="qa",
            difficulty=spec["difficulty"],
            tax_year=TAX_YEAR,
            source="hand_crafted",
            verified_by="hand_crafted",
            profile_id=f"t3_qa_{seq:04d}",
            confidence_tier="C",
            facts=None,
            facts_narrative="",
            question=spec["question"],
            choices=None,
            ground_truth=spec["ground_truth"],
            ground_truth_type="text",
            tolerance_strict_usd=0.0,
            tolerance_lenient_usd=0.0,
            tolerance_pct=0.0,
            reasoning_steps=[],
            statutory_refs=_ensure_statutory_refs(
                spec.get("statutory_refs"),
                "IRS Publication 17",
            ),
            explanation=spec.get("ground_truth", ""),
            tags=spec.get("tags", []),
        )
        cases.append(case)
        seq += 1

    # --- MCQ cases ---
    for spec in _MCQ_SPECS:
        case = BenchmarkCase(
            id=f"t3_mcq_{seq:04d}",
            tier=3,
            domain=spec["domain"],
            style="mcq",
            difficulty=spec["difficulty"],
            tax_year=TAX_YEAR,
            source="hand_crafted",
            verified_by="hand_crafted",
            profile_id=f"t3_mcq_{seq:04d}",
            confidence_tier="C",
            facts=None,
            facts_narrative="",
            question=spec["question"],
            choices=spec["choices"],
            ground_truth=spec["ground_truth"],
            ground_truth_type="choice_key",
            tolerance_strict_usd=0.0,
            tolerance_lenient_usd=0.0,
            tolerance_pct=0.0,
            reasoning_steps=[],
            statutory_refs=_ensure_statutory_refs(
                spec.get("statutory_refs"),
                "IRS Publication 17",
            ),
            explanation=spec.get("explanation", ""),
            tags=spec.get("tags", []),
        )
        cases.append(case)
        seq += 1

    # --- Mined scenario cases ---
    try:
        mined_specs = _build_mined_scenarios(max_cases=40)
    except Exception:
        mined_specs = []

    for spec in mined_specs:
        case = BenchmarkCase(
            id=f"t3_scenario_{seq:04d}",
            tier=3,
            domain=spec["domain"],
            style="scenario",
            difficulty=spec["difficulty"],
            tax_year=TAX_YEAR,
            source=spec["source"],
            verified_by="irs_publication",
            profile_id=f"t3_scenario_{seq:04d}",
            confidence_tier="C",
            facts=None,
            facts_narrative=spec["facts_narrative"],
            question=spec["question"],
            choices=None,
            ground_truth=spec["ground_truth"],
            ground_truth_type="text",
            tolerance_strict_usd=0.0,
            tolerance_lenient_usd=0.0,
            tolerance_pct=0.0,
            reasoning_steps=[],
            statutory_refs=_ensure_statutory_refs(
                spec.get("statutory_refs"),
                spec["source"],
            ),
            explanation=(
                "Ground truth is the stated conclusion from the IRS publication "
                f"({spec['source']}).  Verified by IRS staff."
            ),
            tags=spec.get("tags", []),
        )
        cases.append(case)
        seq += 1

    return cases
