"""Deterministic 2024 federal income tax computation engine.

Computes the federal income tax liability for a ``TaxpayerFacts`` record
using the 2024 IRC rules encoded in ``config.py``.  No external packages
are required — the engine is pure Python and fully reproducible.

Scope and limitations
---------------------
This engine covers the most common individual filer situations:

  - W-2 wages, self-employment (Schedule C), interest, dividends, pensions,
    Social Security (85% inclusion rule applied), rental income, capital gains
  - Standard deduction vs. itemized deductions (SALT cap applied)
  - SE tax (IRC §1401) and the 50% SE deduction (IRC §164(f))
  - QBI deduction (IRC §199A) — basic case only (income below threshold)
  - Child Tax Credit (IRC §24) — no Additional CTC computed here
    - Net Investment Income Tax (IRC §1411)
    - Additional Medicare Tax on earned income (IRC §3101(b)(2) / §1401(b)(2))
  - Long-term capital gains preferential rates (IRC §1(h))
    - IRA contribution deductibility (with workplace-plan phase-out support)
  - Educator expense deduction (IRC §62(a)(2)(D))
  - Student loan interest deduction (IRC §221)

Not modeled (flag complex cases for external tool validation):
  - AMT (IRC §55–59)
  - Passive activity loss rules
  - Net operating loss carryovers
  - Foreign tax credit
  - Phase-out of itemized deductions (Pease limitation repealed by TCJA)
  - QBI W-2 wage and UBIA limitations for income above §199A threshold
  - SSTB phase-outs
  - Earned Income Tax Credit
  - Premium Tax Credit
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ADDL_MEDICARE_RATE,
    ADDL_MEDICARE_THRESHOLD,
    CTC_PER_CHILD,
    CTC_PHASEOUT_THRESHOLD,
    IRA_CATCH_UP_LIMIT,
    IRA_CONTRIBUTION_LIMIT,
    IRA_PHASEOUT_MFJ_HIGH,
    IRA_PHASEOUT_MFJ_LOW,
    IRA_PHASEOUT_SINGLE_HIGH,
    IRA_PHASEOUT_SINGLE_LOW,
    LTCG_0_THRESHOLD,
    LTCG_15_THRESHOLD,
    NIIT_RATE,
    QBI_DEDUCTION_RATE,
    QBI_THRESHOLD,
    SALT_CAP,
    SE_TAX_RATE,
    SE_WAGE_BASE,
    STANDARD_DEDUCTION,
    TAX_BRACKETS,
)
from schema import TaxpayerFacts


@dataclass
class TaxResult:
    """All intermediate and final values from the 2024 tax computation.

    Dollar amounts are rounded to 2 decimal places. The ``reasoning_steps``
    list records each calculation in plain English, suitable for use as the
    ``reasoning_steps`` field in a ``BenchmarkCase``.
    """

    # Income
    gross_income: float = 0.0
    taxable_ss_benefits: float = 0.0
    total_ordinary_income: float = 0.0

    # Above-the-line deductions
    se_tax: float = 0.0
    se_deduction: float = 0.0             # half of SE tax (IRC §164(f))
    educator_deduction: float = 0.0
    student_loan_deduction: float = 0.0
    ira_deduction: float = 0.0
    self_emp_health_deduction: float = 0.0
    sep_ira_deduction: float = 0.0
    total_above_line_deductions: float = 0.0

    # AGI
    agi: float = 0.0

    # Below-the-line deductions
    itemized_deductions: float = 0.0
    standard_deduction: float = 0.0
    deduction_used: float = 0.0
    uses_itemized: bool = False

    # QBI
    qbi_deduction: float = 0.0

    # Taxable income
    taxable_income: float = 0.0

    # Tax computation
    ordinary_taxable_income: float = 0.0  # taxable income minus LTCG and QD
    regular_tax: float = 0.0
    ltcg_tax: float = 0.0
    se_tax_total: float = 0.0             # same as se_tax, re-stated for clarity
    niit: float = 0.0
    addl_medicare_tax: float = 0.0
    total_tax_before_credits: float = 0.0

    # Credits
    child_tax_credit: float = 0.0
    total_credits: float = 0.0

    # Net liability
    net_tax: float = 0.0
    effective_rate: float = 0.0

    # Flags
    needs_amt_review: bool = False        # True if AMT may apply (not computed)
    needs_external_validation: bool = False  # True if out of scope

    reasoning_steps: list[str] = field(default_factory=list)


def _cents(value: float) -> float:
    """Round to the nearest cent."""
    return round(value, 2)


def _tax_from_brackets(taxable_income: float, filing_status: str) -> float:
    """Compute regular income tax from 2024 marginal tax brackets.

    Parameters
    ----------
    taxable_income:
        Taxable income (after deductions, before credits).
    filing_status:
        One of ``"single"``, ``"mfj"``, ``"mfs"``, ``"hoh"``, ``"qss"``.

    Returns
    -------
    float
        Tax liability in dollars, rounded to 2 decimal places.
    """
    if taxable_income <= 0:
        return 0.0

    brackets = TAX_BRACKETS[filing_status]
    tax = 0.0
    prev_upper = 0.0

    for upper, rate in brackets:
        if taxable_income <= prev_upper:
            break
        taxable_in_bracket = min(taxable_income, upper) - prev_upper
        tax += taxable_in_bracket * rate
        prev_upper = upper

    return _cents(tax)


def _ltcg_tax(
    ltcg_plus_qd: float,
    ordinary_ti: float,
    filing_status: str,
) -> float:
    """Compute preferential tax on long-term capital gains and qualified dividends.

    Uses the stacking method required by IRC §1(h): LTCG/QD income is placed
    on top of ordinary taxable income when determining which rate bracket applies.

    Parameters
    ----------
    ltcg_plus_qd:
        Net long-term capital gain plus qualified dividends.
    ordinary_ti:
        Ordinary taxable income (taxable income minus LTCG and QD).
    filing_status:
        Filing status key.

    Returns
    -------
    float
        Tax on LTCG / QD at preferential rates.
    """
    if ltcg_plus_qd <= 0:
        return 0.0

    zero_threshold = LTCG_0_THRESHOLD[filing_status]
    fifteen_threshold = LTCG_15_THRESHOLD[filing_status]

    # Room remaining in the 0% bracket
    room_at_zero = max(0.0, zero_threshold - ordinary_ti)
    taxed_at_zero = min(ltcg_plus_qd, room_at_zero)
    remaining = ltcg_plus_qd - taxed_at_zero

    if remaining <= 0:
        return 0.0

    # Room remaining in the 15% bracket
    top_of_ordinary = ordinary_ti + ltcg_plus_qd
    room_at_fifteen = max(0.0, fifteen_threshold - max(ordinary_ti, zero_threshold))
    taxed_at_fifteen = min(remaining, room_at_fifteen)
    remaining -= taxed_at_fifteen

    taxed_at_twenty = remaining

    tax = (taxed_at_fifteen * 0.15) + (taxed_at_twenty * 0.20)
    return _cents(tax)


def _taxable_social_security(facts: TaxpayerFacts, agi_before_ss: float) -> float:
    """Compute the taxable portion of Social Security benefits (IRC §86).

    Up to 85% of benefits are taxable depending on "combined income"
    (AGI before SS + half of SS benefits).  This uses the simplified
    single/MFJ thresholds; MFS is treated as single.

    Parameters
    ----------
    facts:
        Taxpayer profile.
    agi_before_ss:
        AGI computed without including any Social Security benefits.

    Returns
    -------
    float
        Taxable Social Security benefits.
    """
    ss = facts.social_security_benefits
    if ss <= 0:
        return 0.0

    combined = agi_before_ss + 0.5 * ss
    filing = facts.filing_status

    # Thresholds from IRC §86(c)
    if filing in ("mfj", "qss"):
        low, high = 32_000, 44_000
    else:
        low, high = 25_000, 34_000

    if combined <= low:
        return 0.0
    elif combined <= high:
        # Up to 50% of benefits are taxable
        taxable = min(0.5 * (combined - low), 0.5 * ss)
    else:
        # Up to 85% of benefits are taxable
        taxable = min(
            0.85 * ss,
            0.85 * (combined - high) + 0.5 * min(ss, high - low),
        )

    return _cents(taxable)


def _ira_deduction(facts: TaxpayerFacts, agi_before_ira: float) -> float:
    """Compute the deductible IRA contribution (IRC §219).

    Applies the basic contribution limit and the phase-out for filers covered
    by a workplace retirement plan (``facts.covered_by_workplace_plan``).
    If ``facts.ira_deduction`` is already set (non-zero), that value is used
    directly without re-computing.

    Parameters
    ----------
    facts:
        Taxpayer profile.
    agi_before_ira:
        AGI before the IRA deduction (used for phase-out test).

    Returns
    -------
    float
        Deductible IRA contribution amount.
    """
    if facts.ira_deduction:
        return float(facts.ira_deduction)

    if facts.ira_contribution <= 0:
        return 0.0

    # Contribution limit (age 50+ catch-up)
    age = facts.age_primary
    limit = IRA_CATCH_UP_LIMIT if age >= 50 else IRA_CONTRIBUTION_LIMIT
    contribution = min(facts.ira_contribution, limit)

    if not facts.covered_by_workplace_plan:
        return _cents(contribution)

    # Covered by workplace plan: apply 2024 phase-out ranges.
    if facts.filing_status in ("mfj", "qss"):
        low, high = IRA_PHASEOUT_MFJ_LOW, IRA_PHASEOUT_MFJ_HIGH
    else:
        low, high = IRA_PHASEOUT_SINGLE_LOW, IRA_PHASEOUT_SINGLE_HIGH

    if agi_before_ira <= low:
        return _cents(contribution)
    if agi_before_ira >= high:
        return 0.0

    phaseout_fraction = (high - agi_before_ira) / (high - low)
    deductible = contribution * phaseout_fraction
    return _cents(max(0.0, deductible))


def compute(facts: TaxpayerFacts) -> TaxResult:
    """Compute 2024 federal income tax for the given taxpayer profile.

    Parameters
    ----------
    facts:
        Taxpayer's financial profile for Tax Year 2024.

    Returns
    -------
    TaxResult
        Full computation trace including all intermediate values and a
        list of plain-English reasoning steps.
    """
    result = TaxResult()
    steps = result.reasoning_steps
    fs = facts.filing_status

    # ------------------------------------------------------------------
    # Step 1: Gross income (before SS inclusion decision)
    # ------------------------------------------------------------------
    ordinary_income = (
        facts.wages
        + facts.spouse_wages
        + facts.taxable_interest
        + facts.ordinary_dividends
        + facts.capital_gains_net
        + facts.state_refund
        + facts.alimony_received
        + facts.ira_distributions
        + facts.pension_income
        + facts.rental_income_net
        + facts.farm_income
        + facts.other_income
    )

    steps.append(
        f"Ordinary income sources: wages ${facts.wages:,}, spouse wages "
        f"${facts.spouse_wages:,}, interest ${facts.taxable_interest:,}, "
        f"dividends ${facts.ordinary_dividends:,}, net long-term capital gains "
        f"${facts.capital_gains_net:,}, pensions "
        f"${facts.pension_income:,}, IRA distributions ${facts.ira_distributions:,}, "
        f"rental (net) ${facts.rental_income_net:,}, other ${facts.other_income:,} "
        f"= ${ordinary_income:,}."
    )

    # ------------------------------------------------------------------
    # Step 2: Self-employment tax and deduction
    # ------------------------------------------------------------------
    se_income = facts.business_income
    se_base = 0.0
    se_tax = 0.0
    se_deduction = 0.0

    if se_income > 0:
        # Net earnings subject to SE tax = 92.35% of net SE income
        se_base = _cents(se_income * 0.9235)

        # SS portion: 12.4% up to the SS wage base
        ss_portion = _cents(min(se_base, SE_WAGE_BASE) * 0.124)

        # Medicare portion: 2.9% on all net earnings (no cap)
        mc_portion = _cents(se_base * 0.029)

        se_tax = _cents(ss_portion + mc_portion)
        se_deduction = _cents(se_tax / 2)

        steps.append(
            f"Self-employment income: ${se_income:,}. "
            f"SE tax base (×0.9235) = ${se_base:,.2f}. "
            f"SS portion (12.4% up to ${SE_WAGE_BASE:,}) = ${ss_portion:,.2f}. "
            f"Medicare portion (2.9%) = ${mc_portion:,.2f}. "
            f"SE tax = ${se_tax:,.2f}. "
            f"SE deduction (50%) = ${se_deduction:,.2f} (IRC §164(f))."
        )

    result.se_tax = se_tax
    result.se_deduction = se_deduction
    result.se_tax_total = se_tax

    # ------------------------------------------------------------------
    # Step 3: Above-the-line deductions (before SS inclusion)
    # ------------------------------------------------------------------
    educator = min(facts.educator_expenses, 300)   # $300 per teacher for 2024
    student_loan = min(facts.student_loan_interest, 2_500)
    self_emp_hi = facts.self_employed_health_ins
    sep_ira = facts.sep_simple_ira_deduction

    # Compute a provisional AGI (without IRA, without SS) for IRA phase-out
    provisional_agi = (
        ordinary_income
        + se_income
        - se_deduction
        - educator
        - student_loan
        - self_emp_hi
        - sep_ira
    )

    ira_ded = _ira_deduction(facts, provisional_agi)

    total_above = _cents(
        se_deduction + educator + student_loan + ira_ded + self_emp_hi + sep_ira
    )

    if total_above > 0:
        steps.append(
            f"Above-the-line deductions: SE deduction ${se_deduction:,.2f}, "
            f"educator expenses ${educator:,}, student loan interest ${student_loan:,}, "
            f"IRA deduction ${ira_ded:,.2f}, self-employed health insurance "
            f"${self_emp_hi:,}, SEP/SIMPLE IRA ${sep_ira:,}. "
            f"Total = ${total_above:,.2f}."
        )

    result.educator_deduction = educator
    result.student_loan_deduction = student_loan
    result.ira_deduction = ira_ded
    result.self_emp_health_deduction = self_emp_hi
    result.sep_ira_deduction = sep_ira
    result.total_above_line_deductions = total_above

    # ------------------------------------------------------------------
    # Step 4: Social Security inclusion (IRC §86)
    # AGI-before-SS = ordinary income + SE income - above-line deductions
    # ------------------------------------------------------------------
    agi_before_ss = _cents(ordinary_income + se_income - total_above)
    taxable_ss = _taxable_social_security(facts, agi_before_ss)

    if facts.social_security_benefits > 0:
        steps.append(
            f"Social Security benefits: ${facts.social_security_benefits:,}. "
            f"AGI before SS = ${agi_before_ss:,.2f}. "
            f"Taxable SS benefits (IRC §86) = ${taxable_ss:,.2f}."
        )

    result.taxable_ss_benefits = taxable_ss

    # ------------------------------------------------------------------
    # Step 5: AGI
    # ------------------------------------------------------------------
    agi = _cents(agi_before_ss + taxable_ss)
    result.agi = agi
    result.gross_income = _cents(
        ordinary_income + se_income + facts.social_security_benefits
    )

    steps.append(f"AGI = ${agi:,.2f}.")

    # ------------------------------------------------------------------
    # Step 6: Itemized deductions (Schedule A)
    # ------------------------------------------------------------------
    # Medical: deductible portion exceeds 7.5% of AGI (IRC §213)
    medical_floor = _cents(agi * 0.075)
    medical_deductible = max(0.0, facts.medical_expenses - medical_floor)

    # SALT: capped at $10,000 (IRC §164(b)(6))
    salt = min(
        facts.state_income_tax + facts.real_estate_tax,
        SALT_CAP,
    )

    itemized = _cents(
        medical_deductible
        + salt
        + facts.mortgage_interest
        + facts.charitable_cash
        + facts.charitable_noncash
        + facts.casualty_losses
    )

    std = STANDARD_DEDUCTION[fs]
    uses_itemized = itemized > std
    deduction_used = itemized if uses_itemized else std

    deduction_label = "itemized" if uses_itemized else "standard"
    steps.append(
        f"Itemized deductions: SALT (capped) ${salt:,}, mortgage interest "
        f"${facts.mortgage_interest:,}, charitable cash ${facts.charitable_cash:,}, "
        f"medical (above 7.5% AGI floor) ${medical_deductible:,.2f}. "
        f"Total itemized = ${itemized:,.2f}. "
        f"Standard deduction ({fs}) = ${std:,}. "
        f"Using {deduction_label} deduction = ${deduction_used:,.2f}."
    )

    result.itemized_deductions = itemized
    result.standard_deduction = float(std)
    result.deduction_used = float(deduction_used)
    result.uses_itemized = uses_itemized

    # ------------------------------------------------------------------
    # Step 7: QBI deduction (IRC §199A) — basic case only
    # ------------------------------------------------------------------
    qbi_deduction = 0.0
    if se_income > 0:
        income_threshold = QBI_THRESHOLD[fs]
        if agi > income_threshold:
            # Above threshold: W-2 wage and UBIA limitations apply.
            # These are not modeled; flag for external validation.
            result.needs_external_validation = True
            steps.append(
                f"QBI: AGI ${agi:,.2f} exceeds §199A threshold ${income_threshold:,} "
                f"for {fs}. W-2 wage limitations apply but are not computed here. "
                f"Flag for external validation."
            )
        else:
            ti_before_qbi = max(0.0, agi - deduction_used)
            qbi = float(se_income)   # net QBI = net Schedule C income
            qbi_deduction = _cents(
                min(QBI_DEDUCTION_RATE * qbi, QBI_DEDUCTION_RATE * ti_before_qbi)
            )
            steps.append(
                f"QBI deduction (IRC §199A): 20% × min(QBI ${qbi:,.2f}, "
                f"taxable income before QBI ${ti_before_qbi:,.2f}) = "
                f"${qbi_deduction:,.2f}."
            )

    result.qbi_deduction = qbi_deduction

    # ------------------------------------------------------------------
    # Step 8: Taxable income
    # ------------------------------------------------------------------
    taxable_income = max(0.0, _cents(agi - deduction_used - qbi_deduction))
    result.taxable_income = taxable_income
    steps.append(f"Taxable income = ${taxable_income:,.2f}.")

    # ------------------------------------------------------------------
    # Step 9: Regular income tax on ordinary income
    # LTCG and qualified dividends are taxed at preferential rates (§1(h)).
    # ------------------------------------------------------------------
    ltcg_and_qd = max(0.0, facts.capital_gains_net + facts.qualified_dividends)
    ordinary_ti = max(0.0, taxable_income - ltcg_and_qd)
    result.ordinary_taxable_income = ordinary_ti

    regular_tax = _tax_from_brackets(ordinary_ti, fs)
    ltcg_tax = _ltcg_tax(min(ltcg_and_qd, taxable_income - ordinary_ti), ordinary_ti, fs)

    if ltcg_and_qd > 0:
        steps.append(
            f"Ordinary taxable income (excl. LTCG/QD) = ${ordinary_ti:,.2f}. "
            f"Regular tax from brackets = ${regular_tax:,.2f}. "
            f"LTCG + qualified dividends = ${ltcg_and_qd:,.2f}. "
            f"LTCG/QD preferential tax (IRC §1(h)) = ${ltcg_tax:,.2f}."
        )
    else:
        steps.append(
            f"Regular income tax from 2024 brackets = ${regular_tax:,.2f}."
        )

    result.regular_tax = regular_tax
    result.ltcg_tax = ltcg_tax

    # ------------------------------------------------------------------
    # Step 10: NIIT and Additional Medicare Tax
    # ------------------------------------------------------------------
    threshold = ADDL_MEDICARE_THRESHOLD[fs]

    # NIIT (IRC §1411): 3.8% on lesser of NII or MAGI excess above threshold.
    nii = _cents(
        facts.taxable_interest
        + facts.ordinary_dividends
        + facts.capital_gains_net
        + facts.rental_income_net
    )
    niit_base = max(0.0, min(nii, agi - threshold))
    niit = _cents(niit_base * NIIT_RATE)

    if niit > 0:
        steps.append(
            f"Net investment income (NII) = ${nii:,.2f}. "
            f"NIIT base (lesser of NII or AGI above ${threshold:,}) "
            f"= ${niit_base:,.2f}. NIIT at 3.8% (IRC §1411) = ${niit:,.2f}."
        )

    # Additional Medicare Tax (0.9%) on earned income above threshold.
    earned_income_for_medicare = _cents(facts.wages + facts.spouse_wages + se_base)
    addl_mc_base = max(0.0, earned_income_for_medicare - threshold)
    addl_mc = _cents(addl_mc_base * ADDL_MEDICARE_RATE)

    if addl_mc > 0:
        steps.append(
            f"Earned income for Additional Medicare Tax = ${earned_income_for_medicare:,.2f}. "
            f"Excess above ${threshold:,} = ${addl_mc_base:,.2f}. "
            f"Additional Medicare Tax at 0.9% = ${addl_mc:,.2f}."
        )

    result.niit = niit
    result.addl_medicare_tax = addl_mc

    # ------------------------------------------------------------------
    # Step 11: AMT note (not computed)
    # ------------------------------------------------------------------
    if (
        agi > 500_000
        or (facts.state_income_tax + facts.real_estate_tax) > 20_000
        or facts.business_income > 200_000
    ):
        result.needs_amt_review = True
        steps.append(
            "Note: This taxpayer may be subject to the Alternative Minimum Tax "
            "(IRC §55). AMT computation is out of scope for this engine; "
            "use PolicyEngine-US or Tax-Calculator for validation."
        )

    # Total before credits
    total_before_credits = _cents(regular_tax + ltcg_tax + se_tax + niit + addl_mc)
    result.total_tax_before_credits = total_before_credits

    # ------------------------------------------------------------------
    # Step 12: Child Tax Credit (IRC §24)
    # ------------------------------------------------------------------
    ctc = 0.0
    n_children = facts.num_qualifying_children
    if n_children > 0:
        full_ctc = n_children * CTC_PER_CHILD
        phaseout_threshold = CTC_PHASEOUT_THRESHOLD[fs]
        # Credit reduces by $50 per $1,000 (or fraction) above threshold
        excess = max(0.0, agi - phaseout_threshold)
        phaseout_units = math.ceil(excess / 1_000)
        ctc = max(0.0, full_ctc - phaseout_units * 50)
        # CTC cannot exceed total tax liability (non-refundable portion)
        ctc = min(ctc, total_before_credits)
        ctc = _cents(ctc)

        steps.append(
            f"Child Tax Credit: {n_children} qualifying child(ren) × ${CTC_PER_CHILD:,} "
            f"= ${full_ctc:,}. Phase-out at ${phaseout_threshold:,} AGI "
            f"(reduced by $50 per $1,000). CTC after phase-out = ${ctc:,.2f}."
        )

    result.child_tax_credit = ctc
    result.total_credits = ctc

    # ------------------------------------------------------------------
    # Step 13: Net tax liability
    # ------------------------------------------------------------------
    net_tax = max(0.0, _cents(total_before_credits - ctc))
    result.net_tax = net_tax
    result.effective_rate = (
        _cents(net_tax / agi * 100) if agi > 0 else 0.0
    )

    steps.append(
        f"Total tax before credits = ${total_before_credits:,.2f}. "
        f"Child Tax Credit = ${ctc:,.2f}. "
        f"Net federal tax liability = ${net_tax:,.2f} "
        f"(effective rate {result.effective_rate:.2f}%)."
    )

    return result
