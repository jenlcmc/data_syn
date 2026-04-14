"""Tax-Calculator adapter for differential validation.

This module maps ``TaxpayerFacts`` into a minimal Tax-Calculator input record
and returns comparable outputs for cross-engine agreement checks.

The adapter intentionally focuses on core comparable quantities:
  - AGI
  - Taxable income
  - SE tax
  - Deduction used (max of standard vs itemized)
  - Net tax proxy (iitax + setax)

The net tax proxy is used because this benchmark's ``net_tax`` includes
self-employment tax, while Tax-Calculator reports SE tax separately as
``setax``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from taxcalc import Calculator, Policy, Records

from schema import TaxpayerFacts


_MARS_MAP: dict[str, int] = {
    "single": 1,
    "mfj": 2,
    "mfs": 3,
    "hoh": 4,
    "qss": 5,
}


@dataclass
class TaxCalcResult:
    """Comparable outputs from Tax-Calculator for one taxpayer record."""

    agi: float
    taxable_income: float
    se_tax: float
    standard_deduction: float
    itemized_deductions: float
    deduction_used: float
    child_tax_credit: float
    iitax: float
    net_tax_proxy: float



def _to_taxcalc_row(facts: TaxpayerFacts) -> dict[str, float | int]:
    """Map TaxpayerFacts to a one-row Tax-Calculator Records payload."""
    filing_status = facts.filing_status
    mars = _MARS_MAP[filing_status]

    wages_total = float(facts.wages + facts.spouse_wages)
    se_income = float(facts.business_income)

    charitable_total = float(facts.charitable_cash + facts.charitable_noncash)
    pension_income = float(facts.pension_income)

    # Tax-Calculator expects both split and total values for some variables.
    row: dict[str, float | int] = {
        "RECID": 1,
        "MARS": mars,
        "XTOT": int(1 + (1 if filing_status == "mfj" else 0) + facts.num_qualifying_children),
        "nu18": int(facts.num_qualifying_children),
        "n24": int(facts.num_qualifying_children),
        "e00200p": float(facts.wages),
        "e00200s": float(facts.spouse_wages),
        "e00200": wages_total,
        "e00300": float(facts.taxable_interest),
        "e00600": float(facts.ordinary_dividends),
        "e00650": float(min(facts.qualified_dividends, facts.ordinary_dividends)),
        "p22250": 0.0,
        "p23250": float(facts.capital_gains_net),
        "e00900p": se_income,
        "e00900s": 0.0,
        "e00900": se_income,
        "e01400": float(facts.ira_distributions),
        # Tax-Calculator requires e01500 (total pension/annuity income)
        # to be greater than or equal to e01700 (taxable pension amount).
        "e01500": pension_income,
        "e01700": pension_income,
        "e02400": float(facts.social_security_benefits),
        "e17500": float(facts.medical_expenses),
        "e18400": float(facts.state_income_tax),
        "e18500": float(facts.real_estate_tax),
        "e19200": float(facts.mortgage_interest),
        "e19800": charitable_total,
        "e32800": float(facts.child_care_expenses),
    }
    return row



def compute_taxcalc(facts: TaxpayerFacts, tax_year: int = 2024) -> TaxCalcResult:
    """Run Tax-Calculator for one taxpayer and return comparable outputs."""
    row = _to_taxcalc_row(facts)
    records_df = pd.DataFrame([row])

    calculator = Calculator(
        policy=Policy(),
        records=Records(data=records_df, start_year=tax_year),
    )
    calculator.advance_to_year(tax_year)
    calculator.calc_all()

    agi = float(calculator.array("c00100")[0])
    taxable_income = float(calculator.array("c04800")[0])
    se_tax = float(calculator.array("setax")[0])
    standard = float(calculator.array("standard")[0])
    itemized = float(calculator.array("c04470")[0])
    deduction_used = max(standard, itemized)
    # c07220 is the nonrefundable child tax credit component used on Form 1040.
    child_tax_credit = float(calculator.array("c07220")[0])
    iitax = float(calculator.array("iitax")[0])
    net_tax_proxy = iitax + se_tax

    return TaxCalcResult(
        agi=agi,
        taxable_income=taxable_income,
        se_tax=se_tax,
        standard_deduction=standard,
        itemized_deductions=itemized,
        deduction_used=deduction_used,
        child_tax_credit=child_tax_credit,
        iitax=iitax,
        net_tax_proxy=net_tax_proxy,
    )
