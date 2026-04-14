# Synthetic Tax Benchmark (`data_syn`)

A pipeline for generating a structured, ground-truth-verified benchmark
dataset for evaluating LLM performance on US federal income tax and
self-employment tax assistance.

**Tax Year:** 2024
**Scope:** Individual resident filers (Form 1040)
**Current dataset size:** 491 cases across 3 tiers (rebuildable)

---

## Table of contents

1. [What this benchmark is](#what-this-benchmark-is)
2. [Dataset structure](#dataset-structure)
3. [Step-by-step: run from scratch](#step-by-step-run-from-scratch)
4. [Ground truth methodology](#ground-truth-methodology)
5. [Occupation coverage](#occupation-coverage)
6. [Data sources](#data-sources)
7. [Scoring API](#scoring-api)
8. [Adding new cases](#adding-new-cases)

---

## What this benchmark is

Existing tax LLM evaluations have two problems:

1. **Unreliable ground truth.** Using an LLM to generate expected answers
   does not work for tax — models share the same failure modes (bracket
   miscalculation, phase-out errors, SE tax base confusion). A majority vote
   among wrong models is still wrong.

2. **Unrealistic profiles.** Independently randomized fields (like
   `self_employed: false` with `business_income: $5,249`) make cases
   logically incoherent and too easy for models to spot.

This pipeline fixes both:

- **Ground truth** for numeric cases comes from a deterministic Python
  engine that encodes 2024 IRC rules directly — no LLM involved.
- **Taxpayer profiles** are grounded in DOL H-1B LCA filings and BLS OEWS
  wage percentiles. All deductions and benefits are derived from those
  wages by rule (401k deferral reduces Box 1, SE health insurance is
  bounded by SE income, property tax comes from state medians, etc.).
- **Occupations** span the full wage spectrum: software engineers, nurses,
  lawyers, truck drivers, retail workers, gig drivers, plumbers, CNAs,
  real estate agents, and more.

---

## Dataset structure

### Tier 1 — Numeric (majority of cases)

The LLM receives a taxpayer profile in plain English and must compute a
specific dollar amount: AGI, taxable income, net federal tax, SE tax,
Child Tax Credit, or total deduction used.

```json
{
  "id": "t1_numeric_0001",
  "tier": 1,
  "style": "numeric",
  "domain": "federal_income_tax",
  "difficulty": "basic",
  "facts_narrative": "The taxpayer is filing as single for Tax Year 2024. Wages and salaries: $75,523...",
  "question": "What is this taxpayer's adjusted gross income (AGI) for Tax Year 2024?",
  "ground_truth": 61247.50,
  "ground_truth_type": "numeric_usd",
  "tolerance_strict_usd": 0.0,
  "tolerance_lenient_usd": 5.0,
  "tolerance_pct": 0.10,
  "reasoning_steps": ["Ordinary income sources: wages $75,523...", "AGI = $61,247.50."],
  "statutory_refs": ["26 USC §62"],
  "source": "lca_oews",
  "verified_by": "python_engine"
}
```

**Sources:** LCA-grounded profiles + hand-crafted seed profiles.
Each profile generates 3–6 cases (one per applicable quantity).

### Tier 2 — Statutory Entailment (~32 cases)

The LLM receives a taxpayer situation and a Yes/No question about whether
a specific IRC rule applies. Modeled after
[LegalBench SARA](https://github.com/HazyResearch/legalbench).

```json
{
  "id": "t2_entail_0003",
  "tier": 2,
  "style": "entailment",
  "question": "Under IRC §164(b)(6), does the $10,000 SALT cap limit this taxpayer's deduction?",
  "ground_truth": "Yes",
  "ground_truth_type": "boolean_str",
  "statutory_refs": ["26 USC §164(b)(6)"]
}
```

**IRC sections covered:** §1(h), §24, §63, §86, §164, §199A, §219,
§1401, §1411.

### Tier 3 — Q&A, MCQ, Scenario (~63 cases)

- **Q&A (15):** Open-ended tax knowledge questions with reference answers
  verified against IRS publications.
- **MCQ (8):** Four-choice questions testing conceptual understanding.
- **Scenario (40):** Worked examples mined directly from IRS XML
  publications (Pub 17, Pub 560, Sch C/SE instructions, etc.).

---

## Step-by-step: run from scratch

Follow these steps in order. Each step builds on the previous one.

### Step 0: Prerequisites

You need Python 3.10+ and conda (or any virtual environment manager).

```bash
# Check Python version
python --version          # must be 3.10 or higher

# Check conda
conda --version
```

You also need the project's conda environment. If you have not created it:

```bash
conda create -n cs789_research python=3.11 -y
conda activate cs789_research
pip install -r requirements.txt   # from the project root

# Required for external differential validation
conda install -c conda-forge -y taxcalc
```

### Step 1: Activate the environment

```bash
conda activate cs789_research
```

All subsequent commands assume this environment is active.

### Step 2: Verify the data directory

The pipeline now supports dynamic calibration directly from local data files:

- `dataset/LCA_Disclosure_Data_FY*.xlsx` (SOC-level wage ranges + top states)
- `dataset/oesm24nat/national_M2024_dl.xlsx` (OEWS percentiles)

If these files are present, `sources/profiles.py` overlays embedded
defaults with file-derived ranges and writes a cache to
`data_syn/output/wage_calibration_cache.json`.

If files are missing, generation safely falls back to embedded defaults.

```bash
# Optional: check what data files are present
ls dataset/
```

### Step 3: Dry-run to validate the pipeline

Run the pipeline without writing any output. This checks that all imports,
constants, and logic are working correctly.

```bash
python data_syn/build_dataset.py --dry-run
```

Expected output (last lines):

```text
Building Tier 1 (numeric) cases... <n> cases.
Building Tier 2 (entailment) cases... <n> cases.
Building Tier 3 (Q&A / MCQ / scenario) cases... <n> cases.
Validating... OK.
[dry-run] Output not written.
```

If you see import errors, check that your conda environment is active and
that you are running from the project root (not from inside `data_syn/`).

### Step 4: Build the full dataset

On a cold run, this can take a few minutes because wage calibration files are
scanned and cached. Warm runs are usually much faster.

`build_dataset.py` does not run external differential checks. Those run in the
release gate (`scripts/release.py`) in Step 6.

```bash
python data_syn/build_dataset.py
```

This writes the benchmark to `data_syn/output/benchmark.json`.

Runtime notes:

- First cold run with dynamic LCA calibration can take a few minutes.
- Warm runs use cache and are typically under a few seconds.

To build only specific tiers:

```bash
python data_syn/build_dataset.py --tiers 1 2
python data_syn/build_dataset.py --tiers 3
```

To write to a custom location:

```bash
python data_syn/build_dataset.py --output /path/to/my_benchmark.json
```

### Step 5: Validate the output

```bash
# Basic validation (checks IDs, types, enum values)
python data_syn/scripts/validate.py data_syn/output/benchmark.json

# Strict validation (also requires reasoning_steps and statutory_refs)
python data_syn/scripts/validate.py data_syn/output/benchmark.json --strict
```

What basic validation checks:

- Unique case IDs (no duplicates).
- Enum integrity for `tier`, `style`, `domain`, `difficulty`, `ground_truth_type`.
- Required fields are present (`question`, `ground_truth`).
- Style-specific rules:
  - `numeric` cases must have numeric `ground_truth` and `numeric_usd` type.
  - `entailment` cases must have `ground_truth` in `{Yes, No}`.
  - `mcq` cases must have choices `A/B/C/D` and answer key in `A/B/C/D`.
- Taxpayer fact consistency checks, including:
  - `qualified_dividends <= ordinary_dividends`
  - child care expense requires qualifying children
  - self-employed health insurance / SEP deductions require business income
  - spouse fields must match filing status constraints
  - retirement income age sanity checks
  - IRA contribution cannot exceed earned compensation
- Confidence metadata checks:
  - Tier 1 requires non-empty `profile_id`
  - confidence tier `A` requires `external_check_passed=True`
  - Tier 3 must use confidence tier `C`

What `--strict` adds:

- Tier 1 must include `reasoning_steps`.
- Tier 2 must include `statutory_refs` and non-empty `facts_narrative`.
- Tier 3 must include `statutory_refs`.

How to read validator output:

- The script prints distribution stats (tier/style/difficulty/domain).
- On failure, it prints each violated rule and exits non-zero.
- Fix all failures before running release or evaluation.

### Step 6: Run release-quality reliability gates

Run this after each dataset rebuild. It executes strict validation, IRS
worked-example harness checks, property invariants, Tax-Calculator
differential validation, release-card generation, and profile-safe splits.

```bash
python data_syn/scripts/release.py \
  --dataset data_syn/output/benchmark.json \
  --harness data_syn/fixtures/irs_locked_examples.json \
  --differential-report data_syn/output/differential_taxcalc_report.json \
  --release-card data_syn/output/release_card.md \
  --splits-dir data_syn/output/splits \
  --threshold-usd 25 \
  --random-rate 0.20 \
  --seed 42 \
  --version v1
```

Key outputs:

- `data_syn/output/differential_taxcalc_report.json`
- `data_syn/output/release_card.md`
- `data_syn/output/splits/{train.json,dev.json,test.json,manifest.json}`

What each output file contains:

- `data_syn/output/differential_taxcalc_report.json`
  - Sampling metadata (`seed`, `random_rate`, `threshold_usd`, mandatory vs random sample sizes).
  - Aggregate results (`comparable_cases`, `passed`, `failed`, `skipped`, agreement rates).
  - Confidence tier counts after differential validation.
  - Per-case comparison rows (`engine_value`, `external_value`, `abs_diff`, pass/fail).
- `data_syn/output/release_card.md`
  - Human-readable release summary: source mix, confidence tiers, filing-status coverage,
    income-band coverage, top topic tags, external agreement snapshot, and out-of-scope areas.
- `data_syn/output/splits/train.json`, `dev.json`, `test.json`
  - Final benchmark cases for each split.
  - Split assignment is profile-safe: all cases from the same `profile_id` stay in one split.
- `data_syn/output/splits/manifest.json`
  - Split metadata: seed used, profile counts per split, and case counts per split.

What is an IRS locked example?

- `data_syn/fixtures/irs_locked_examples.json` is a deterministic snapshot
  of mined IRS worked examples (source, section, conclusion, and text
  signature).
- `scripts/harness.py run` re-mines the current `knowledge/` XML and verifies
  those locked signatures still match.
- This prevents silent miner drift from changing Tier 3 scenario quality.

If you need to refresh the locked IRS harness fixture:

```bash
python data_syn/scripts/harness.py build --max-cases 40
python data_syn/scripts/harness.py run --harness data_syn/fixtures/irs_locked_examples.json
```

Tax-Calculator references:

- Public API docs: [Tax-Calculator Public API](https://taxcalc.pslmodels.org/api/public_api.html)

### Step 7: Preview cases (optional, for inspection)

```bash
# First 5 Tier 1 cases with reasoning steps shown
python data_syn/scripts/preview.py data_syn/output/benchmark.json \
  --tier 1 --n 5 --reasoning

# All entailment cases
python data_syn/scripts/preview.py data_syn/output/benchmark.json \
  --style entailment

# Cases involving self-employment tax
python data_syn/scripts/preview.py data_syn/output/benchmark.json \
  --tag se_tax

# Cases for gig workers or retail workers specifically
python data_syn/scripts/preview.py data_syn/output/benchmark.json \
  --tag wage_income --difficulty basic --n 10

# A single case by ID
python data_syn/scripts/preview.py data_syn/output/benchmark.json \
  --id t1_numeric_0007
```

### Step 8: Use the scoring API

To evaluate an LLM's response against a benchmark case:

```python
from data_syn.scoring.scorer import score_numeric, score_entailment, score_mcq

# Tier 1: numeric
result = score_numeric("t1_numeric_0001", prediction="$61,200", ground_truth=61247.50)
print(result.correct_strict)   # False (not exact)
print(result.correct_lenient)  # True (within $5)
print(result.correct_pct)      # True (within 10%)

# Tier 2: entailment
result = score_entailment("t2_entail_0003", prediction="Yes, the SALT cap applies.", ground_truth="Yes")
print(result.correct_strict)   # True

# Tier 3: MCQ
result = score_mcq("t3_mcq_0002", prediction="The answer is B.", ground_truth="B")
print(result.correct_strict)   # True
```

For Tier 3 Q&A and scenario cases, use the LLM judge workflow:

```python
from data_syn.scoring.scorer import judge_prompt, score_text

# Format a judge prompt to send to an LLM
prompt = judge_prompt(ground_truth, model_response)
# ... send prompt to your LLM judge, get back a JSON response ...

# Score with the judge's JSON response
result = score_text(
    case_id=case_id,
    prediction=model_response,
    ground_truth=ground_truth,
    judge_response=judge_json_response,
)
print(result.prediction_parsed)   # Judge total score (0–12)
print(result.correct_strict)      # True when judge marks pass=true
```

### Step 9: Run one-command LLM evaluation on `benchmark.json`

This is the direct `data_syn` benchmark flow:

1. Load cases from `data_syn/output/benchmark.json`.
2. Prompt the model with case facts + question.
3. Score predictions against each case `ground_truth`.
4. Save per-case outputs and aggregate metrics to a JSON report.

```bash
# Claude on all cases (real API calls)
python data_syn/scripts/eval_llm.py \
  --dataset data_syn/output/benchmark.json \
  --model claude \
  --output data_syn/output/llm_eval_results_claude.json

# Gemini on a small subset
python data_syn/scripts/eval_llm.py \
  --dataset data_syn/output/benchmark.json \
  --model gemini \
  --limit 25 \
  --output data_syn/output/llm_eval_results_gemini_limit25.json

# Fast smoke test (no API calls)
python data_syn/scripts/eval_llm.py --dry-run --limit 10
```

### Step 10: Run external benchmark evaluation (main RAG pipeline)

The command below runs the shared evaluator in `evaluation/run_eval.py`.
It evaluates registered datasets (TaxCalcBench, TaxBench, IRS QA pairs,
etc.), not `data_syn/output/benchmark.json` directly.

For `data_syn/output/benchmark.json`, run your model over each case and
score predictions with the Step 8 scoring API (`score_numeric`,
`score_entailment`, `score_mcq`, `score_text`).

From the project root:

```bash
# Evaluate Claude (hybrid RAG mode) on TaxCalcBench
python evaluation/run_eval.py \
  --dataset taxcalcbench \
  --model claude \
  --mode hybrid

# Dry run (pipeline check, no API calls)
python evaluation/run_eval.py --dry-run

# Score and analyze saved results
python evaluation/score_results.py
python evaluation/analyze_results.py --dataset taxcalcbench
```

Results are saved to `evaluation/results/<dataset>__<model>__<mode>.json`.

---

## Ground truth methodology

|Tier|Style|Ground truth source|Confidence tier|
|----|-----|-------------------|---------------|
|1|Numeric|Deterministic Python engine + sampled Tax-Calculator differential checks|A or B|
|2|Entailment|Engine output + statute text|B|
|3|Q&A / MCQ / scenario|IRS publication text or hand-crafted references|C|

### How ground truth is computed before external checks

For Tier 1 and Tier 2, the ground truth is produced internally first:

1. Generate coherent `TaxpayerFacts` from calibrated profiles and deterministic seed records.
2. Run the deterministic engine in `engine/ground_truth.py`.
3. Populate `ground_truth`, `reasoning_steps`, and `statutory_refs` from engine/statute logic.
4. Set initial confidence tiers (`B` for Tier 1/2, `C` for Tier 3).

Only after that, release-time differential validation compares sampled Tier 1
cases with Tax-Calculator and promotes agreeing profiles to confidence tier `A`.
Tax-Calculator is a verifier, not the primary source that creates initial ground truth.

Confidence tier semantics:

- `A`: deterministic engine output with external-engine agreement.
- `B`: deterministic engine output without external promotion.
- `C`: text/scenario judged cases (non-numeric generation targets).

**Cases flagged for external validation:** Cases where AMT or QBI W-2
wage limitations may apply are included in the benchmark but are marked
`needs_amt_review=True` or `needs_external_validation=True`. The release
gate cross-checks a mandatory sample of these against Tax-Calculator.

---

## Occupation coverage

The dataset covers 27 distinct occupation profiles spanning the full US
wage spectrum. All wages come from DOL LCA FY2024 filings or BLS OEWS
May 2024 national data.

| Domain | Occupations |
| ------ | ----------- |
| Technology | Software Developer, Systems Analyst, Data Scientist, IT Manager, IT Consultant (SE) |
| Finance | Accountant / Auditor, Financial Analyst |
| Healthcare | Registered Nurse, Pharmacist, Dental Hygienist, Nursing Assistant / Home Health Aide |
| Legal | Lawyer |
| Education | Elementary Teacher, College Lecturer |
| Skilled Trades | Construction Laborer, Maintenance Worker, Electrician, Plumber / Pipefitter |
| Transportation | Heavy Truck Driver, Rideshare / Delivery Driver (Gig) |
| Retail / Food Service | Cashier / Retail Sales Worker, Cook / Food Preparation Worker |
| Logistics | Warehouse / Material Handler |
| Office / Admin | Administrative Assistant, Customer Service Representative |
| Sales | Real Estate Agent / Broker (SE) |
| Creative (SE) | Freelance Designer / Photographer / Writer |

Filing situations covered: single, married filing jointly (MFJ),
married filing separately (MFS), head of household (HOH), and
qualifying surviving spouse (QSS). Income levels range from ~$20k (part-time
cashier) to ~$500k (senior lawyer or staff software engineer).

---

## Data sources

### Taxpayer profile inputs

|Source|What it provides|Used for|
|------|---------------|--------|
|United States Department of Labor Labor Condition Application (LCA) Disclosure Data, Fiscal Years 2024-2026|Legally sworn employer wage certifications by SOC code, wage level, and worksite state|Dynamic SOC wage range overlays and state-pool calibration for Tier 1 profiles|
|United States Bureau of Labor Statistics Occupational Employment and Wage Statistics (OEWS), May 2024 national release|P10/P25/P50/P75/P90 annual wages by SOC code|Dynamic percentile overlays and fallback for occupations underrepresented in LCA|
|United States Census Bureau American Community Survey (ACS) homeownership statistics|Homeownership probability by income and age|Determines itemized deduction eligibility|
|Internal Revenue Service Statistics of Income (SOI) aggregate statistics|Itemize rates and deduction behavior by AGI band|Calibrates charitable and SALT amounts|

### Ground truth inputs

|Source|What it provides|Used for|
|------|---------------|--------|
|Internal Revenue Service Revenue Procedure 2023-34|2024 inflation-adjusted tax parameters|All constants in `config.py`|
|Title 26 of the United States Code (Internal Revenue Code), including sections 1, 24, 62, 63, 86, 164, 199A, 219, 1401, and 1411|Statutory rules|Tax engine logic|
|Internal Revenue Service XML publications in `knowledge/`|Worked examples with stated conclusions|Tier 3 scenario cases|

### What is NOT used

- `comp_0/1/2.json` — independently randomized fields, logically inconsistent
- levels.fyi / Glassdoor — unverified, ToS restrictions on scraping
- Any LLM for computing ground truth (Tier 1 or Tier 2)

### When should `benchmark.json` be updated?

- Do not manually edit `benchmark.json` based on model outputs or leaderboard results.
- If TaxCalcBench/TaxBench or manual review reveals a true ground-truth issue,
  fix the source logic first (engine rules, case specs, or profile generation).
- Then regenerate and re-validate:
  - `python data_syn/build_dataset.py`
  - `python data_syn/scripts/release.py`

This keeps benchmark updates reproducible and auditable.

---

## Scoring API

### Numeric (Tier 1)

```python
from data_syn.scoring.scorer import score_numeric

result = score_numeric(case_id, prediction="$61,200", ground_truth=61247.50)

result.correct_strict    # bool: prediction == ground_truth exactly
result.correct_lenient   # bool: |prediction - ground_truth| <= $5
result.correct_pct       # bool: |prediction - ground_truth| / ground_truth <= 10%
result.extracted_value   # float: dollar amount extracted from prediction text
result.error_usd         # float: |prediction - ground_truth|
```

### Entailment (Tier 2)

```python
from data_syn.scoring.scorer import score_entailment

result = score_entailment(case_id, prediction="Yes, the cap applies.", ground_truth="Yes")

result.correct_strict    # bool: first Yes/No word matches ground_truth
result.extracted_label   # str: "Yes" | "No" | None
```

### MCQ (Tier 3)

```python
from data_syn.scoring.scorer import score_mcq

result = score_mcq(case_id, prediction="The answer is B.", ground_truth="B")

result.correct_strict    # bool: extracted letter matches ground_truth
result.extracted_choice  # str: "A" | "B" | "C" | "D" | None
```

### Q&A and scenario (Tier 3)

```python
from data_syn.scoring.scorer import judge_prompt, score_text, aggregate, format_report

# 1. Format a judge prompt
prompt = judge_prompt(case_id, question, ground_truth, model_response)

# 2. Send prompt to an LLM judge and get back JSON:
#    {"accuracy": 0-3, "completeness": 0-3, "reasoning": 0-3, "citation": 0-3}

# 3. Score
result = score_text(case_id, judge_json_response, ground_truth)
result.judge_score       # int: 0–12

# 4. Aggregate across many results
summary = aggregate(results)
print(format_report(summary))
```

---

## Adding new cases

See `data_syn/CLAUDE.md` for step-by-step instructions on adding
occupations, entailment specs, Q&A specs, and MCQ questions.

### Quick reference

**New Tier 1 occupation:** Add an `OccupationProfile` to
`sources/profiles.py` (cite LCA/OEWS sources for wage ranges).
Dynamic scenario generation will automatically include it. If needed,
adjust `_DOMAIN_STATE_POOLS` for better geographic coverage, then rebuild.

**New Tier 2 entailment question:** Add an `EntailmentSpec` to
`_build_specs()` in `cases/tier2.py`. The `label` field must
be a callable lambda, not a hardcoded string.

**New Tier 3 Q&A:** Add a dict to `_QA_SPECS` in `cases/tier3.py`.
The `ground_truth` must be verifiable against an IRS publication or IRC
section (cite in `statutory_refs`).

After any change, rebuild and re-validate:

```bash
python data_syn/build_dataset.py
python data_syn/scripts/validate.py data_syn/output/benchmark.json --strict
```
