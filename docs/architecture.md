# Architecture: Synthetic Tax Benchmark Pipeline

This document describes the current `data_syn/` architecture.

Primary goals:

- deterministic benchmark generation for Tax Year 2024,
- profile realism grounded in public wage datasets,
- strict release gates before publishing artifacts,
- external differential checks against Tax-Calculator for Tier 1 numeric cases.

Data inputs come from:

- `dataset/`:
  - United States Department of Labor Labor Condition Application Disclosure Data
        (Fiscal Years 2024-2026; Excel files).
  - United States Bureau of Labor Statistics Occupational Employment and Wage
        Statistics (May 2024 national release; Excel files).
- `knowledge/`:
  - Internal Revenue Service XML publications and instructions.
  - Title 26 of the United States Code (Internal Revenue Code) XML.

---

## 1. End-to-End Flow

```text
┌───────────────────────────────────────────────────────────────────────┐
│ INPUT SOURCES                                                         │
│                                                                       │
│ dataset/LCA_Disclosure_Data_FY*.xlsx   dataset/oesm24nat/*.xlsx      │
│ knowledge/*/*.xml + knowledge/usc26.xml                               │
└──────────────┬───────────────────────────────┬────────────────────────┘
               │                               │
               ▼                               ▼
┌────────────────────────────────────┐  ┌───────────────────────────────┐
│ sources/loader.py                  │  │ sources/miner.py              │
│                                    │  │                               │
│ - Reads LCA + OEWS files           │  │ - Parses IRS XML examples     │
│ - Builds SOC wage overlays         │  │ - Extracts scenario text      │
│ - Builds top-state pools           │  │ - Extracts conclusion lines   │
│ - Writes cache JSON                │  │ - Tags topics                 │
└──────────────┬─────────────────────┘  └──────────────┬────────────────┘
               │                                        │
               ▼                                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ sources/profiles.py                                                   │
│                                                                       │
│ - Applies dynamic wage/state overlays (fallback to embedded defaults) │
│ - Generates coherent TaxpayerFacts                                    │
│ - Builds dynamic scenario specs by occupation/domain                 │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ CASE BUILDERS + ENGINE                                                │
│                                                                       │
│ cases/tier1.py     cases/tier2.py     cases/tier3.py                 │
│ engine/ground_truth.py                                                │
│                                                                       │
│ - Tier 1 numeric ground truth from deterministic engine               │
│ - Tier 2 rule entailment labels from fact-derived logic              │
│ - Tier 3 Q&A/MCQ + IRS-mined scenarios                               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
                      data_syn/build_dataset.py
                                │
                                ▼
                   data_syn/output/benchmark.json
                                │
                                ▼
                      data_syn/scripts/release.py
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ RELEASE ARTIFACTS                                                     │
│                                                                       │
│ output/differential_taxcalc_report.json                               │
│ output/release_card.md                                                │
│ output/splits/{train,dev,test,manifest}.json                          │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Package Responsibilities

| Path | Responsibility |
|------|----------------|
| `data_syn/config.py` | Tax-year constants, thresholds, and feature toggles |
| `data_syn/schema.py` | `TaxpayerFacts`/`BenchmarkCase` dataclasses and JSON I/O |
| `data_syn/engine/ground_truth.py` | Deterministic 2024 tax computation (`compute`) |
| `data_syn/engine/taxcalc.py` | Tax-Calculator adapter (`compute_taxcalc`) |
| `data_syn/sources/loader.py` | Dynamic calibration loader for LCA + OEWS |
| `data_syn/sources/profiles.py` | Occupation profiles and `generate_lca_grounded_records` |
| `data_syn/sources/miner.py` | IRS XML worked-example mining (`mine_all_sources`) |
| `data_syn/cases/tier1.py` | Tier 1 numeric case construction |
| `data_syn/cases/tier2.py` | Tier 2 statutory entailment case construction |
| `data_syn/cases/tier3.py` | Tier 3 Q&A/MCQ/scenario case construction |
| `data_syn/build_dataset.py` | Full build orchestration and summary |
| `data_syn/scripts/validate.py` | Structural + logical dataset validation |
| `data_syn/scripts/harness.py` | Build and run locked IRS harness checks |
| `data_syn/scripts/checks.py` | Targeted tax checks + property invariants |
| `data_syn/scripts/differential.py` | Cross-engine differential validation |
| `data_syn/scripts/release.py` | End-to-end release gate orchestration |

---

## 3. Dynamic Profile Calibration

### 3.1 Loader (`sources/loader.py`)

`load_dynamic_calibration(...)` builds SOC-level overlays from local files in `dataset/`:

- LCA overlays by SOC and prevailing wage level (`I`..`IV`),
- OEWS annual wage percentiles (`p10,p25,p50,p75,p90`),
- top worksite-state pools by SOC.

LCA quantiles used for wage ranges:

- Levels `I`..`III`: `p25..p75`
- Level `IV`: `p50..p90`

### 3.2 Caching

Calibration is cached to:

- `data_syn/output/wage_calibration_cache.json`

Cache validity is keyed by:

- source file fingerprint (`size`, `mtime_ns`),
- target SOC set,
- `max_lca_files` parameter.

If files are missing or `pandas` is unavailable, generation falls back to embedded defaults.

### 3.3 Profile Integration (`sources/profiles.py`)

Profile generation merges three layers:

1. embedded deterministic occupation defaults,
2. optional dynamic overlays from `dataset/`,
3. scenario/state mix derived from active calibrated profiles.

This design keeps builds deterministic while reducing brittle hardcoded wage assumptions.

---

## 4. Case Generation and Ground Truth

### 4.1 Tier 1 (`cases/tier1.py`)

- Input records: LCA-grounded records + deterministic seed records.
- Engine: `engine/ground_truth.py::compute`.
- Output style: `numeric`.
- Typical quantities: AGI, taxable income, net tax, SE tax, child tax credit, deduction used.

Tier 1 cases carry full `reasoning_steps`, strict/lenient/pct tolerances, and statutory references.

### 4.2 Tier 2 (`cases/tier2.py`)

- Format: statutory entailment (`Yes`/`No`).
- Specs: `EntailmentSpec` list with callable `label` and `explanation`.
- Output style: `entailment` with statute-grounded reasoning.

### 4.3 Tier 3 (`cases/tier3.py`)

- Hand-crafted Q&A (`qa`) and MCQ (`mcq`) cases.
- IRS-mined scenario (`scenario`) cases via `sources/miner.py`.
- Output style: text-based references and citation-oriented scoring.

Tier 3 is always confidence tier `C`.

### 4.4 Ground-Truth Flow Before Differential Validation

For Tier 1 and Tier 2, ground truth is generated internally before any
external comparison:

1. Build coherent `TaxpayerFacts` records from calibrated profiles and seed records.
2. Run `engine/ground_truth.py::compute` for deterministic rule-based outputs.
3. Materialize benchmark fields:
    - `ground_truth`
    - `reasoning_steps`
    - `statutory_refs`
4. Assign initial confidence tiers:
    - Tier 1 and Tier 2 -> `B`
    - Tier 3 -> `C`

Tax-Calculator does not create initial ground truth. It is used later as a
cross-engine validator for sampled Tier 1 numeric cases.

---

## 5. Reliability and Release Gates

`data_syn/scripts/release.py` executes checks in sequence:

1. strict validation (`scripts/validate.py`),
2. IRS harness validation (`scripts/harness.py run`),
3. invariant checks (`scripts/checks.py`),
4. differential validation (`scripts/differential.py`),
5. release-card generation,
6. profile-safe split generation (`train/dev/test`).

Release passes only when all enabled gates pass.

### 5.1 What Strict Validation Actually Checks

`scripts/validate.py` performs both structural and logical checks.

Baseline checks include:

- ID uniqueness.
- Enum/value validity for `tier`, `style`, `domain`, `difficulty`, and `ground_truth_type`.
- Required fields (`question`, `ground_truth`).
- Style-level constraints:
  - numeric ground truth type/value
  - entailment label in `{Yes, No}`
  - MCQ choice keys and answer key consistency
- Cross-field `TaxpayerFacts` consistency, including:
  - dividends consistency
  - childcare vs qualifying children
  - self-employment deductions vs business income
  - filing-status spouse constraints
  - age sanity checks for retirement-income fields
  - IRA contribution vs earned compensation
- Confidence metadata consistency:
  - Tier 1 requires `profile_id`
  - confidence tier `A` requires `external_check_passed=True`
  - Tier 3 must remain confidence tier `C`

Strict mode adds:

- Tier 1 must include `reasoning_steps`.
- Tier 2 must include `statutory_refs` and non-empty `facts_narrative`.
- Tier 3 must include `statutory_refs`.

---

## 6. Differential Validation (Tax-Calculator)

`data_syn/scripts/differential.py` compares sampled Tier 1 cases against Tax-Calculator.

Comparable metrics:

- AGI,
- taxable income,
- deduction used,
- SE tax,
- child tax credit,
- net tax proxy (`iitax + setax`).

Sampling policy:

- mandatory: advanced + flagged cases,
- optional: configurable random sample of remaining Tier 1 cases.

Per-case metadata updated:

- `external_engine`,
- `external_delta_usd`,
- `external_check_passed`,
- `confidence_tier` promotion to `A` when profile checks agree.

Tax-Calculator API reference:

- [Tax-Calculator Public API](https://taxcalc.pslmodels.org/api/public_api.html)

---

## 7. IRS Locked Harness

Fixture:

- `data_syn/fixtures/irs_locked_examples.json`

Contains a deterministic snapshot of mined IRS worked examples using stable IDs
built from source + section + heading + normalized text prefix.

Commands:

- build/refresh fixture: `python data_syn/scripts/harness.py build --max-cases 40`
- validate fixture: `python data_syn/scripts/harness.py run`

Purpose:

- detect miner regressions or XML format drift,
- prevent silent quality regressions in Tier 3 scenario generation.

---

## 8. Release Artifacts

Primary artifacts emitted under `data_syn/output/`:

- `benchmark.json`
  - Full benchmark dataset with current confidence metadata.
- `differential_taxcalc_report.json`
  - Differential sampling config and aggregate agreement metrics.
  - Per-case comparison rows (engine value, external value, absolute delta, status).
- `release_card.md`
  - Human-readable release summary (source mix, confidence tiers, coverage,
        external agreement snapshot, and out-of-scope items).
- `splits/train.json`, `splits/dev.json`, `splits/test.json`
  - Profile-safe split payloads for model development/evaluation.
- `splits/manifest.json`
  - Seed and split statistics (profile counts and case counts).

Metric values vary by build inputs and random seed; treat artifacts as source of truth for a given run.

---

## 9. Extensibility Guidelines

For new coverage:

- Add or update `OccupationProfile` entries in `sources/profiles.py`.
- Keep source citations for wage ranges and percentiles.
- Add new validations in `scripts/validate.py` for any new facts fields.
- If miner logic changes, rebuild + re-run `scripts/harness.py` before release.
- Re-run `scripts/release.py` before publishing benchmark artifacts.

---

## 10. Known Boundaries

Still out of full scope in deterministic engine:

- AMT full computation,
- passive activity loss constraints,
- NOL carryovers,
- foreign tax credit,
- high-income QBI W-2/UBIA branch and SSTB phase-outs,
- EITC and Premium Tax Credit.

These remain flagged and are expected to be reviewed through external validation workflows.

---

## 11. When `benchmark.json` Should Change

Do not manually edit `benchmark.json` based on model evaluation outcomes.

If a benchmark run (for example, TaxCalcBench or TaxBench) reveals a genuine
ground-truth defect, the correct workflow is:

1. fix source code (engine rules, case specs, or profile generation),
2. rebuild (`build_dataset.py`),
3. re-run release gate (`scripts/release.py`).

This preserves reproducibility and provenance for every benchmark revision.
