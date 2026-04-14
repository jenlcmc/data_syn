"""Load dynamic wage calibration data from dataset files.

This module reads the local LCA and OEWS files under ``dataset/`` and
produces SOC-level overlays for profile generation.  The overlay is optional:
if files are missing or dependencies are unavailable, callers can safely
fallback to embedded defaults.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency fallback
    pd = None


_WAGE_LEVELS = ("I", "II", "III", "IV")
_LEVEL_QUANTILES: dict[str, tuple[float, float]] = {
    "I": (0.25, 0.75),
    "II": (0.25, 0.75),
    "III": (0.25, 0.75),
    "IV": (0.50, 0.90),
}
_LCA_FILE_RE = re.compile(r"FY(\d{4})_Q(\d)")
_SOC_RE = re.compile(r"(\d{2}-\d{4})")
_VALID_STATE_RE = re.compile(r"^[A-Z]{2}$")


@dataclass
class DynamicCalibration:
    """SOC-level calibration output from local dataset files."""

    wage_by_level: dict[str, dict[str, tuple[int, int]]]
    oews_percentiles: dict[str, tuple[int, int, int, int, int]]
    top_states_by_soc: dict[str, list[str]]
    source_files: dict[str, list[str]]



def _normalize_soc(value: object) -> str | None:
    text = str(value or "")
    match = _SOC_RE.search(text)
    if not match:
        return None
    return match.group(1)



def _normalize_level(value: object) -> str | None:
    text = str(value or "").upper()
    if "IV" in text or "LEVEL 4" in text or text.strip() == "4":
        return "IV"
    if "III" in text or "LEVEL 3" in text or text.strip() == "3":
        return "III"
    if "II" in text or "LEVEL 2" in text or text.strip() == "2":
        return "II"
    if "I" in text or "LEVEL 1" in text or text.strip() == "1":
        return "I"
    return None



def _to_int_wage(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "")
    if text in {"*", "**", "***", "#", "N/A", "NA"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number <= 0:
        return None
    return int(round(number))



def _quantile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * q
    low_idx = int(index)
    high_idx = min(low_idx + 1, len(sorted_values) - 1)
    frac = index - low_idx
    low = sorted_values[low_idx]
    high = sorted_values[high_idx]
    return int(round(low + (high - low) * frac))



def _rank_lca_file(path: Path) -> tuple[int, int, str]:
    match = _LCA_FILE_RE.search(path.name)
    if not match:
        return (0, 0, path.name)
    fiscal_year = int(match.group(1))
    quarter = int(match.group(2))
    return (fiscal_year, quarter, path.name)



def _latest_lca_files(dataset_dir: Path, max_files: int) -> list[Path]:
    files = sorted(
        dataset_dir.glob("LCA_Disclosure_Data_FY*.xlsx"),
        key=_rank_lca_file,
        reverse=True,
    )
    return files[:max_files]



def _read_lca_overlay(
    lca_files: list[Path],
    dataset_dir: Path,
    target_socs: set[str],
) -> tuple[dict[str, dict[str, tuple[int, int]]], dict[str, list[str]], list[str]]:
    wage_samples: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)

    used_files: list[str] = []

    if pd is None:
        return {}, {}, used_files

    required_cols = [
        "CASE_STATUS",
        "SOC_CODE",
        "PW_WAGE_LEVEL",
        "WAGE_RATE_OF_PAY_FROM",
        "WAGE_UNIT_OF_PAY",
        "WORKSITE_STATE",
    ]

    for file_path in lca_files:
        try:
            df = pd.read_excel(file_path, usecols=required_cols)
        except ValueError:
            # Column drift fallback: load full sheet then subset.
            df = pd.read_excel(file_path)
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                continue
            df = df[required_cols]

        used_files.append(str(file_path.relative_to(dataset_dir.parent)))

        status_mask = (
            df["CASE_STATUS"]
            .astype(str)
            .str.lower()
            .str.contains("certified", na=False)
        )
        unit_mask = (
            df["WAGE_UNIT_OF_PAY"]
            .astype(str)
            .str.lower()
            .str.contains("year|yr", na=False, regex=True)
        )
        subset = df.loc[status_mask & unit_mask].copy()
        if subset.empty:
            continue

        subset["soc_code"] = subset["SOC_CODE"].map(_normalize_soc)
        subset["wage_level"] = subset["PW_WAGE_LEVEL"].map(_normalize_level)
        subset["wage"] = (
            subset["WAGE_RATE_OF_PAY_FROM"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
        )
        subset["wage"] = pd.to_numeric(subset["wage"], errors="coerce")
        subset["state"] = subset["WORKSITE_STATE"].astype(str).str.upper().str[:2]

        subset = subset[
            subset["soc_code"].isin(target_socs)
            & subset["wage_level"].isin(_WAGE_LEVELS)
            & subset["wage"].notna()
            & (subset["wage"] > 0)
        ]

        if subset.empty:
            continue

        for row in subset[["soc_code", "wage_level", "wage", "state"]].itertuples(index=False):
            soc_code = row.soc_code
            level = row.wage_level
            wage = int(round(float(row.wage)))
            state = row.state

            wage_samples[soc_code][level].append(wage)
            if isinstance(state, str) and _VALID_STATE_RE.match(state):
                state_counts[soc_code][state] += 1

    wage_overlay: dict[str, dict[str, tuple[int, int]]] = {}
    for soc_code, by_level in wage_samples.items():
        level_ranges: dict[str, tuple[int, int]] = {}
        for level in _WAGE_LEVELS:
            values = sorted(by_level.get(level, []))
            if len(values) < 20:
                continue

            low_q, high_q = _LEVEL_QUANTILES[level]
            low = _quantile(values, low_q)
            high = _quantile(values, high_q)
            if high <= low:
                high = low + 1_000
            level_ranges[level] = (low, high)

        if level_ranges:
            wage_overlay[soc_code] = level_ranges

    state_overlay: dict[str, list[str]] = {}
    for soc_code, counter in state_counts.items():
        states = [s for s, _ in counter.most_common(8)]
        if len(states) >= 3:
            state_overlay[soc_code] = states

    return wage_overlay, state_overlay, used_files



def _read_oews_overlay(
    dataset_dir: Path,
    target_socs: set[str],
) -> tuple[dict[str, tuple[int, int, int, int, int]], list[str]]:
    if pd is None:
        return {}, []

    file_path = dataset_dir / "oesm24nat" / "national_M2024_dl.xlsx"
    if not file_path.exists():
        return {}, []

    usecols = [
        "AREA_TITLE",
        "OCC_CODE",
        "A_PCT10",
        "A_PCT25",
        "A_MEDIAN",
        "A_PCT75",
        "A_PCT90",
    ]

    try:
        df = pd.read_excel(file_path, usecols=usecols)
    except ValueError:
        df = pd.read_excel(file_path)
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            return {}, []
        df = df[usecols]

    # National rows only; some files include state or metro rows.
    df = df[df["AREA_TITLE"].astype(str).str.lower() == "national"].copy()
    if df.empty:
        return {}, []

    df["soc_code"] = df["OCC_CODE"].map(_normalize_soc)
    df = df[df["soc_code"].isin(target_socs)]

    oews_overlay: dict[str, tuple[int, int, int, int, int]] = {}
    for row in df[["soc_code", "A_PCT10", "A_PCT25", "A_MEDIAN", "A_PCT75", "A_PCT90"]].itertuples(index=False):
        p10 = _to_int_wage(row[1])
        p25 = _to_int_wage(row[2])
        p50 = _to_int_wage(row[3])
        p75 = _to_int_wage(row[4])
        p90 = _to_int_wage(row[5])

        if None in (p10, p25, p50, p75, p90):
            continue
        if not (p10 <= p25 <= p50 <= p75 <= p90):
            continue

        oews_overlay[row[0]] = (p10, p25, p50, p75, p90)

    return oews_overlay, [str(file_path.relative_to(dataset_dir.parent))]


def _cache_path(dataset_dir: Path) -> Path:
    return dataset_dir.parent / "data_syn" / "output" / "wage_calibration_cache.json"


def _fingerprint(files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_path in files:
        if not file_path.exists():
            continue
        stat = file_path.stat()
        rows.append(
            {
                "path": str(file_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rows


def _serialize_level_ranges(
    value: dict[str, dict[str, tuple[int, int]]],
) -> dict[str, dict[str, list[int]]]:
    payload: dict[str, dict[str, list[int]]] = {}
    for soc, by_level in value.items():
        payload[soc] = {}
        for level, bounds in by_level.items():
            payload[soc][level] = [int(bounds[0]), int(bounds[1])]
    return payload


def _deserialize_level_ranges(
    value: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, tuple[int, int]]]:
    payload: dict[str, dict[str, tuple[int, int]]] = {}
    for soc, by_level in value.items():
        payload[soc] = {}
        for level, bounds in by_level.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                continue
            payload[soc][level] = (int(bounds[0]), int(bounds[1]))
    return payload


def _serialize_oews(
    value: dict[str, tuple[int, int, int, int, int]],
) -> dict[str, list[int]]:
    return {soc: [int(x) for x in values] for soc, values in value.items()}


def _deserialize_oews(
    value: dict[str, list[int]],
) -> dict[str, tuple[int, int, int, int, int]]:
    payload: dict[str, tuple[int, int, int, int, int]] = {}
    for soc, values in value.items():
        if not isinstance(values, list) or len(values) != 5:
            continue
        payload[soc] = (
            int(values[0]),
            int(values[1]),
            int(values[2]),
            int(values[3]),
            int(values[4]),
        )
    return payload



def load_dynamic_calibration(
    dataset_dir: Path,
    target_socs: Iterable[str],
    max_lca_files: int = 2,
) -> DynamicCalibration:
    """Build SOC-level calibration overlays from local dataset files.

    Parameters
    ----------
    dataset_dir:
        Directory containing LCA and OEWS files.
    target_socs:
        SOC codes that should be calibrated.
    max_lca_files:
        Number of latest LCA quarterly files to scan.

    Returns
    -------
    DynamicCalibration
        Overlay values and source-file provenance. Empty overlays indicate that
        dynamic calibration data could not be loaded.
    """
    target_soc_set = {
        soc
        for soc in target_socs
        if isinstance(soc, str) and soc and not soc.startswith("00-")
    }

    if not dataset_dir.exists() or not target_soc_set or pd is None:
        return DynamicCalibration(
            wage_by_level={},
            oews_percentiles={},
            top_states_by_soc={},
            source_files={"lca": [], "oews": []},
        )

    lca_files = _latest_lca_files(dataset_dir, max_files=max_lca_files)
    oews_file = dataset_dir / "oesm24nat" / "national_M2024_dl.xlsx"
    cache_file = _cache_path(dataset_dir)

    input_fingerprint = {
        "max_lca_files": int(max_lca_files),
        "target_socs": sorted(target_soc_set),
        "lca": _fingerprint(lca_files),
        "oews": _fingerprint([oews_file]),
    }

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None

        if isinstance(cached, dict) and cached.get("fingerprint") == input_fingerprint:
            result = cached.get("result", {})
            return DynamicCalibration(
                wage_by_level=_deserialize_level_ranges(
                    result.get("wage_by_level", {})
                ),
                oews_percentiles=_deserialize_oews(
                    result.get("oews_percentiles", {})
                ),
                top_states_by_soc={
                    soc: [str(s) for s in states]
                    for soc, states in result.get("top_states_by_soc", {}).items()
                    if isinstance(states, list)
                },
                source_files={
                    "lca": [str(x) for x in result.get("source_files", {}).get("lca", [])],
                    "oews": [str(x) for x in result.get("source_files", {}).get("oews", [])],
                },
            )

    wage_by_level, top_states, lca_files = _read_lca_overlay(
        lca_files=lca_files,
        dataset_dir=dataset_dir,
        target_socs=target_soc_set,
    )
    oews_percentiles, oews_files = _read_oews_overlay(
        dataset_dir=dataset_dir,
        target_socs=target_soc_set,
    )

    result = DynamicCalibration(
        wage_by_level=wage_by_level,
        oews_percentiles=oews_percentiles,
        top_states_by_soc=top_states,
        source_files={"lca": lca_files, "oews": oews_files},
    )

    cache_payload = {
        "fingerprint": input_fingerprint,
        "result": {
            "wage_by_level": _serialize_level_ranges(result.wage_by_level),
            "oews_percentiles": _serialize_oews(result.oews_percentiles),
            "top_states_by_soc": result.top_states_by_soc,
            "source_files": result.source_files,
        },
    }

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(cache_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # Cache write failure should not block generation.
        pass

    return result
