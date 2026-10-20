import json
import os
import re

import pandas as pd

from core.audit import AuditLogger
from core.config import AUDIT_DIR, PROFILES_DIR
from core.llm import complete


_US_DATE_PATTERN = re.compile(r"\d{2}/\d{2}/\d{4}")
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _profile_column(column: pd.Series) -> dict:
    null_count = int(column.isna().sum())
    value_count = len(column) - null_count
    profile = {
        "dtype": str(column.dtype),
        "null_count": null_count,
        "null_pct": round(null_count / len(column) * 100, 2) if len(column) else 0.0,
        "unique_count": int(column.nunique(dropna=True)),
        "sample_values": [
            _json_value(value)
            for value in column.dropna().drop_duplicates().head(5).tolist()
        ],
        "quality_flags": [],
    }

    if pd.api.types.is_numeric_dtype(column.dtype) and not pd.api.types.is_bool_dtype(
        column.dtype
    ):
        profile.update(
            {
                "min": _json_value(column.min()),
                "max": _json_value(column.max()),
                "mean": _json_value(column.mean()),
            }
        )

    if pd.api.types.is_object_dtype(column.dtype) or pd.api.types.is_string_dtype(
        column.dtype
    ):
        non_null_values = column.dropna().map(str)
        if value_count:
            us_date_pct = non_null_values.str.fullmatch(_US_DATE_PATTERN).mean() * 100
            iso_date_pct = non_null_values.str.fullmatch(_ISO_DATE_PATTERN).mean() * 100
            if us_date_pct > 10 and iso_date_pct > 10:
                profile["quality_flags"].append("mixed_date_formats")

            if non_null_values.str.len().mean() < 6:
                profile["quality_flags"].append("possible_abbreviations")

    return profile


def _fallback_analysis() -> dict:
    return {
        "quality_summary": "Rule-based only",
        "critical_issues": [],
        "transformation_recommendations": {},
        "join_key_reasoning": "Candidate join keys are shared *_id columns across multiple tables.",
    }


def _call_llm(prompt: str) -> dict:
    content = complete(prompt, max_tokens=800)
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```").strip()
    analysis = json.loads(content)
    if (
        not isinstance(analysis, dict)
        or not isinstance(analysis.get("quality_summary"), str)
        or not isinstance(analysis.get("critical_issues"), list)
        or not isinstance(analysis.get("transformation_recommendations"), dict)
        or not isinstance(analysis.get("join_key_reasoning"), str)
    ):
        raise ValueError("The LLM response did not match the requested JSON schema")
    return {
        "quality_summary": analysis["quality_summary"],
        "critical_issues": analysis["critical_issues"],
        "transformation_recommendations": analysis[
            "transformation_recommendations"
        ],
        "join_key_reasoning": analysis["join_key_reasoning"],
    }


def _build_summary(tables: dict, candidate_join_keys: dict) -> str:
    summary = []
    for table_name, table_profile in tables.items():
        summary.append(
            f"{table_name}: "
            f"{json.dumps(table_profile, ensure_ascii=False)}"
        )
    summary.append(f"Candidate join keys: {json.dumps(candidate_join_keys)}")
    return "\n".join(summary)


def profile(file_paths: list[str], run_id: str) -> str:
    tables = {}
    join_key_tables = {}

    for file_path in file_paths:
        dataframe = pd.read_csv(file_path, low_memory=False)
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        columns = {
            column_name: _profile_column(dataframe[column_name])
            for column_name in dataframe.columns
        }
        tables[table_name] = {
            "row_count": int(len(dataframe)),
            "columns": columns,
        }

        for column_name in dataframe.columns:
            if column_name.endswith("_id"):
                table_names = join_key_tables.setdefault(column_name, [])
                if table_name not in table_names:
                    table_names.append(table_name)

    candidate_join_keys = {
        key: table_names
        for key, table_names in join_key_tables.items()
        if len(table_names) >= 2
    }
    summary = _build_summary(tables, candidate_join_keys)
    prompt = (
        "Analyze this CSV profiling summary. Return JSON with exactly these keys: "
        '"quality_summary" (string), "critical_issues" (list), '
        '"transformation_recommendations" (object keyed by table.column), and '
        '"join_key_reasoning" (string). Do not include markdown fences.\n\n'
        f"{summary}"
    )
    try:
        llm_analysis = _call_llm(prompt)
    except Exception:
        llm_analysis = _fallback_analysis()

    profile_data = {
        "run_id": run_id,
        "tables": tables,
        "candidate_join_keys": candidate_join_keys,
        "llm_analysis": llm_analysis,
    }

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROFILES_DIR / f"profile_combined_{run_id[:8]}.json"
    with output_path.open("w", encoding="utf-8") as profile_file:
        json.dump(profile_data, profile_file, indent=2, ensure_ascii=False)

    AuditLogger(run_id).log(
        agent="profiler",
        action="profile_completed",
        output_path=str(output_path),
        table_count=len(tables),
    )
    return str(output_path)