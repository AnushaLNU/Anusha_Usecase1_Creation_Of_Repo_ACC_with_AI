import json
import csv
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from core.config import STTM_DIR
from core.llm import complete


_REQUIRED_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]


def _call_llm(prompt: str) -> str:
    return complete(prompt, max_tokens=4096)


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _extract_csv(text: str) -> str:
    csv_match = re.search(
        r"```\s*csv\s*(.*?)(?:```|$)", text, flags=re.IGNORECASE | re.DOTALL
    )
    if csv_match:
        return csv_match.group(1).strip()
    return text.strip()


def _validate_and_save(csv_text: str, out_path: str | Path) -> str:
    try:
        rows = list(csv.reader(StringIO(csv_text), strict=True))
        if not rows:
            raise ValueError("CSV response is empty")

        headers = [str(header).strip().lstrip("\ufeff") for header in rows[0]]
        if any(column not in headers for column in _REQUIRED_COLUMNS):
            raise ValueError("CSV header does not contain all required columns")

        normalized_rows = []
        for row in rows[1:]:
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) > len(headers):
                row = row[: len(headers) - 1] + [
                    ",".join(row[len(headers) - 1 :])
                ]
            elif len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            normalized_rows.append(row)
        dataframe = pd.DataFrame(normalized_rows, columns=headers)
    except Exception as error:
        raise ValueError(f"LLM response is not valid CSV: {error}") from error

    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    missing_columns = [
        column for column in _REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "STTM CSV is missing required columns: " + ", ".join(missing_columns)
        )

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return str(output_path)


def _read_parquet_schemas(file_paths: list[str]) -> dict:
    schemas = {}
    for file_path in file_paths:
        dataframe = pd.read_parquet(file_path)
        table_name = Path(file_path).stem
        schemas[table_name] = {
            "columns": {
                column_name: {
                    "dtype": str(dataframe[column_name].dtype),
                    "sample_values": dataframe[column_name]
                    .dropna()
                    .drop_duplicates()
                    .head(5)
                    .tolist(),
                }
                for column_name in dataframe.columns
            }
        }
    return schemas


def _read_sttm(sttm_path: str) -> str:
    with open(sttm_path, "r", encoding="utf-8-sig") as sttm_file:
        return sttm_file.read()


def _write_sttm(prompt: str, run_id: str, phase: str) -> str:
    response = _call_llm(prompt)
    csv_text = _extract_csv(response)
    output_path = STTM_DIR / f"sttm_{phase}_{run_id[:8]}.csv"
    return _validate_and_save(csv_text, output_path)


def generate_bronze_sttm(
    profile_path: str, business_intent: str, run_id: str
) -> str:
    with open(profile_path, "r", encoding="utf-8") as profile_file:
        profile_data = json.load(profile_file)

    table_schemas = {}
    for table_name, table_profile in profile_data["tables"].items():
        table_schemas[table_name] = {
            "row_count": table_profile.get("row_count"),
            "columns": {
                column_name: {
                    "dtype": column_profile.get("dtype"),
                    "sample_values": column_profile.get("sample_values", []),
                    "quality_flags": column_profile.get("quality_flags", []),
                }
                for column_name, column_profile in table_profile["columns"].items()
            },
        }

    prompt = (
        "Generate a Bronze STTM as CSV, with exactly one header row containing "
        "these columns in this order: "
        + ", ".join(_REQUIRED_COLUMNS)
        + ". Quote any field containing commas or newlines. Do not include "
        "commentary, markdown tables, or extra columns. A closing code-fence "
        "marker may be omitted.\nCreate one type_cast rule for every source column, inferring the "
        "target type from its dtype and quality flags. Also create metadata_inject "
        "rules for _load_timestamp with logic utc_now and _source_file with logic "
        "source_path for every table. Keep source and target table names aligned; "
        "use bronze as target_schema.\n"
        f"Business intent: {business_intent}\n"
        f"Table schemas, samples, and quality flags: {json.dumps(table_schemas)}"
    )
    return _write_sttm(prompt, run_id, "bronze")


def generate_silver_sttm(
    bronze_paths: list[str],
    bronze_sttm_path: str,
    business_intent: str,
    run_id: str,
) -> str:
    schemas = _read_parquet_schemas(bronze_paths)
    bronze_sttm = _read_sttm(bronze_sttm_path)
    prompt = (
        "Generate a Silver STTM as CSV only, using exactly these columns: "
        + ", ".join(_REQUIRED_COLUMNS)
        + ". Quote any field containing commas or newlines, especially "
        "transformation_logic. Do not add commentary or extra columns.\n"
        "Use applicable rules from null_handler, standardize_date, "
        "normalize_category, normalize_segment, dedup, and surrogate_key. "
        "Use the schemas and samples to select rules and transformation logic; "
        "do not invent source columns.\n"
        f"Business intent: {business_intent}\n"
        f"Bronze parquet schemas and samples: {json.dumps(schemas, default=_json_default)}\n"
        f"Prior Bronze STTM:\n{bronze_sttm}"
    )
    return _write_sttm(prompt, run_id, "silver")


def generate_gold_sttm(
    silver_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
) -> str:
    schemas = _read_parquet_schemas(silver_paths)
    silver_sttm = _read_sttm(silver_sttm_path)
    prompt = (
        "Generate a Gold STTM as CSV only, using exactly these columns: "
        + ", ".join(_REQUIRED_COLUMNS)
        + ". Quote any field containing commas or newlines, especially "
        "transformation_logic. Do not add commentary or extra columns.\n"
        "Design transformations that answer the business intent. Include "
        "join rules with transformation_logic formatted exactly as "
        "join_left:table_a:table_b:key, group_by rules for grouping columns, "
        "and aggregate rules using SUM or COUNT where appropriate. Use only "
        "columns present in the supplied schemas.\n"
        f"Business intent: {business_intent}\n"
        f"Silver parquet schemas and samples: {json.dumps(schemas, default=_json_default)}\n"
        f"Prior Silver STTM:\n{silver_sttm}"
    )
    return _write_sttm(prompt, run_id, "gold")