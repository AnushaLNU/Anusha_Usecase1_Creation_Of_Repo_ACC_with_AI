import os
import re
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from core.audit import AuditLogger
from core.config import AUDIT_DIR, BRONZE_DIR
from core.llm import complete


def _call_llm(prompt: str) -> str:
    return complete(prompt, max_tokens=800)


def _extract_sql(text: str) -> str:
    sql_match = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        select_match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE)
        if not select_match:
            raise ValueError("LLM response does not contain a SELECT query")
        sql = text[select_match.start() :].strip()
        if ";" in sql:
            sql = sql[: sql.index(";") + 1]

    if not re.match(r"^SELECT\b", sql, flags=re.IGNORECASE):
        raise ValueError("LLM response must contain one SELECT query")
    return sql


def _build_prompt(file_path: str, table_rules: pd.DataFrame, sample: pd.DataFrame) -> str:
    formatted_rules = table_rules.to_string(index=False)
    column_types = {
        column_name: {
            "dtype": str(sample[column_name].dtype),
            "sample_values": sample[column_name].dropna().tolist(),
        }
        for column_name in sample.columns
    }
    sql_path = Path(file_path).resolve().as_posix().replace("'", "''")
    return (
        "Generate ONE DuckDB SELECT SQL query only. Read the source using "
        f"read_csv_auto('{sql_path}', header=true, all_varchar=true).\n"
        f"CSV file path: {file_path}\n"
        f"STTM rules for this table:\n{formatted_rules}\n"
        f"Column types and sample values from the first three rows: {column_types}\n"
        "Apply type_cast rules with TRY_CAST(). Map datetime to TIMESTAMP, "
        "float to DOUBLE, int to INTEGER, and str to VARCHAR. Pass through "
        "columns without a type_cast rule. Do not add _load_timestamp or "
        "_source_file in SQL; Python adds those metadata columns afterward. "
        "Return a single SELECT statement, with no markdown or explanation."
    )


def run(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    written_paths = []
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(run_id)

    for file_path in input_files:
        table_name = Path(file_path).stem
        table_rules = sttm[sttm["source_table"].astype(str) == table_name]
        sample = pd.read_csv(file_path, nrows=3, low_memory=False)
        prompt = _build_prompt(file_path, table_rules, sample)
        used_fallback = False
        connection = None

        try:
            sql = _extract_sql(_call_llm(prompt))
            connection = duckdb.connect()
            dataframe = connection.execute(sql).df()
        except Exception:
            dataframe = pd.read_csv(file_path, low_memory=False)
            used_fallback = True
        finally:
            if connection is not None:
                connection.close()

        dataframe["_load_timestamp"] = datetime.utcnow().isoformat() + "Z"
        dataframe["_source_file"] = str(file_path)

        output_path = BRONZE_DIR / f"{table_name}_bronze_{run_id[:8]}.parquet"
        dataframe.to_parquet(output_path, index=False)
        output_path_string = str(output_path)
        written_paths.append(output_path_string)
        audit_logger.log(
            agent="bronze_agent",
            action="bronze_written",
            input_file=str(file_path),
            output_path=output_path_string,
            row_count=len(dataframe),
            used_fallback=used_fallback,
        )

    return written_paths