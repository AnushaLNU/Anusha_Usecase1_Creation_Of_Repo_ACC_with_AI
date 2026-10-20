import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from core.audit import AuditLogger
from core.config import AUDIT_DIR, GOLD_DIR
from core.llm import complete


def _call_llm(prompt: str) -> str:
    return complete(prompt, max_tokens=1200)


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


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _clean_table_name(file_path: str, run_id: str) -> str:
    stem = Path(file_path).stem
    run_suffix = f"_{run_id[:8]}"
    if stem.endswith(run_suffix):
        stem = stem[: -len(run_suffix)]
    return stem


def _build_prompt(business_intent: str, schemas: dict, gold_rules: pd.DataFrame) -> str:
    return (
        "Generate ONE DuckDB SELECT query that answers the business intent. "
        "Return SQL only.\n"
        f"Business intent: {business_intent}\n"
        f"Registered Silver table schemas: {json.dumps(schemas, ensure_ascii=False)}\n"
        f"Gold STTM rules:\n{gold_rules.to_string(index=False)}\n"
        "Parse each join rule whose transformation_logic is "
        "join_left:A:B:key as A LEFT JOIN B ON A.key = B.key. Parse "
        "group_by rules to determine GROUP BY columns, and aggregate rules "
        "such as SUM(total_amount) AS total_revenue as aggregate expressions. "
        "Generate one SELECT using the required joins, grouping, and aggregates. "
        "Use only registered tables and columns. Do not add pk_gold_id; Python "
        "adds it after the query. Order results by the main aggregate in "
        "descending order."
    )


def run(
    silver_paths: list[str],
    sttm_path: str,
    business_intent: str,
    run_id: str,
) -> list[str]:
    if not silver_paths:
        raise ValueError("At least one Silver parquet path is required")

    gold_rules = pd.read_csv(sttm_path)
    schemas = {}
    table_registry = {}
    connection = duckdb.connect()
    used_fallback = False

    try:
        for file_path in silver_paths:
            clean_name = _clean_table_name(file_path, run_id)
            table_registry[clean_name] = file_path
            quoted_name = _quote_identifier(clean_name)
            parquet_path = Path(file_path).resolve().as_posix().replace("'", "''")
            connection.execute(
                f"CREATE OR REPLACE TABLE {quoted_name} AS "
                f"SELECT * FROM read_parquet('{parquet_path}')"
            )
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {quoted_name}"
            ).fetchone()[0]
            column_rows = connection.execute(f"DESCRIBE {quoted_name}").fetchall()
            schemas[clean_name] = {
                "row_count": row_count,
                "columns": [
                    {"name": row[0], "dtype": row[1]} for row in column_rows
                ],
            }

        prompt = _build_prompt(business_intent, schemas, gold_rules)
        sql = _extract_sql(_call_llm(prompt))
        dataframe = connection.execute(sql).df()
    except Exception:
        dataframe = pd.read_parquet(silver_paths[0])
        used_fallback = True
    finally:
        connection.close()

    dataframe.insert(0, "pk_gold_id", range(1, len(dataframe) + 1))
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GOLD_DIR / f"sales_analytics_gold_{run_id[:8]}.parquet"
    dataframe.to_parquet(output_path, index=False)
    output_path_string = str(output_path)

    AuditLogger(run_id).log(
        agent="gold_agent",
        action="gold_written",
        input_files=silver_paths,
        output_path=output_path_string,
        row_count=len(dataframe),
        used_fallback=used_fallback,
    )
    return [output_path_string]