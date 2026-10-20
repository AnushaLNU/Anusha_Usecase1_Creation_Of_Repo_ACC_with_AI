import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from core.audit import AuditLogger
from core.config import AUDIT_DIR, SILVER_DIR
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


def _table_base_name(table_name: str) -> str:
    stem = Path(str(table_name)).stem
    return re.sub(r"_bronze(?:_[^_]+)?$", "", stem)


def _match_sttm_rows(sttm: pd.DataFrame, base_name: str) -> pd.DataFrame:
    matched = pd.Series(False, index=sttm.index)
    for column_name in ("source_table", "target_table"):
        if column_name in sttm.columns:
            names = sttm[column_name].fillna("").astype(str).map(_table_base_name)
            matched |= names == base_name
    return sttm.loc[matched]


def _collect_context(connection, table_name: str) -> tuple[list[dict], dict]:
    quoted_table = _quote_identifier(table_name)
    schema_rows = connection.execute(f"DESCRIBE {quoted_table}").fetchall()
    schema = [{"column": row[0], "dtype": row[1]} for row in schema_rows]
    categorical_values = {}

    for column in schema:
        if column["dtype"].upper() == "VARCHAR":
            quoted_column = _quote_identifier(column["column"])
            values = connection.execute(
                f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
                f"WHERE {quoted_column} IS NOT NULL ORDER BY 1"
            ).fetchall()
            categorical_values[column["column"]] = [value[0] for value in values]

    return schema, categorical_values


def _build_prompt(
    table_name: str,
    schema: list[dict],
    categorical_values: dict,
    table_rules: pd.DataFrame,
) -> str:
    return (
        "Generate one DuckDB SELECT statement for the Silver table. Return SQL "
        "only.\n"
        f"Table: {table_name}\n"
        f"Table schema: {json.dumps(schema, default=str)}\n"
        "Distinct values for VARCHAR columns: "
        f"{json.dumps(categorical_values, ensure_ascii=False, default=str)}\n"
        f"STTM rules:\n{table_rules.to_string(index=False)}\n"
        "Apply the STTM rules using this guide: standardize_date should parse "
        "both MM/DD/YYYY and YYYY-MM-DD with "
        "COALESCE(TRY_STRPTIME(col, '%m/%d/%Y'), "
        "TRY_STRPTIME(col, '%Y-%m-%d')). For normalize_category, use CASE WHEN "
        "patterns elec% -> Electronics, apparel% or cloth% -> Apparel, beauty% "
        "or cosmetic% -> Beauty, sport% -> Sports, and home% or kitchen% -> "
        "Home & Kitchen. For normalize_segment, map prem% -> Premium, bgt% or "
        "budg% -> Budget, stand% or std% -> Standard, and loyal% -> Loyal. "
        "For null_handler with drop_nulls, add WHERE col IS NOT NULL; "
        f"for fill_median, use COALESCE(col, (SELECT median(col) FROM {_quote_identifier(table_name)})). "
        "For dedup, use SELECT DISTINCT. For passthrough, select col AS col "
        "unchanged. Use only columns present in the table schema."
    )


def run(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    written_paths = []
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_logger = AuditLogger(run_id)

    for bronze_path in bronze_paths:
        base_name = _table_base_name(bronze_path)
        table_name = base_name
        table_rules = _match_sttm_rows(sttm, base_name)
        quoted_table = _quote_identifier(table_name)
        parquet_path = Path(bronze_path).resolve().as_posix().replace("'", "''")
        connection = duckdb.connect()
        used_fallback = False

        try:
            connection.execute(
                f"CREATE OR REPLACE TABLE {quoted_table} AS "
                f"SELECT * FROM read_parquet('{parquet_path}')"
            )
            schema, categorical_values = _collect_context(connection, table_name)
            prompt = _build_prompt(
                table_name, schema, categorical_values, table_rules
            )
            sql = _extract_sql(_call_llm(prompt))
            dataframe = connection.execute(sql).df()
        except Exception:
            dataframe = pd.read_parquet(bronze_path).drop_duplicates()
            used_fallback = True
        finally:
            connection.close()

        surrogate_key = f"pk_{base_name}_silver_id"
        dataframe.insert(0, surrogate_key, range(1, len(dataframe) + 1))
        output_path = SILVER_DIR / f"{base_name}_silver_{run_id[:8]}.parquet"
        dataframe.to_parquet(output_path, index=False)
        output_path_string = str(output_path)
        written_paths.append(output_path_string)
        audit_logger.log(
            agent="silver_agent",
            action="silver_written",
            input_file=str(bronze_path),
            output_path=output_path_string,
            row_count=len(dataframe),
            used_fallback=used_fallback,
        )

    return written_paths