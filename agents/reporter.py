import html
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px

from core.audit import AuditLogger
from core.config import AUDIT_DIR, REPORTS_DIR
from core.llm import complete


def _call_llm(prompt: str) -> str:
    return complete(prompt, max_tokens=1800)


def _extract_sql(text: str) -> str:
    sql_match = re.search(r"```sql\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    sql_match = re.search(
        r"(SELECT\s+.+?;)", text, flags=re.DOTALL | re.IGNORECASE
    )
    return sql_match.group(1).strip() if sql_match else ""


def _extract_narrative(text: str) -> str:
    narrative_match = re.search(
        r"NARRATIVE:\s*(.*?)(?=CHARTS:|SQL:|$)", text, flags=re.DOTALL
    )
    return narrative_match.group(1).strip() if narrative_match else ""


def _extract_json_block(text: str) -> list:
    charts_match = re.search(
        r"CHARTS:.*?```json\s*(.*?)```", text, flags=re.DOTALL
    )
    if not charts_match:
        return []

    try:
        chart_specs = json.loads(charts_match.group(1).strip())
    except (json.JSONDecodeError, TypeError):
        return []

    required_keys = {"type", "x", "y", "title"}
    if not isinstance(chart_specs, list) or any(
        not isinstance(spec, dict) or not required_keys.issubset(spec)
        for spec in chart_specs
    ):
        return []
    return chart_specs


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _build_prompt(table_context: list[str], business_intent: str) -> str:
    return (
        "Answer the business question using the registered DuckDB tables.\n"
        f"Business intent: {business_intent}\n\n"
        "Table schemas and preview rows:\n"
        + "\n\n".join(table_context)
        + "\n\nOutput exactly three labeled sections in this order:\n"
        "SQL: a single DuckDB-compatible SELECT statement answering the "
        "business question. Wrap the SQL in a ```sql code fence.\n"
        "NARRATIVE: a 2-3 paragraph executive summary in plain English, "
        "highlighting key trends, top performers, and business implications.\n"
        "CHARTS: a JSON array of chart specs, each with keys type, x, y, and "
        "title. Wrap the JSON in a ```json code fence. Use only chart types "
        "bar, pie, or line. Column names must exactly match those shown in "
        "the table previews."
    )


def _error_page(message: str) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>Report Error</title></head><body>"
        f"<h1>Report generation failed</h1><p>{html.escape(message)}</p>"
        "</body></html>"
    )


def generate_report(gold_paths: list[str], business_intent: str, run_id: str) -> str:
    connection = None
    audit_logger = None
    execution_path = "fallback"

    try:
        if not gold_paths:
            raise ValueError("At least one Gold parquet path is required")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(run_id)
        connection = duckdb.connect()
        table_context = []
        first_gold_table = None

        for gold_path in gold_paths:
            table_name = Path(gold_path).stem
            quoted_table = _quote_identifier(table_name)
            quoted_path = Path(gold_path).resolve().as_posix().replace("'", "''")
            if first_gold_table is None:
                first_gold_table = table_name
            connection.execute(
                f"CREATE VIEW {quoted_table} AS "
                f"SELECT * FROM read_parquet('{quoted_path}')"
            )
            columns = connection.execute(f"DESCRIBE {quoted_table}").df()
            preview = connection.execute(
                f"SELECT * FROM {quoted_table} LIMIT 5"
            ).df()
            table_context.append(
                f"Table: {table_name}\n"
                f"Columns:\n{columns.to_string(index=False)}\n"
                f"Preview rows:\n{preview.to_string(index=False)}"
            )

        prompt = _build_prompt(table_context, business_intent)
        model_response = _call_llm(prompt)
        sql = _extract_sql(model_response)

        try:
            if not sql:
                raise ValueError("LLM response did not contain a SQL query")
            result_df = connection.execute(sql).df()
            execution_path = "sql_executed"
        except Exception:
            result_df = connection.execute(
                f"SELECT * FROM {_quote_identifier(first_gold_table)} LIMIT 20"
            ).df()
            execution_path = "fallback"

        narrative = _extract_narrative(model_response)
        chart_specs = _extract_json_block(model_response)
        chart_htmls = []
        for spec in chart_specs:
            chart_type = spec["type"]
            x_column = spec["x"]
            y_column = spec["y"]
            if (
                chart_type not in {"bar", "pie", "line"}
                or not isinstance(x_column, str)
                or not isinstance(y_column, str)
                or x_column not in result_df.columns
                or y_column not in result_df.columns
            ):
                continue

            if chart_type == "bar":
                figure = px.bar(
                    result_df, x=x_column, y=y_column, title=spec["title"]
                )
            elif chart_type == "pie":
                figure = px.pie(
                    result_df, names=x_column, values=y_column, title=spec["title"]
                )
            else:
                figure = px.line(
                    result_df, x=x_column, y=y_column, title=spec["title"]
                )
            figure.update_layout(
                template="plotly_white",
                font=dict(family="Arial", size=13),
                margin=dict(t=50, b=40, l=40, r=40),
            )
            chart_htmls.append(
                figure.to_html(full_html=False, include_plotlyjs=False)
            )

        table_html = result_df.to_html(
            index=False,
            classes="data-table",
            border=0,
            na_rep="\u2014",
        )
        chart_divs = "\n".join(
            f'<div class="chart-box">{chart_html}</div>'
            for chart_html in chart_htmls
        )
        escaped_run_id = html.escape(str(run_id))
        escaped_business_intent = html.escape(str(business_intent))
        escaped_narrative = html.escape(narrative)
        report_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Report \u2014 {escaped_run_id}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1a1a2e; }}
    h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }}
    h2 {{ color: #1565C0; margin-top: 36px; }}
    .narrative {{ background: #f0f5ff; border-left: 4px solid #1565C0; padding: 16px 20px; border-radius: 4px; line-height: 1.7; white-space: pre-wrap; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
    .data-table th {{ background: #1565C0; color: white; padding: 8px 12px; text-align: left; }}
    .data-table td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }}
    .data-table tr:nth-child(even) {{ background: #f5f7fa; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 24px; margin-top: 16px; }}
    .chart-box {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; background: #ffffff; }}
    footer {{ margin-top: 48px; font-size: 12px; color: #999; text-align: center; border-top: 1px solid #e0e0e0; padding-top: 16px; }}
  </style>
</head>
<body>
  <h1>Business Intelligence Report</h1>
  <p><strong>Run ID:</strong> {escaped_run_id}</p>
  <p><strong>Business Intent:</strong> {escaped_business_intent}</p>
  <h2>Executive Summary</h2>
  <div class="narrative">{escaped_narrative}</div>
  <h2>Data Results</h2>
  {table_html}
  <h2>Visual Analysis</h2>
  <div class="chart-grid">
    {chart_divs}
  </div>
  <footer>Generated by Agentic Medallion Pipeline · Run {escaped_run_id}</footer>
</body>
</html>"""

        output_path = REPORTS_DIR / f"report_{run_id[:8]}.html"
        output_path.write_text(report_html, encoding="utf-8")
        audit_logger.log(
            agent="reporter",
            action="report_generated",
            output_path=str(output_path),
            execution_path=execution_path,
            row_count=len(result_df),
            chart_count=len(chart_htmls),
        )
        return str(output_path)
    except Exception as error:
        try:
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            if audit_logger is None:
                audit_logger = AuditLogger(run_id)
            audit_logger.log(
                agent="reporter",
                action="report_error",
                error=str(error),
            )
        except Exception:
            pass
        return _error_page(str(error))
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass