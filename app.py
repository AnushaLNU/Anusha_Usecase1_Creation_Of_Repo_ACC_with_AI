import json
import pathlib
import uuid

import pandas as pd
import streamlit as st

from agents import bronze_agent, gold_agent, profiler, reporter, silver_agent, sttm_generator
from core.config import AUDIT_DIR, REPORTS_DIR, STTM_DIR, ensure_dirs
from core.state import PipelineState


st.set_page_config(
    page_title="IDAMP — Agentic Medallion Pipeline",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      :root {
        --ink: #17312b;
        --muted: #63766f;
        --green: #176b58;
        --green-dark: #103d35;
        --gold: #d89936;
        --paper: #f5f7f3;
        --line: #dce5de;
      }
    #MainMenu, footer { visibility: hidden; }
    header [data-testid="stToolbar"] { visibility: hidden; }
    header [data-testid="stExpandSidebarButton"] { visibility: visible !important; z-index: 10000; }
    [data-testid="stSidebarCollapseButton"] { visibility: visible !important; color: #eef5f0 !important; }
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
      [data-testid="stAppViewContainer"] { background: var(--paper); }
      [data-testid="stSidebar"] { background: var(--green-dark); }
      [data-testid="stSidebar"] * { color: #eef5f0; }
      [data-testid="stSidebar"] code { color: #17312b; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: "DM Sans", sans-serif; }
      h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
      h1 { font-weight: 700; }
      [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line); padding: 14px 16px; border-radius: 6px; }
      [data-testid="stFileUploader"] section { border: 1px dashed #91aaa0; background: #fbfcfa; }
      button[kind="primary"] { background: var(--green); border-color: var(--green); }
      .step-row { display: flex; gap: 10px; align-items: baseline; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,.1); }
      .step-symbol { min-width: 20px; font-family: "IBM Plex Mono", monospace; }
      .step-current { color: #ffd17c; font-weight: 700; }
      .step-aborted { color: #ff9a91; font-weight: 700; }
      .eyebrow { color: var(--green); font: 600 12px "IBM Plex Mono", monospace; text-transform: uppercase; }
      .quiet-rule { border: 0; border-top: 1px solid var(--line); margin: 1.25rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


_SESSION_DEFAULTS = {
    "stage": "input",
    "state": None,
    "run_id": None,
    "input_files": [],
    "intent": "",
    "log_messages": [],
    "report_html": None,
    "edited_sttm": {},
    "editor_versions": {},
    "aborted_at": None,
    "error_message": None,
    "report_is_error": False,
}

for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default.copy() if isinstance(_default, (list, dict)) else _default


_STEPS = [
    "Upload & Intent",
    "Profile + Bronze STTM",
    "[HITL] Review Bronze STTM",
    "Bronze Agent",
    "Profile + Silver STTM",
    "[HITL] Review Silver STTM",
    "Silver Agent",
    "Gold STTM",
    "[HITL] Review Gold STTM",
    "Gold Agent + Report",
]

_ACTIVE_STEP = {
    "input": 0,
    "running_phase1": 1,
    "bronze_hitl": 2,
    "running_phase2": 3,
    "silver_hitl": 5,
    "running_phase3": 6,
    "gold_hitl": 8,
    "running_phase4": 9,
    "complete": len(_STEPS),
}
_ABORT_STEP = {
    "input": 0,
    "running_phase1": 1,
    "bronze_hitl": 2,
    "running_phase2": 3,
    "silver_hitl": 5,
    "running_phase3": 6,
    "gold_hitl": 8,
    "running_phase4": 9,
}


def _log(message: str) -> None:
    st.session_state["log_messages"].append(message)


def _aborted_step() -> int:
    gate = st.session_state.get("aborted_at")
    return _ABORT_STEP.get(gate, 0)


def _step_marker(index: int) -> tuple[str, str]:
    stage = st.session_state["stage"]
    if stage == "complete":
        return "✅", ""
    if stage == "aborted":
        failed_step = _aborted_step()
        if index < failed_step:
            return "✅", ""
        if index == failed_step:
            return "❌", "step-aborted"
        return "⚪", ""

    current_step = _ACTIVE_STEP.get(stage, 0)
    if index < current_step:
        return "✅", ""
    if index == current_step:
        return "🔵", "step-current"
    return "⚪", ""


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="eyebrow">IDAMP / Run monitor</div>', unsafe_allow_html=True)
        st.markdown("### Pipeline status")
        for index, step in enumerate(_STEPS):
            symbol, status_class = _step_marker(index)
            st.markdown(
                f'<div class="step-row {status_class}">'
                f'<span class="step-symbol">{symbol}</span><span>{step}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<hr class="quiet-rule">', unsafe_allow_html=True)
        if st.session_state["run_id"]:
            st.caption("RUN ID")
            st.code(st.session_state["run_id"][:8])
        if st.session_state["intent"]:
            st.caption("BUSINESS INTENT")
            st.caption(st.session_state["intent"])
        st.caption(f"Progress messages: {len(st.session_state['log_messages'])}")
        if st.session_state["log_messages"]:
            with st.expander("Recent activity"):
                for message in st.session_state["log_messages"][-8:]:
                    st.caption(message)


def _handle_error(error: Exception) -> None:
    st.session_state["error_message"] = str(error)
    st.session_state["aborted_at"] = st.session_state.get("stage")
    st.session_state["stage"] = "aborted"
    _log(f"Pipeline error: {error}")
    st.rerun()


def _count_type(dataframe: pd.DataFrame, value: str) -> int:
    if "transformation_type" not in dataframe.columns:
        return 0
    return int(dataframe["transformation_type"].fillna("").astype(str).eq(value).sum())


def _render_review(layer_name: str) -> None:
    state = st.session_state["state"]
    layer_key = layer_name.lower()
    path_attribute = f"{layer_key}_sttm_path"
    sttm_path = getattr(state, path_attribute)
    dataframe = pd.read_csv(sttm_path)

    descriptions = {
        "Bronze": (
            "The LLM has generated the Bronze transformation rules. Review them below. "
            "You can edit any cell before approving."
        ),
        "Silver": (
            "Silver rules cleanse and standardize the Bronze data. Review normalization "
            "rules carefully, especially normalize_category and standardize_date."
        ),
        "Gold": (
            "Gold rules define how tables are joined and aggregated to answer your "
            "business question. Verify the join keys and aggregate columns match your intent."
        ),
    }
    stage_names = {
        "Bronze": "🔶 Review Bronze STTM",
        "Silver": "⬜ Review Silver STTM",
        "Gold": "🥇 Review Gold STTM",
    }
    st.subheader(stage_names[layer_name])
    st.info(descriptions[layer_name])

    if layer_name == "Gold":
        st.markdown("**Business intent being answered:**")
        st.info(st.session_state["intent"])

    editor_version = st.session_state["editor_versions"].get(layer_key, 0)
    edited_dataframe = st.data_editor(
        dataframe,
        use_container_width=True,
        num_rows="dynamic",
        key=f"{layer_key}_editor_{editor_version}",
    )
    st.session_state["edited_sttm"][layer_name] = edited_dataframe

    if layer_name == "Bronze":
        metrics = [
            ("Total rules", len(dataframe)),
            ("type_cast rules", _count_type(dataframe, "type_cast")),
            ("metadata_inject rules", _count_type(dataframe, "metadata_inject")),
        ]
    elif layer_name == "Silver":
        normalize_count = 0
        if "transformation_type" in dataframe.columns:
            normalize_count = int(
                dataframe["transformation_type"]
                .fillna("")
                .astype(str)
                .str.startswith("normalize_")
                .sum()
            )
        metrics = [
            ("null_handler rules", _count_type(dataframe, "null_handler")),
            ("normalize rules", normalize_count),
            ("dedup rules", _count_type(dataframe, "dedup")),
        ]
    else:
        metrics = [
            ("join rules", _count_type(dataframe, "join")),
            ("group_by rules", _count_type(dataframe, "group_by")),
            ("aggregate rules", _count_type(dataframe, "aggregate")),
        ]

    metric_columns = st.columns(3)
    for column, (label, value) in zip(metric_columns, metrics):
        column.metric(label, value)

    approve_column, regenerate_column, abort_column = st.columns(3)
    if approve_column.button(
        "✅ Approve & Continue", type="primary", key=f"{layer_key}_approve"
    ):
        edited_dataframe.to_csv(sttm_path, index=False)
        next_stage = {
            "Bronze": "running_phase2",
            "Silver": "running_phase3",
            "Gold": "running_phase4",
        }[layer_name]
        st.session_state["stage"] = next_stage
        st.session_state["aborted_at"] = None
        _log(f"{layer_name} STTM approved")
        st.rerun()

    if regenerate_column.button(
        f"🔄 Regenerate {layer_name} STTM", key=f"{layer_key}_regenerate"
    ):
        try:
            with st.spinner(f"Regenerating {layer_name} STTM..."):
                run_id = st.session_state["run_id"]
                intent = st.session_state["intent"]
                if layer_name == "Bronze":
                    new_path = sttm_generator.generate_bronze_sttm(
                        state.profile_path, intent, run_id
                    )
                elif layer_name == "Silver":
                    new_path = sttm_generator.generate_silver_sttm(
                        state.bronze_paths, state.bronze_sttm_path, intent, run_id
                    )
                else:
                    new_path = sttm_generator.generate_gold_sttm(
                        state.silver_paths, state.silver_sttm_path, intent, run_id
                    )
                setattr(state, path_attribute, new_path)
                st.session_state["edited_sttm"].pop(layer_name, None)
                st.session_state["editor_versions"][layer_key] = editor_version + 1
                _log(f"{layer_name} STTM regenerated")
            st.rerun()
        except Exception as error:
            _handle_error(error)

    if abort_column.button("❌ Abort Pipeline", key=f"{layer_key}_abort"):
        st.session_state["stage"] = "aborted"
        st.session_state["aborted_at"] = f"{layer_key}_hitl"
        _log(f"Pipeline aborted at {layer_name} STTM review")
        st.rerun()


def _start_new_run() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


_render_sidebar()


if st.session_state["stage"] == "input":
    st.markdown('<div class="eyebrow">Data operations / Medallion pipeline</div>', unsafe_allow_html=True)
    st.title("🥇 Intent-Driven Agentic Medallion Pipeline")
    st.markdown(
        "Upload raw CSV files and ask a business question. The pipeline profiles, "
        "cleans, aggregates, and prepares an HTML analytics report, with review "
        "gates before each data layer proceeds."
    )
    st.markdown('<hr class="quiet-rule">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload raw CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload all source CSV files (for example, sales_data.csv, products.csv, stores.csv).",
    )
    upload_directory = pathlib.Path("data/uploads")
    upload_directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files or []:
        safe_name = pathlib.Path(uploaded_file.name).name
        destination = upload_directory / safe_name
        uploaded_file.seek(0)
        destination.write_bytes(uploaded_file.read())
        saved_paths.append(str(destination))
    st.session_state["input_files"] = saved_paths

    intent = st.text_area(
        "Business Question / Intent",
        placeholder="Which category had the highest Q1 2024 revenue, and how does it break down by region?",
        height=100,
        key="intent_input",
    )
    st.session_state["intent"] = intent.strip()

    with st.expander("Example business questions"):
        st.markdown(
            "- Which customer segment had the highest total spending?\n"
            "- How does revenue break down by region and payment method?\n"
            "- Which products have the highest return rate?\n"
            "- What is the average order value per store?\n"
            "- Which month in Q1 2024 had the most transactions?"
        )

    if st.button(
        "🚀 Start Pipeline",
        type="primary",
        disabled=not saved_paths or not st.session_state["intent"],
    ):
        try:
            ensure_dirs()
            st.session_state["run_id"] = str(uuid.uuid4())
            st.session_state["intent"] = intent.strip()
            st.session_state["stage"] = "running_phase1"
            st.session_state["active_step"] = 1
            st.session_state["aborted_at"] = None
            st.session_state["error_message"] = None
            st.session_state["report_is_error"] = False
            st.session_state["log_messages"] = []
            st.session_state["edited_sttm"] = {}
            st.session_state["editor_versions"] = {}
            st.rerun()
        except Exception as error:
            _handle_error(error)


elif st.session_state["stage"] == "running_phase1":
    with st.spinner("Phase 1 — Profiling data and generating Bronze STTM..."):
        try:
            run_id = st.session_state["run_id"]
            files = st.session_state["input_files"]
            intent = st.session_state["intent"]
            pipeline_state = PipelineState(
                run_id=run_id,
                input_files=files,
                business_intent=intent,
            )
            st.session_state["state"] = pipeline_state
            pipeline_state.profile_path = profiler.profile(files, run_id)
            _log(f"Profile created: {pipeline_state.profile_path}")
            pipeline_state.bronze_sttm_path = sttm_generator.generate_bronze_sttm(
                pipeline_state.profile_path, intent, run_id
            )
            st.session_state["state"] = pipeline_state
            st.session_state["stage"] = "bronze_hitl"
            _log("Bronze STTM ready for review")
        except Exception as error:
            _handle_error(error)
    st.rerun()


elif st.session_state["stage"] == "bronze_hitl":
    _render_review("Bronze")


elif st.session_state["stage"] == "running_phase2":
    with st.spinner("Phase 2 — Running Bronze Agent and generating Silver STTM..."):
        try:
            state = st.session_state["state"]
            state.bronze_paths = bronze_agent.run(
                state.input_files, state.bronze_sttm_path, state.run_id
            )
            _log(f"Bronze layer written: {len(state.bronze_paths)} files")
            state.silver_sttm_path = sttm_generator.generate_silver_sttm(
                state.bronze_paths,
                state.bronze_sttm_path,
                st.session_state["intent"],
                state.run_id,
            )
            st.session_state["state"] = state
            st.session_state["stage"] = "silver_hitl"
            _log("Silver STTM ready for review")
        except Exception as error:
            _handle_error(error)
    st.rerun()


elif st.session_state["stage"] == "silver_hitl":
    _render_review("Silver")


elif st.session_state["stage"] == "running_phase3":
    with st.spinner("Phase 3 — Running Silver Agent and generating Gold STTM..."):
        try:
            state = st.session_state["state"]
            state.silver_paths = silver_agent.run(
                state.bronze_paths, state.silver_sttm_path, state.run_id
            )
            _log(f"Silver layer written: {len(state.silver_paths)} files")
            state.gold_sttm_path = sttm_generator.generate_gold_sttm(
                state.silver_paths,
                state.silver_sttm_path,
                st.session_state["intent"],
                state.run_id,
            )
            st.session_state["state"] = state
            st.session_state["stage"] = "gold_hitl"
            _log("Gold STTM ready for review")
        except Exception as error:
            _handle_error(error)
    st.rerun()


elif st.session_state["stage"] == "gold_hitl":
    _render_review("Gold")


elif st.session_state["stage"] == "running_phase4":
    with st.spinner("Phase 4 — Running Gold Agent and generating the analytics report..."):
        try:
            state = st.session_state["state"]
            state.gold_paths = gold_agent.run(
                state.silver_paths,
                state.gold_sttm_path,
                st.session_state["intent"],
                state.run_id,
            )
            _log(f"Gold layer written: {len(state.gold_paths)} files")
            report_result = reporter.generate_report(
                state.gold_paths, st.session_state["intent"], state.run_id
            )
            if report_result.lstrip().startswith("<"):
                state.report_path = None
                st.session_state["report_html"] = report_result
                st.session_state["report_is_error"] = True
            else:
                report_path = pathlib.Path(report_result)
                if not report_path.is_file():
                    raise FileNotFoundError(f"Report output was not found: {report_result}")
                state.report_path = report_result
                st.session_state["report_html"] = report_path.read_text(encoding="utf-8")
            st.session_state["state"] = state
            st.session_state["stage"] = "complete"
            _log("Report ready")
        except Exception as error:
            _handle_error(error)
    st.rerun()


elif st.session_state["stage"] == "complete":
    if st.session_state["report_is_error"]:
        st.warning("The data pipeline completed, but the reporter returned an error page.")
    else:
        st.success("✅ Pipeline complete!")
    st.subheader("📊 Analytics Report")

    report_tab, gold_tab, sttm_tab, audit_tab = st.tabs(
        ["📄 Report", "🥇 Gold Data", "📋 All STTMs", "🪵 Audit Log"]
    )
    with report_tab:
        report_html = st.session_state["report_html"] or ""
        st.components.v1.html(report_html, height=900, scrolling=True)
        st.download_button(
            label="⬇️ Download HTML Report",
            data=report_html,
            file_name=f"report_{st.session_state['run_id'][:8]}.html",
            mime="text/html",
        )
        st.caption(f"Artifacts: {REPORTS_DIR}")

    with gold_tab:
        state = st.session_state["state"]
        for gold_path in state.gold_paths:
            dataframe = pd.read_parquet(gold_path)
            st.dataframe(dataframe, use_container_width=True)
            st.caption(
                f"Source: {gold_path}  |  {len(dataframe)} rows, "
                f"{len(dataframe.columns)} columns"
            )

    with sttm_tab:
        state = st.session_state["state"]
        bronze_tab, silver_tab, gold_sttm_tab = st.tabs(
            ["🔶 Bronze", "⬜ Silver", "🥇 Gold"]
        )
        with bronze_tab:
            if state.bronze_sttm_path:
                st.dataframe(pd.read_csv(state.bronze_sttm_path), use_container_width=True)
        with silver_tab:
            if state.silver_sttm_path:
                st.dataframe(pd.read_csv(state.silver_sttm_path), use_container_width=True)
        with gold_sttm_tab:
            if state.gold_sttm_path:
                st.dataframe(pd.read_csv(state.gold_sttm_path), use_container_width=True)
        st.caption(f"STTM artifacts: {STTM_DIR}")

    with audit_tab:
        state = st.session_state["state"]
        audit_file = AUDIT_DIR / f"{state.run_id}.jsonl"
        if audit_file.exists():
            entries = [
                json.loads(line)
                for line in audit_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            st.metric("Total audit entries", len(entries))
            st.dataframe(pd.DataFrame(entries), use_container_width=True)
        else:
            st.info("No audit log found for this run.")

    st.divider()
    if st.button("🔁 Start a New Run"):
        _start_new_run()


elif st.session_state["stage"] == "aborted":
    st.error("❌ Pipeline aborted.")
    if st.session_state["error_message"]:
        st.exception(RuntimeError(st.session_state["error_message"]))
    else:
        st.markdown("The pipeline was stopped at the HITL review stage. No report was generated.")

    state = st.session_state.get("state")
    if state:
        if state.bronze_paths:
            st.success(f"Bronze layer: {len(state.bronze_paths)} files written")
        if state.silver_paths:
            st.success(f"Silver layer: {len(state.silver_paths)} files written")

    if st.button("🔁 Start a New Run"):
        _start_new_run()