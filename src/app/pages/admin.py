"""Admin page with live status and Modal-triggered actions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import duckdb
import streamlit as st

from src.config import KNOWLEDGE_SOURCES

DUCKDB_PATH = "/data/processed/sentiment.duckdb"
TRAIN_PATH = Path("/data/training/training_data.jsonl")
VAL_PATH = Path("/data/training/training_data_val.jsonl")
FINAL_MODEL_PATH = Path("/models/fine-tuned/final")
MERGED_MODEL_PATH = Path("/models/fine-tuned/merged")
MODAL_APP_NAME = os.environ.get("MODAL_APP_NAME", "modal-social-sentiment")


SOURCE_TO_FUNCTION = {
    "docs": "ingest_docs",
    "github": "ingest_github",
    "hackernews": "ingest_hackernews",
}


def _db_available() -> bool:
    return Path(DUCKDB_PATH).exists()


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r") as file:
        return sum(1 for line in file if line.strip())


def _load_snapshot() -> dict:
    snapshot = {
        "total_messages": 0,
        "voice_messages": 0,
        "knowledge_messages": 0,
        "processed": 0,
        "embedded": 0,
        "pending_embeddings": 0,
        "pending_sentiment": 0,
        "sources": [],
        "ingestion_state": {},
        "training_examples": _count_jsonl_lines(TRAIN_PATH),
        "validation_examples": _count_jsonl_lines(VAL_PATH),
        "model_final": FINAL_MODEL_PATH.exists(),
        "model_merged": MERGED_MODEL_PATH.exists(),
    }

    if not _db_available():
        return snapshot

    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        knowledge_placeholders = ", ".join(["?"] * len(KNOWLEDGE_SOURCES))
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_messages,
                COUNT(*) FILTER (WHERE source NOT IN ({knowledge_placeholders})) AS voice_messages,
                COUNT(*) FILTER (WHERE source IN ({knowledge_placeholders})) AS knowledge_messages,
                COUNT(*) FILTER (WHERE processed_at IS NOT NULL) AS processed,
                COUNT(*) FILTER (WHERE embedding_id IS NOT NULL) AS embedded,
                COUNT(*) FILTER (WHERE embedding_id IS NULL) AS pending_embeddings,
                COUNT(*) FILTER (WHERE sentiment_simple IS NULL) AS pending_sentiment
            FROM messages
            """,
            KNOWLEDGE_SOURCES + KNOWLEDGE_SOURCES,
        ).fetchone()

        snapshot.update(
            {
                "total_messages": totals[0] or 0,
                "voice_messages": totals[1] or 0,
                "knowledge_messages": totals[2] or 0,
                "processed": totals[3] or 0,
                "embedded": totals[4] or 0,
                "pending_embeddings": totals[5] or 0,
                "pending_sentiment": totals[6] or 0,
            }
        )

        source_rows = conn.execute(
            """
            SELECT source, COUNT(*) AS total, MAX(created_at) AS latest_message_at
            FROM messages
            GROUP BY source
            ORDER BY total DESC
            """
        ).fetchall()
        snapshot["sources"] = source_rows

        state_rows = conn.execute(
            """
            SELECT source, last_fetched_at
            FROM ingestion_state
            ORDER BY source
            """
        ).fetchall()
        snapshot["ingestion_state"] = {source: last for source, last in state_rows}
    finally:
        conn.close()

    return snapshot


def _trigger_modal_function(function_name: str, *args, **kwargs) -> tuple[bool, str]:
    try:
        import modal

        function_ref = modal.Function.from_name(MODAL_APP_NAME, function_name)
        function_call = function_ref.spawn(*args, **kwargs)
        return True, f"Started `{function_name}` (call id: `{function_call.object_id}`)."
    except Exception as exc:
        return False, f"Failed to start `{function_name}`: {exc}"


def _format_time(value) -> str:
    if value is None:
        return "never"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def render_admin():
    """Render admin controls and system status."""
    st.title("Admin Panel")
    st.caption("Manage ingestion and processing jobs")

    snapshot = _load_snapshot()

    if not _db_available():
        st.warning("No database found at `/data/processed/sentiment.duckdb`.")

    st.markdown("### System Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Voice Messages", f"{snapshot['voice_messages']:,}")
    with col2:
        st.metric("Knowledge Messages", f"{snapshot['knowledge_messages']:,}")
    with col3:
        st.metric("Pending Embeddings", f"{snapshot['pending_embeddings']:,}")
    with col4:
        st.metric("Pending Sentiment", f"{snapshot['pending_sentiment']:,}")

    st.markdown("### Ingestion Jobs")

    if snapshot["sources"]:
        for source, total, latest in snapshot["sources"]:
            source_name = source.replace("_", " ").title()
            last_fetch = snapshot["ingestion_state"].get(source)

            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.markdown(f"**{source_name}**")
            with col2:
                st.caption(f"Messages: {total:,}")
            with col3:
                st.caption(
                    f"Last fetched: {_format_time(last_fetch)} | "
                    f"Last message: {_format_time(latest)}"
                )
            with col4:
                function_name = SOURCE_TO_FUNCTION.get(source)
                if function_name and st.button("Run", key=f"run_{source}"):
                    ok, message = _trigger_modal_function(function_name)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
    else:
        st.info("No source data available yet.")

    st.markdown("### Processing Jobs")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Embeddings", use_container_width=True):
            ok, message = _trigger_modal_function("generate_embeddings")
            st.success(message) if ok else st.error(message)

    with col2:
        if st.button("Run Sentiment", use_container_width=True):
            ok, message = _trigger_modal_function("analyze_sentiment")
            st.success(message) if ok else st.error(message)

    with col3:
        if st.button("Reset Embeddings", use_container_width=True):
            ok, message = _trigger_modal_function("reset_embeddings")
            st.success(message) if ok else st.error(message)

    st.markdown("### Fine-tuning")
    with st.expander("Training Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
        with col2:
            train_path = st.text_input(
                "Train Path (optional)",
                value=str(TRAIN_PATH),
                help="Leave as default unless you use a custom dataset path.",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Prepare Data", use_container_width=True):
                ok, message = _trigger_modal_function("prepare_training_data")
                st.success(message) if ok else st.error(message)

        with col2:
            if st.button("Start Training", use_container_width=True, type="primary"):
                kwargs = {"epochs": int(epochs)}
                if train_path.strip() and train_path.strip() != str(TRAIN_PATH):
                    kwargs["train_path"] = train_path.strip()
                ok, message = _trigger_modal_function("run_finetuning", **kwargs)
                st.success(message) if ok else st.error(message)

        with col3:
            if st.button("Merge Weights", use_container_width=True):
                ok, message = _trigger_modal_function("merge_finetuned_weights")
                st.success(message) if ok else st.error(message)

    st.markdown("### Artifacts")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Embedded", f"{snapshot['embedded']:,}")
    with col2:
        st.metric("Processed", f"{snapshot['processed']:,}")
    with col3:
        st.metric("Train Examples", f"{snapshot['training_examples']:,}")
    with col4:
        st.metric("Val Examples", f"{snapshot['validation_examples']:,}")

    st.caption(
        f"Model final: {'yes' if snapshot['model_final'] else 'no'} | "
        f"Model merged: {'yes' if snapshot['model_merged'] else 'no'} | "
        f"Modal app: `{MODAL_APP_NAME}`"
    )
