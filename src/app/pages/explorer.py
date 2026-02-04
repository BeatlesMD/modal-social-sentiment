"""Message explorer page with live DuckDB-backed filters."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

DUCKDB_PATH = "/data/processed/sentiment.duckdb"
DEFAULT_LIMIT = 500

DATE_FILTERS = {
    "Last 24 hours": 1,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "All time": None,
}


def _db_available() -> bool:
    return Path(DUCKDB_PATH).exists()


def _parse_topics(raw_value) -> list[str]:
    if isinstance(raw_value, list):
        return [str(v) for v in raw_value]
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        return []
    return []


def _get_filter_options() -> dict[str, list[str]]:
    if not _db_available():
        return {"sources": [], "sentiments": [], "content_types": [], "topics": []}

    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        sources = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM messages WHERE source IS NOT NULL ORDER BY source"
            ).fetchall()
        ]
        sentiments = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT sentiment_simple FROM messages "
                "WHERE sentiment_simple IS NOT NULL ORDER BY sentiment_simple"
            ).fetchall()
        ]
        content_types = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT content_type FROM messages "
                "WHERE content_type IS NOT NULL ORDER BY content_type"
            ).fetchall()
        ]
        topics = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT topic
                FROM messages,
                UNNEST(from_json(topics, '["VARCHAR"]')) AS t(topic)
                WHERE topics IS NOT NULL AND topics != '[]'
                ORDER BY topic
                """
            ).fetchall()
        ]
    finally:
        conn.close()

    return {
        "sources": sources,
        "sentiments": sentiments,
        "content_types": content_types,
        "topics": topics,
    }


def _fetch_messages(
    source_filter: list[str],
    sentiment_filter: list[str],
    content_type_filter: list[str],
    topic_filter: list[str],
    search_query: str,
    date_range: str,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    if not _db_available():
        return pd.DataFrame()

    conditions = []
    params: list[object] = []

    if source_filter:
        placeholders = ", ".join(["?"] * len(source_filter))
        conditions.append(f"source IN ({placeholders})")
        params.extend(source_filter)

    if sentiment_filter:
        placeholders = ", ".join(["?"] * len(sentiment_filter))
        conditions.append(f"sentiment_simple IN ({placeholders})")
        params.extend(sentiment_filter)

    if content_type_filter:
        placeholders = ", ".join(["?"] * len(content_type_filter))
        conditions.append(f"content_type IN ({placeholders})")
        params.extend(content_type_filter)

    lookback_days = DATE_FILTERS.get(date_range)
    if lookback_days:
        conditions.append("created_at >= CURRENT_TIMESTAMP - (? * INTERVAL '1 day')")
        params.append(lookback_days)

    if search_query.strip():
        pattern = f"%{search_query.strip().lower()}%"
        conditions.append("(LOWER(content) LIKE ? OR LOWER(COALESCE(title, '')) LIKE ?)")
        params.extend([pattern, pattern])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            id,
            source,
            content,
            author,
            created_at,
            COALESCE(sentiment_simple, 'neutral') AS sentiment,
            sentiment_rich,
            COALESCE(content_type, 'discussion') AS content_type,
            topics,
            url,
            title
        FROM messages
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
    """
    params.append(limit)

    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        rows = conn.execute(query, params).fetchall()
        columns = [d[0] for d in conn.description]
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "id",
            "source",
            "content",
            "author",
            "created_at",
            "sentiment",
            "sentiment_rich",
            "content_type",
            "topics",
            "url",
            "title",
        ])

    df = pd.DataFrame(rows, columns=columns)
    df["topic_list"] = df["topics"].apply(_parse_topics)

    if topic_filter:
        wanted = set(topic_filter)
        df = df[df["topic_list"].apply(lambda values: bool(wanted.intersection(values)))]

    return df


def _display_source(source: str) -> str:
    return source.replace("_", " ").title()


def render_explorer():
    """Render the live message explorer."""
    st.title("Message Explorer")
    st.caption("Search and filter through collected messages")

    if not _db_available():
        st.warning("No database found at `/data/processed/sentiment.duckdb`. Run ingestion first.")
        return

    options = _get_filter_options()

    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            source_filter = st.multiselect("Source", options["sources"], default=[])

        with col2:
            sentiment_filter = st.multiselect(
                "Sentiment", options["sentiments"], default=[]
            )

        with col3:
            content_type_filter = st.multiselect(
                "Content Type", options["content_types"], default=[]
            )

        with col4:
            topic_filter = st.multiselect("Topics", options["topics"], default=[])

        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search", placeholder="Search messages...")
        with col2:
            date_range = st.selectbox("Time Range", list(DATE_FILTERS.keys()), index=2)

    messages = _fetch_messages(
        source_filter=source_filter,
        sentiment_filter=sentiment_filter,
        content_type_filter=content_type_filter,
        topic_filter=topic_filter,
        search_query=search_query,
        date_range=date_range,
    )

    st.markdown(f"### Results ({len(messages)} messages)")

    if messages.empty:
        st.info("No messages matched your filters.")
        return

    for _, row in messages.iterrows():
        sentiment = (row.get("sentiment") or "neutral").lower()
        allowed_sentiments = {"positive", "negative", "neutral"}
        sentiment_value = sentiment if sentiment in allowed_sentiments else "neutral"
        sentiment_class = f"sentiment-{sentiment_value}"

        content = escape(str(row.get("content") or ""))
        author = escape(str(row.get("author") or "unknown"))
        source = escape(_display_source(str(row.get("source") or "unknown")))
        created_at = row.get("created_at")
        created_text = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "N/A"
        content_type = escape(str(row.get("content_type") or "discussion"))
        url = escape(str(row.get("url") or "#"))

        topics_html = "".join(
            [
                (
                    "<span style=\"background: #16213e; padding: 2px 8px; "
                    "border-radius: 8px; font-size: 0.75rem; color: #00d4ff;\">"
                    f"{escape(topic)}</span>"
                )
                for topic in row.get("topic_list", [])
            ]
        )

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
                        border-radius: 12px; padding: 16px; margin: 8px 0;
                        border: 1px solid #00d4ff22;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="color: #00d4ff; font-weight: bold;">{source}</span>
                        <span style="color: #666; margin-left: 8px;">@{author}</span>
                        <span style="color: #666; margin-left: 8px;">{created_text}</span>
                    </div>
                    <div>
                        <span class="{sentiment_class}">{escape(sentiment)}</span>
                        <span style="background: #2d4a6f; padding: 4px 8px; border-radius: 12px;
                                     margin-left: 4px; font-size: 0.8rem; color: #8b9dc3;">
                            {content_type}
                        </span>
                    </div>
                </div>
                <p style="color: #e0e0e0; margin: 8px 0;">{content}</p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">{topics_html}</div>
                <a href="{url}" target="_blank" style="color: #00d4ff; font-size: 0.8rem; text-decoration: none;">
                    View original →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
