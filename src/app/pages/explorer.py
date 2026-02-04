"""Message explorer with voice-first defaults and triage workflows."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from src.config import KNOWLEDGE_SOURCES

DUCKDB_PATH = "/data/processed/sentiment.duckdb"
DEFAULT_LIMIT = 800

DATE_FILTERS = {
    "Last 24 hours": 1,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "All time": None,
}

TRIAGE_FILTERS = [
    "Bug reports",
    "Feature requests",
    "Unanswered questions",
    "High impact",
]

SORT_OPTIONS = ["Newest", "Most negative", "Highest impact"]


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


def _parse_metadata(raw_value) -> dict:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _safe_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _impact_score(row: dict) -> float:
    score = 0.0
    sentiment = row.get("sentiment")
    sentiment_rich = row.get("sentiment_rich")
    content_type = row.get("content_type")

    if sentiment == "negative":
        score += 4.0
    elif sentiment == "neutral":
        score += 1.0

    if sentiment_rich in {"frustration", "confusion", "complaint"}:
        score += 2.0

    if content_type == "bug_report":
        score += 3.0
    elif content_type == "feature_request":
        score += 2.0
    elif content_type == "question":
        score += 1.0

    metadata = row.get("metadata_dict", {})
    comments = _safe_int(metadata.get("num_comments") or metadata.get("comments"))
    points = _safe_int(metadata.get("points") or metadata.get("score"))

    score += min(comments / 5.0, 3.0)
    score += min(points / 20.0, 3.0)

    return round(score, 2)


def _voice_filter_clause(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    placeholders = ", ".join(["?"] * len(KNOWLEDGE_SOURCES))
    return f"{prefix}source NOT IN ({placeholders})"


def _should_include_knowledge(source_filter: list[str]) -> bool:
    return bool(set(source_filter).intersection(KNOWLEDGE_SOURCES))


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
                FROM messages m,
                UNNEST(from_json(m.topics, '["VARCHAR"]')) AS t(topic)
                WHERE m.topics IS NOT NULL AND m.topics != '[]'
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
    triage_filters: list[str],
    sort_by: str,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    if not _db_available():
        return pd.DataFrame()

    conditions = []
    params: list[object] = []

    if not _should_include_knowledge(source_filter):
        conditions.append(_voice_filter_clause("m"))
        params.extend(KNOWLEDGE_SOURCES)

    if source_filter:
        placeholders = ", ".join(["?"] * len(source_filter))
        conditions.append(f"m.source IN ({placeholders})")
        params.extend(source_filter)

    if sentiment_filter:
        placeholders = ", ".join(["?"] * len(sentiment_filter))
        conditions.append(f"m.sentiment_simple IN ({placeholders})")
        params.extend(sentiment_filter)

    if content_type_filter:
        placeholders = ", ".join(["?"] * len(content_type_filter))
        conditions.append(f"m.content_type IN ({placeholders})")
        params.extend(content_type_filter)

    lookback_days = DATE_FILTERS.get(date_range)
    if lookback_days:
        conditions.append("m.created_at >= CURRENT_TIMESTAMP - (? * INTERVAL '1 day')")
        params.append(lookback_days)

    if search_query.strip():
        pattern = f"%{search_query.strip().lower()}%"
        conditions.append(
            "(LOWER(m.content) LIKE ? OR LOWER(COALESCE(m.title, '')) LIKE ?)"
        )
        params.extend([pattern, pattern])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            m.id,
            m.source,
            m.content,
            m.author,
            m.created_at,
            COALESCE(m.sentiment_simple, 'neutral') AS sentiment,
            m.sentiment_rich,
            COALESCE(m.content_type, 'discussion') AS content_type,
            m.topics,
            m.metadata,
            m.url,
            m.title,
            EXISTS(
                SELECT 1
                FROM messages r
                WHERE r.parent_id = m.id OR r.thread_id = m.id
            ) AS has_reply
        FROM messages m
        {where_clause}
        ORDER BY m.created_at DESC
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
            "metadata",
            "url",
            "title",
            "has_reply",
            "topic_list",
            "metadata_dict",
            "impact_score",
        ])

    df = pd.DataFrame(rows, columns=columns)
    df["topic_list"] = df["topics"].apply(_parse_topics)
    df["metadata_dict"] = df["metadata"].apply(_parse_metadata)
    df["impact_score"] = df.apply(lambda row: _impact_score(row.to_dict()), axis=1)

    if topic_filter:
        wanted = set(topic_filter)
        df = df[df["topic_list"].apply(lambda values: bool(wanted.intersection(values)))]

    if triage_filters:
        masks = []
        if "Bug reports" in triage_filters:
            masks.append(df["content_type"] == "bug_report")
        if "Feature requests" in triage_filters:
            masks.append(df["content_type"] == "feature_request")
        if "Unanswered questions" in triage_filters:
            masks.append((df["content_type"] == "question") & (~df["has_reply"]))
        if "High impact" in triage_filters:
            masks.append(df["impact_score"] >= 6.0)

        if masks:
            combined_mask = pd.concat(masks, axis=1).any(axis=1)
            df = df[combined_mask]

    if sort_by == "Most negative":
        sentiment_rank = {"negative": 0, "neutral": 1, "positive": 2}
        df["sentiment_rank"] = df["sentiment"].map(sentiment_rank).fillna(3)
        df = df.sort_values(
            by=["sentiment_rank", "impact_score", "created_at"],
            ascending=[True, False, False],
        )
    elif sort_by == "Highest impact":
        df = df.sort_values(by=["impact_score", "created_at"], ascending=[False, False])
    else:
        df = df.sort_values(by="created_at", ascending=False)

    return df


def _display_source(source: str) -> str:
    return source.replace("_", " ").title()


def _preview_text(content: str, length: int = 220) -> str:
    content = content.strip()
    if len(content) <= length:
        return content
    return f"{content[:length].rstrip()}..."


def render_explorer():
    """Render the message explorer."""
    st.title("Message Explorer")
    st.caption("Voice sources shown by default; choose docs/blog explicitly via Source filter")

    if not _db_available():
        st.warning("No database found at `/data/processed/sentiment.duckdb`. Run ingestion first.")
        return

    options = _get_filter_options()

    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            source_filter = st.multiselect("Source", options["sources"], default=[])

        with col2:
            sentiment_filter = st.multiselect("Sentiment", options["sentiments"], default=[])

        with col3:
            content_type_filter = st.multiselect(
                "Content Type", options["content_types"], default=[]
            )

        with col4:
            topic_filter = st.multiselect("Topics", options["topics"], default=[])

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("Search", placeholder="Search messages...")
        with col2:
            date_range = st.selectbox("Time Range", list(DATE_FILTERS.keys()), index=2)
        with col3:
            sort_by = st.selectbox("Sort", SORT_OPTIONS, index=0)

        triage_filters = st.multiselect("Quick Triage", TRIAGE_FILTERS, default=[])

    messages = _fetch_messages(
        source_filter=source_filter,
        sentiment_filter=sentiment_filter,
        content_type_filter=content_type_filter,
        topic_filter=topic_filter,
        search_query=search_query,
        date_range=date_range,
        triage_filters=triage_filters,
        sort_by=sort_by,
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

        content = str(row.get("content") or "")
        content_preview = escape(_preview_text(content))
        author = escape(str(row.get("author") or "unknown"))
        source = escape(_display_source(str(row.get("source") or "unknown")))
        created_at = row.get("created_at")
        created_text = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "N/A"
        content_type = escape(str(row.get("content_type") or "discussion"))
        url = escape(str(row.get("url") or "#"))
        impact_score = row.get("impact_score", 0)

        topic_badges = "".join(
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
                        border-radius: 12px; padding: 14px; margin: 8px 0;
                        border: 1px solid #00d4ff22;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
                    <div>
                        <span style="color:#00d4ff; font-weight:700;">{source}</span>
                        <span style="color:#8b9dc3; margin-left:8px;">@{author}</span>
                        <span style="color:#8b9dc3; margin-left:8px;">{created_text}</span>
                    </div>
                    <div>
                        <span style="background:#0f1c36; color:#00d4ff; padding:4px 8px; border-radius:12px; font-size:0.8rem;">
                            impact {impact_score:.1f}
                        </span>
                        <span class="{sentiment_class}" style="margin-left:4px;">{escape(sentiment)}</span>
                        <span style="background:#2d4a6f; padding:4px 8px; border-radius:12px;
                                     margin-left:4px; font-size:0.8rem; color:#8b9dc3;">
                            {content_type}
                        </span>
                    </div>
                </div>
                <p style="color:#e0e0e0; margin:10px 0 8px 0;">{content_preview}</p>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">{topic_badges}</div>
                <a href="{url}" target="_blank" style="color:#00d4ff; font-size:0.8rem; text-decoration:none;">
                    View original →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(content) > 220:
            with st.expander(f"Show details for {row.get('id', 'message')}"):
                st.write(content)
