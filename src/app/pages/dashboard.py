"""Sentiment dashboard page focused on voice-of-customer sources."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import KNOWLEDGE_SOURCES

DUCKDB_PATH = "/data/processed/sentiment.duckdb"


def _voice_where_clause(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    placeholders = ", ".join(["?"] * len(KNOWLEDGE_SOURCES))
    return f"{prefix}source NOT IN ({placeholders})"


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

    sentiment = row.get("sentiment_simple")
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

    metadata = _parse_metadata(row.get("metadata"))
    comments = _safe_int(metadata.get("num_comments") or metadata.get("comments"))
    score += min(comments / 5.0, 3.0)

    points = _safe_int(metadata.get("points") or metadata.get("score"))
    score += min(points / 20.0, 3.0)

    created_at = row.get("created_at")
    if created_at:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_days = max(
            (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0,
            0.0,
        )
        score += max(0.0, 2.0 - age_days / 7.0)

    return round(score, 2)


def _load_dashboard_data() -> dict | None:
    if not Path(DUCKDB_PATH).exists():
        return None

    voice_where = _voice_where_clause()
    voice_params = list(KNOWLEDGE_SOURCES)

    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)

        stats_result = conn.execute(
            f"""
            SELECT
                COUNT(*) as total_messages,
                COUNT(*) FILTER (WHERE sentiment_simple = 'positive') as positive,
                COUNT(*) FILTER (WHERE sentiment_simple = 'negative') as negative,
                COUNT(*) FILTER (WHERE sentiment_simple = 'neutral') as neutral,
                COUNT(*) FILTER (WHERE content_type = 'bug_report') as bug_reports,
                COUNT(*) FILTER (WHERE content_type = 'feature_request') as feature_requests,
                COUNT(*) FILTER (WHERE content_type = 'question') as questions
            FROM messages
            WHERE {voice_where}
            """,
            voice_params,
        ).fetchone()

        stats = {
            "total_messages": stats_result[0] or 0,
            "positive": stats_result[1] or 0,
            "negative": stats_result[2] or 0,
            "neutral": stats_result[3] or 0,
            "bug_reports": stats_result[4] or 0,
            "feature_requests": stats_result[5] or 0,
            "questions": stats_result[6] or 0,
        }

        # Use 30-day window for better signal with sparse data
        unanswered_questions = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages m
            WHERE {_voice_where_clause('m')}
              AND m.content_type = 'question'
              AND m.created_at >= CURRENT_DATE - INTERVAL '30 days'
              AND NOT EXISTS (
                  SELECT 1
                  FROM messages r
                  WHERE r.parent_id = m.id OR r.thread_id = m.id
              )
            """,
            voice_params,
        ).fetchone()[0]

        new_bug_reports = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages
            WHERE {voice_where}
              AND content_type = 'bug_report'
              AND created_at >= CURRENT_DATE - INTERVAL '30 days'
            """,
            voice_params,
        ).fetchone()[0]

        sentiment_result = conn.execute(
            f"""
            WITH daily AS (
                SELECT
                    DATE_TRUNC('day', created_at)::DATE as date,
                    COUNT(*) FILTER (WHERE sentiment_simple = 'positive') as positive,
                    COUNT(*) FILTER (WHERE sentiment_simple = 'negative') as negative,
                    COUNT(*) FILTER (WHERE sentiment_simple = 'neutral') as neutral
                FROM messages
                WHERE {voice_where}
                  AND created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY date
            )
            SELECT
                date,
                AVG(positive) OVER (
                    ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as positive,
                AVG(negative) OVER (
                    ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as negative,
                AVG(neutral) OVER (
                    ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as neutral
            FROM daily
            """,
            voice_params,
        ).fetchall()

        sentiment_data = (
            pd.DataFrame(
                sentiment_result,
                columns=["date", "positive", "negative", "neutral"],
            )
            if sentiment_result
            else pd.DataFrame({"date": [], "positive": [], "negative": [], "neutral": []})
        )

        topics_result = conn.execute(
            f"""
            SELECT topic, COUNT(*) as count
            FROM messages m,
            UNNEST(from_json(m.topics, '["VARCHAR"]')) AS t(topic)
            WHERE {_voice_where_clause('m')}
              AND m.topics IS NOT NULL
              AND m.topics != '[]'
            GROUP BY topic
            ORDER BY count DESC
            LIMIT 10
            """,
            voice_params,
        ).fetchall()

        topics = (
            pd.DataFrame(topics_result, columns=["topic", "count"])
            if topics_result
            else pd.DataFrame({"topic": ["No topics yet"], "count": [0]})
        )

        sources_result = conn.execute(
            f"""
            SELECT source, COUNT(*) as count
            FROM messages
            WHERE {voice_where}
            GROUP BY source
            ORDER BY count DESC
            """,
            voice_params,
        ).fetchall()

        sources = (
            pd.DataFrame(sources_result, columns=["source", "count"])
            if sources_result
            else pd.DataFrame({"source": ["No data"], "count": [0]})
        )

        pain_points_result = conn.execute(
            f"""
            SELECT
                topic,
                COUNT(*) AS mentions,
                COUNT(*) FILTER (WHERE m.sentiment_simple = 'negative') AS negative_mentions,
                COUNT(*) FILTER (
                    WHERE m.sentiment_rich IN ('frustration', 'confusion', 'complaint')
                ) AS friction_mentions
            FROM messages m,
            UNNEST(from_json(m.topics, '["VARCHAR"]')) AS t(topic)
            WHERE {_voice_where_clause('m')}
              AND m.created_at >= CURRENT_DATE - INTERVAL '30 days'
              AND m.topics IS NOT NULL
              AND m.topics != '[]'
            GROUP BY topic
            ORDER BY (negative_mentions + friction_mentions) DESC, mentions DESC
            LIMIT 8
            """,
            voice_params,
        ).fetchall()

        pain_points = (
            pd.DataFrame(
                pain_points_result,
                columns=[
                    "topic",
                    "mentions",
                    "negative_mentions",
                    "friction_mentions",
                ],
            )
            if pain_points_result
            else pd.DataFrame(
                {
                    "topic": [],
                    "mentions": [],
                    "negative_mentions": [],
                    "friction_mentions": [],
                }
            )
        )

        # Get high-impact mentions, excluding neutral sentiment
        high_impact_rows = conn.execute(
            f"""
            SELECT
                id,
                title,
                content,
                source,
                created_at,
                sentiment_simple,
                sentiment_rich,
                content_type,
                metadata,
                url
            FROM messages
            WHERE {voice_where}
              AND created_at >= CURRENT_DATE - INTERVAL '30 days'
              AND (sentiment_simple != 'neutral' OR sentiment_simple IS NULL)
            ORDER BY created_at DESC
            LIMIT 300
            """,
            voice_params,
        ).fetchall()

        high_impact_columns = [d[0] for d in conn.description]
        high_impact = [dict(zip(high_impact_columns, row)) for row in high_impact_rows]
        for item in high_impact:
            item["impact_score"] = _impact_score(item)

        high_impact = sorted(
            high_impact,
            key=lambda item: (
                item.get("impact_score", 0),
                item.get("created_at") or datetime.min,
            ),
            reverse=True,
        )[:8]

        recent_result = conn.execute(
            f"""
            SELECT title, source, sentiment_simple, created_at
            FROM messages
            WHERE {voice_where}
              AND title IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 6
            """,
            voice_params,
        ).fetchall()

        # Get uncategorized messages (topic = 'other' or no topics assigned)
        uncategorized_rows = conn.execute(
            f"""
            SELECT
                id,
                title,
                content,
                source,
                created_at,
                sentiment_simple,
                url
            FROM messages m
            WHERE {_voice_where_clause('m')}
              AND m.created_at >= CURRENT_DATE - INTERVAL '30 days'
              AND (
                  m.topics IS NULL
                  OR m.topics = '[]'
                  OR m.topics = '["other"]'
                  OR EXISTS (
                      SELECT 1
                      FROM UNNEST(from_json(m.topics, '["VARCHAR"]')) AS t(topic)
                      WHERE t.topic = 'other'
                  )
              )
            ORDER BY m.created_at DESC
            LIMIT 20
            """,
            voice_params,
        ).fetchall()

        uncategorized_cols = [d[0] for d in conn.description]
        uncategorized = [dict(zip(uncategorized_cols, row)) for row in uncategorized_rows]

        conn.close()

        return {
            "stats": stats,
            "sentiment_data": sentiment_data,
            "topics": topics,
            "sources": sources,
            "pain_points": pain_points,
            "unanswered_questions": unanswered_questions or 0,
            "new_bug_reports": new_bug_reports or 0,
            "high_impact": high_impact,
            "recent_messages": recent_result,
            "uncategorized": uncategorized,
        }
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return None


def _fallback_data() -> dict:
    return {
        "stats": {
            "total_messages": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "bug_reports": 0,
            "feature_requests": 0,
            "questions": 0,
        },
        "sentiment_data": pd.DataFrame({"date": [], "positive": [], "negative": [], "neutral": []}),
        "topics": pd.DataFrame({"topic": ["No data"], "count": [0]}),
        "sources": pd.DataFrame({"source": ["No data"], "count": [0]}),
        "uncategorized": [],
        "pain_points": pd.DataFrame(
            {"topic": [], "mentions": [], "negative_mentions": [], "friction_mentions": []}
        ),
        "unanswered_questions": 0,
        "new_bug_reports": 0,
        "high_impact": [],
        "recent_messages": [],
    }


def render_dashboard():
    """Render the main dashboard."""
    st.title("Social Sentiment Dashboard")
    st.caption("Voice-of-customer analytics (docs/blog excluded from KPI calculations)")

    data = _load_dashboard_data() or _fallback_data()
    stats = data["stats"]

    st.markdown("### Key Metrics (Voice Sources)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_messages']:,}</div>
                <div class="metric-label">Total Mentions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        positive_pct = (
            (stats["positive"] / stats["total_messages"] * 100)
            if stats["total_messages"]
            else 0.0
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #00c853;">{positive_pct:.1f}%</div>
                <div class="metric-label">Positive Sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ff9800;">
                    {data['unanswered_questions']:,}
                </div>
                <div class="metric-label">Unanswered Questions (30d)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ff5252;">
                    {data['new_bug_reports']:,}
                </div>
                <div class="metric-label">New Bug Reports (30d)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    sentiment_data = data["sentiment_data"]
    topics = data["topics"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sentiment Trends")

        num_days = len(sentiment_data)

        if num_days >= 7:
            # Full line chart for 7+ days
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data["date"],
                    y=sentiment_data["positive"],
                    name="Positive",
                    fill="tonexty",
                    line=dict(color="#00c853", width=2),
                    fillcolor="rgba(0, 200, 83, 0.1)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data["date"],
                    y=sentiment_data["neutral"],
                    name="Neutral",
                    fill="tonexty",
                    line=dict(color="#78909c", width=2),
                    fillcolor="rgba(120, 144, 156, 0.1)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data["date"],
                    y=sentiment_data["negative"],
                    name="Negative",
                    fill="tozeroy",
                    line=dict(color="#ff5252", width=2),
                    fillcolor="rgba(255, 82, 82, 0.1)",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            )

            st.plotly_chart(fig, use_container_width=True)

        elif num_days >= 1:
            # Mini bar chart for 1-6 days (not enough for trend line)
            st.caption(f"📊 {num_days} day(s) of data — showing daily breakdown")

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=sentiment_data["date"],
                    y=sentiment_data["positive"],
                    name="Positive",
                    marker_color="#00c853",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=sentiment_data["date"],
                    y=sentiment_data["neutral"],
                    name="Neutral",
                    marker_color="#78909c",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=sentiment_data["date"],
                    y=sentiment_data["negative"],
                    name="Negative",
                    marker_color="#ff5252",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.1),
                barmode="group",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                height=200,
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("📊 No sentiment data yet. Run ingestion and sentiment analysis.")

    with col2:
        st.markdown("### Top Topics")

        fig = px.bar(
            topics,
            x="count",
            y="topic",
            orientation="h",
            color="count",
            color_continuous_scale=["#0f2744", "#00d4ff"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=False),
        )

        st.plotly_chart(fig, use_container_width=True)

    sources = data["sources"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Source Distribution")

        fig = px.pie(
            sources,
            values="count",
            names="source",
            hole=0.6,
            color_discrete_sequence=[
                "#00d4ff",
                "#00ff88",
                "#ff9800",
                "#ff5252",
                "#9c27b0",
            ],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Sentiment Breakdown")

        sentiment_breakdown = pd.DataFrame(
            {
                "sentiment": ["Positive", "Negative", "Neutral"],
                "count": [stats["positive"], stats["negative"], stats["neutral"]],
            }
        )

        fig = px.bar(
            sentiment_breakdown,
            x="sentiment",
            y="count",
            color="sentiment",
            color_discrete_map={
                "Positive": "#00c853",
                "Negative": "#ff5252",
                "Neutral": "#78909c",
            },
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Pain Points (30 Days)")
    pain_points = data["pain_points"]
    if pain_points.empty:
        st.info("No clear pain points detected in the last 30 days.")
    else:
        st.dataframe(
            pain_points.rename(
                columns={
                    "topic": "Topic",
                    "mentions": "Mentions",
                    "negative_mentions": "Negative",
                    "friction_mentions": "Friction",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### High-Impact Mentions")
    high_impact = data["high_impact"]
    if not high_impact:
        st.info("No high-impact mentions found yet.")
    else:
        for item in high_impact[:6]:
            # Escape user content to prevent HTML injection
            raw_title = item.get("title") or item.get("content", "")[:140]
            title = html.escape(raw_title[:140])
            source = html.escape(str(item.get("source", "unknown")))
            score = item.get("impact_score", 0)
            sentiment = html.escape(str(item.get("sentiment_simple") or "unknown"))
            created_at = item.get("created_at")
            created_text = created_at.strftime("%Y-%m-%d") if created_at else "N/A"
            # Validate URL to prevent javascript: injection
            raw_url = item.get("url") or ""
            url = raw_url if raw_url.startswith(("http://", "https://")) else "#"

            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 10px;">
                    <div style="display:flex; justify-content:space-between; gap:12px;">
                        <div>
                            <strong>{title}</strong><br>
                            <small style="color:#8b9dc3;">{source} • {created_text}</small>
                        </div>
                        <div style="text-align:right; min-width:120px;">
                            <span style="font-weight:700; color:#00d4ff;">impact {score:.1f}</span><br>
                            <small style="color:#8b9dc3;">{sentiment}</small>
                        </div>
                    </div>
                    <a href="{url}" target="_blank" style="color:#00d4ff;">Open source →</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Recent Voice Messages")
    recent_messages = data["recent_messages"]
    if not recent_messages:
        st.info("No recent messages. Run ingestion to populate data.")
    else:
        for title, source, sentiment, created in recent_messages:
            sentiment_icon = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "⚪",
            }.get(sentiment, "⚪")
            snippet = (title or "Untitled")[:70]
            created_text = created.strftime("%b %d") if created else "N/A"
            st.markdown(f"{sentiment_icon} **{snippet}** `{source}` - {created_text}")

    # Uncategorized topics section
    uncategorized = data.get("uncategorized", [])
    if uncategorized:
        st.markdown("---")
        st.markdown("### 🏷️ Uncategorized Mentions")
        st.caption(
            "These messages have no clear topic or are tagged as 'other'. "
            "Review them to identify emerging patterns or new topic categories."
        )

        with st.expander(f"View {len(uncategorized)} uncategorized mentions", expanded=False):
            for item in uncategorized:
                # Escape user content to prevent HTML injection
                raw_title = item.get("title") or item.get("content", "")[:100]
                title = html.escape(raw_title[:120])
                source = html.escape(str(item.get("source", "unknown")))
                sentiment = html.escape(str(item.get("sentiment_simple") or "unknown"))
                created_at = item.get("created_at")
                created_text = created_at.strftime("%Y-%m-%d") if created_at else "N/A"
                # Validate URL to prevent javascript: injection
                raw_url = item.get("url") or ""
                url = raw_url if raw_url.startswith(("http://", "https://")) else "#"

                sentiment_color = {
                    "positive": "#00c853",
                    "negative": "#ff5252",
                    "neutral": "#78909c",
                }.get(item.get("sentiment_simple"), "#78909c")

                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom: 8px; padding: 12px;">
                        <div style="display:flex; justify-content:space-between; gap:12px;">
                            <div style="flex: 1;">
                                <strong>{title}</strong><br>
                                <small style="color:#8b9dc3;">{source} • {created_text}</small>
                            </div>
                            <div style="text-align:right; min-width:80px;">
                                <span style="color:{sentiment_color};">{sentiment}</span>
                            </div>
                        </div>
                        <a href="{url}" target="_blank" style="color:#00d4ff; font-size: 12px;">
                            View original →
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.info(
            "💡 **Tip:** Common patterns in uncategorized mentions may indicate new topics "
            "to add to the classifier. Consider running sentiment analysis to auto-tag these."
        )
