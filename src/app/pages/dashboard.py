"""
Sentiment dashboard page.

Displays:
- Key metrics
- Sentiment trends over time
- Topic distribution
- Source breakdown
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

# Database path (same as in app.py)
DUCKDB_PATH = "/data/processed/sentiment.duckdb"


def get_real_data():
    """Get real data from DuckDB."""
    import duckdb
    
    if not Path(DUCKDB_PATH).exists():
        return None, None, None, None
    
    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        
        # Get summary stats
        stats_result = conn.execute("""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN sentiment_simple = 'positive' THEN 1 END) as positive,
                COUNT(CASE WHEN sentiment_simple = 'negative' THEN 1 END) as negative,
                COUNT(CASE WHEN sentiment_simple = 'neutral' THEN 1 END) as neutral,
                COUNT(CASE WHEN content_type = 'bug_report' THEN 1 END) as bug_reports,
                COUNT(CASE WHEN content_type = 'feature_request' THEN 1 END) as feature_requests,
                COUNT(CASE WHEN content_type = 'question' THEN 1 END) as questions
            FROM messages
        """).fetchone()
        
        stats = {
            "total_messages": stats_result[0] or 0,
            "positive": stats_result[1] or 0,
            "negative": stats_result[2] or 0,
            "neutral": stats_result[3] or 0,
            "bug_reports": stats_result[4] or 0,
            "feature_requests": stats_result[5] or 0,
            "questions": stats_result[6] or 0,
        }
        
        # Get sentiment trends over time (smoothed with 3-day rolling average)
        sentiment_result = conn.execute("""
            WITH daily AS (
                SELECT 
                    DATE_TRUNC('day', created_at)::DATE as date,
                    COUNT(CASE WHEN sentiment_simple = 'positive' THEN 1 END) as positive,
                    COUNT(CASE WHEN sentiment_simple = 'negative' THEN 1 END) as negative,
                    COUNT(CASE WHEN sentiment_simple = 'neutral' THEN 1 END) as neutral
                FROM messages
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY date
            )
            SELECT 
                date,
                AVG(positive) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as positive,
                AVG(negative) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as negative,
                AVG(neutral) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as neutral
            FROM daily
        """).fetchall()
        
        if sentiment_result:
            sentiment_data = pd.DataFrame(sentiment_result, columns=['date', 'positive', 'negative', 'neutral'])
        else:
            sentiment_data = pd.DataFrame({'date': [], 'positive': [], 'negative': [], 'neutral': []})
        
        # Get topic distribution (from topics JSON field)
        topics_result = conn.execute("""
            SELECT 
                topic,
                COUNT(*) as count
            FROM (
                SELECT unnest(from_json(topics, '["VARCHAR"]')) as topic
                FROM messages
                WHERE topics IS NOT NULL AND topics != '[]'
            ) t
            GROUP BY topic
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()
        
        if topics_result:
            topics = pd.DataFrame(topics_result, columns=['topic', 'count'])
        else:
            topics = pd.DataFrame({'topic': ['No topics yet'], 'count': [0]})
        
        # Get source distribution
        sources_result = conn.execute("""
            SELECT source, COUNT(*) as count
            FROM messages
            GROUP BY source
            ORDER BY count DESC
        """).fetchall()
        
        if sources_result:
            sources = pd.DataFrame(sources_result, columns=['source', 'count'])
        else:
            sources = pd.DataFrame({'source': ['No data'], 'count': [0]})
        
        conn.close()
        return stats, sentiment_data, topics, sources
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return None, None, None, None


def get_fallback_data():
    """Return empty data when DB is not available."""
    stats = {
        "total_messages": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "bug_reports": 0,
        "feature_requests": 0,
        "questions": 0,
    }
    sentiment_data = pd.DataFrame({'date': [], 'positive': [], 'negative': [], 'neutral': []})
    topics = pd.DataFrame({'topic': ['No data'], 'count': [0]})
    sources = pd.DataFrame({'source': ['No data'], 'count': [0]})
    return stats, sentiment_data, topics, sources


def render_dashboard():
    """Render the main dashboard."""
    st.title("📊 Social Sentiment Dashboard")
    st.caption("Real-time sentiment analysis across Modal's community channels")
    
    # Get real data from DuckDB
    result = get_real_data()
    if result[0] is None:
        stats, sentiment_data, topics, sources = get_fallback_data()
        st.warning("⚠️ Could not connect to database. Showing empty data.")
    else:
        stats, sentiment_data, topics, sources = result
    
    # Top metrics row
    st.markdown("### Key Metrics (Last 30 Days)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_messages']:,}</div>
            <div class="metric-label">Total Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        positive_pct = stats['positive'] / stats['total_messages'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #00c853;">{positive_pct:.1f}%</div>
            <div class="metric-label">Positive Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ff9800;">{stats['feature_requests']}</div>
            <div class="metric-label">Feature Requests</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ff5252;">{stats['bug_reports']}</div>
            <div class="metric-label">Bug Reports</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Trends")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sentiment_data['date'],
            y=sentiment_data['positive'],
            name='Positive',
            fill='tonexty',
            line=dict(color='#00c853', width=2),
            fillcolor='rgba(0, 200, 83, 0.1)',
        ))
        
        fig.add_trace(go.Scatter(
            x=sentiment_data['date'],
            y=sentiment_data['neutral'],
            name='Neutral',
            fill='tonexty',
            line=dict(color='#78909c', width=2),
            fillcolor='rgba(120, 144, 156, 0.1)',
        ))
        
        fig.add_trace(go.Scatter(
            x=sentiment_data['date'],
            y=sentiment_data['negative'],
            name='Negative',
            fill='tozeroy',
            line=dict(color='#ff5252', width=2),
            fillcolor='rgba(255, 82, 82, 0.1)',
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation='h', y=1.1),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top Topics")
        
        fig = px.bar(
            topics,
            x='count',
            y='topic',
            orientation='h',
            color='count',
            color_continuous_scale=['#0f2744', '#00d4ff'],
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=False),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Source Distribution")
        
        fig = px.pie(
            sources,
            values='count',
            names='source',
            hole=0.6,
            color_discrete_sequence=['#00d4ff', '#00ff88', '#ff9800', '#ff5252', '#9c27b0'],
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Sentiment Breakdown")
        
        sentiment_breakdown = pd.DataFrame({
            'sentiment': ['Positive', 'Negative', 'Neutral'],
            'count': [stats['positive'], stats['negative'], stats['neutral']],
            'color': ['#00c853', '#ff5252', '#78909c'],
        })
        
        fig = px.bar(
            sentiment_breakdown,
            x='sentiment',
            y='count',
            color='sentiment',
            color_discrete_map={
                'Positive': '#00c853',
                'Negative': '#ff5252',
                'Neutral': '#78909c',
            },
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("### Recent Messages")
    
    # Show recent messages from DB
    if stats["total_messages"] > 0:
        try:
            import duckdb
            conn = duckdb.connect(DUCKDB_PATH, read_only=True)
            recent = conn.execute("""
                SELECT title, source, sentiment_simple, created_at 
                FROM messages 
                WHERE title IS NOT NULL
                ORDER BY created_at DESC 
                LIMIT 5
            """).fetchall()
            conn.close()
            
            for title, source, sentiment, created in recent:
                sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sentiment, "⚪")
                st.markdown(f"{sentiment_color} **{title[:60]}...** `{source}` - {created.strftime('%b %d') if created else 'N/A'}")
        except Exception as e:
            st.caption(f"Could not load recent messages: {e}")
    else:
        st.info("📌 No messages yet. Run ingestion to populate data.")
