"""
Message explorer page.

Allows filtering and searching through collected messages.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def get_mock_messages():
    """Generate mock messages for development."""
    messages = [
        {
            "id": "1",
            "source": "GitHub",
            "content": "I'm having trouble with Modal volumes not persisting data between function calls. Is this expected behavior?",
            "author": "user123",
            "created_at": datetime.now() - timedelta(hours=2),
            "sentiment": "negative",
            "sentiment_rich": "confusion",
            "content_type": "question",
            "topics": ["volumes", "reliability"],
            "url": "https://github.com/modal-labs/modal-client/issues/123",
        },
        {
            "id": "2",
            "source": "Twitter",
            "content": "Just deployed my first ML model on Modal in 5 minutes. The DX is incredible! 🚀",
            "author": "@mldev",
            "created_at": datetime.now() - timedelta(hours=5),
            "sentiment": "positive",
            "sentiment_rich": "delight",
            "content_type": "praise",
            "topics": ["ease_of_use", "performance"],
            "url": "https://twitter.com/mldev/status/123",
        },
        {
            "id": "3",
            "source": "Reddit",
            "content": "How does Modal pricing compare to AWS Lambda for GPU workloads? Looking to migrate our inference pipeline.",
            "author": "clouddev99",
            "created_at": datetime.now() - timedelta(hours=8),
            "sentiment": "neutral",
            "sentiment_rich": "curiosity",
            "content_type": "question",
            "topics": ["pricing", "gpu_availability"],
            "url": "https://reddit.com/r/MachineLearning/comments/abc123",
        },
        {
            "id": "4",
            "source": "HackerNews",
            "content": "Modal's approach to serverless GPUs is really interesting. No cold starts for A100s is a game changer for real-time inference.",
            "author": "hnuser",
            "created_at": datetime.now() - timedelta(hours=12),
            "sentiment": "positive",
            "sentiment_rich": "delight",
            "content_type": "discussion",
            "topics": ["gpu_availability", "performance"],
            "url": "https://news.ycombinator.com/item?id=123",
        },
        {
            "id": "5",
            "source": "GitHub",
            "content": "Feature request: It would be great to have built-in support for streaming responses in web endpoints.",
            "author": "streamfan",
            "created_at": datetime.now() - timedelta(days=1),
            "sentiment": "neutral",
            "sentiment_rich": "curiosity",
            "content_type": "feature_request",
            "topics": ["web_endpoints", "functions"],
            "url": "https://github.com/modal-labs/modal-client/issues/456",
        },
    ]
    return pd.DataFrame(messages)


def render_explorer():
    """Render the message explorer."""
    st.title("🔍 Message Explorer")
    st.caption("Search and filter through community messages")
    
    # Filters
    with st.expander("🎛️ Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            source_filter = st.multiselect(
                "Source",
                ["GitHub", "Twitter", "Reddit", "HackerNews", "Docs"],
                default=[],
            )
        
        with col2:
            sentiment_filter = st.multiselect(
                "Sentiment",
                ["positive", "negative", "neutral"],
                default=[],
            )
        
        with col3:
            content_type_filter = st.multiselect(
                "Content Type",
                ["question", "bug_report", "feature_request", "praise", "discussion"],
                default=[],
            )
        
        with col4:
            topic_filter = st.multiselect(
                "Topics",
                ["gpu_availability", "pricing", "performance", "documentation", 
                 "ease_of_use", "reliability", "volumes", "functions", "web_endpoints"],
                default=[],
            )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("🔎 Search", placeholder="Search messages...")
        with col2:
            date_range = st.selectbox(
                "Time Range",
                ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
            )
    
    # Get and filter data
    df = get_mock_messages()
    
    if source_filter:
        df = df[df['source'].isin(source_filter)]
    if sentiment_filter:
        df = df[df['sentiment'].isin(sentiment_filter)]
    if content_type_filter:
        df = df[df['content_type'].isin(content_type_filter)]
    if search_query:
        df = df[df['content'].str.contains(search_query, case=False, na=False)]
    
    # Display results
    st.markdown(f"### Results ({len(df)} messages)")
    
    for _, row in df.iterrows():
        # Sentiment badge
        sentiment_class = f"sentiment-{row['sentiment']}"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%); 
                    border-radius: 12px; padding: 16px; margin: 8px 0;
                    border: 1px solid #00d4ff22;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span style="color: #00d4ff; font-weight: bold;">{row['source']}</span>
                    <span style="color: #666; margin-left: 8px;">@{row['author']}</span>
                    <span style="color: #666; margin-left: 8px;">{row['created_at'].strftime('%Y-%m-%d %H:%M')}</span>
                </div>
                <div>
                    <span class="{sentiment_class}">{row['sentiment']}</span>
                    <span style="background: #2d4a6f; padding: 4px 8px; border-radius: 12px; 
                                 margin-left: 4px; font-size: 0.8rem; color: #8b9dc3;">
                        {row['content_type']}
                    </span>
                </div>
            </div>
            <p style="color: #e0e0e0; margin: 8px 0;">{row['content']}</p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                {''.join([f'<span style="background: #16213e; padding: 2px 8px; border-radius: 8px; font-size: 0.75rem; color: #00d4ff;">{t}</span>' for t in row['topics']])}
            </div>
            <a href="{row['url']}" target="_blank" style="color: #00d4ff; font-size: 0.8rem; text-decoration: none;">
                View original →
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Pagination placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Load More", use_container_width=True)
