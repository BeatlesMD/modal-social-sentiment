"""
Main Streamlit application for Modal Social Sentiment.

A social listening dashboard and support assistant interface.
"""

import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="Modal Social Sentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Dark theme with Modal-inspired colors */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #00d4ff33;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #8b9dc3;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sentiment badges */
    .sentiment-positive { 
        background: linear-gradient(135deg, #00c853 0%, #00a843 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
    }
    .sentiment-negative { 
        background: linear-gradient(135deg, #ff5252 0%, #d32f2f 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
    }
    .sentiment-neutral { 
        background: linear-gradient(135deg, #78909c 0%, #546e7a 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 3px solid #00d4ff;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #0f2744 0%, #1a3a5f 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 3px solid #00ff88;
    }
    
    /* Source cards */
    .source-card {
        background: #16213e;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        border: 1px solid #ffffff11;
        font-size: 0.85rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #0f0f1a;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00e5ff 0%, #00b8e6 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main app entry point."""
    # Sidebar navigation
    with st.sidebar:
        # Use text instead of potentially broken image
        st.markdown("## 🚀 Modal")
        st.title("Social Sentiment")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Explorer", "🤖 Assistant", "⚙️ Admin"],
            label_visibility="collapsed",
        )
        
        st.markdown("---")
        st.caption("Built on Modal 🚀")
    
    # Route to pages
    if page == "📊 Dashboard":
        from src.app.pages.dashboard import render_dashboard
        render_dashboard()
    elif page == "🔍 Explorer":
        from src.app.pages.explorer import render_explorer
        render_explorer()
    elif page == "🤖 Assistant":
        from src.app.pages.assistant import render_assistant
        render_assistant()
    elif page == "⚙️ Admin":
        from src.app.pages.admin import render_admin
        render_admin()


if __name__ == "__main__":
    main()
