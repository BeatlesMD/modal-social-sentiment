"""
Admin page for job management and system status.
"""

import streamlit as st
from datetime import datetime


def render_admin():
    """Render the admin page."""
    st.title("⚙️ Admin Panel")
    st.caption("Manage ingestion jobs and monitor system status")
    
    # System status
    st.markdown("### System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #00c853; font-size: 24px;">●</span>
                <span style="color: #e0e0e0;">Ingestion Jobs</span>
            </div>
            <small style="color: #8b9dc3;">All scheduled jobs running</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #00c853; font-size: 24px;">●</span>
                <span style="color: #e0e0e0;">Inference Service</span>
            </div>
            <small style="color: #8b9dc3;">Model loaded, accepting requests</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #ff9800; font-size: 24px;">●</span>
                <span style="color: #e0e0e0;">Fine-tuning</span>
            </div>
            <small style="color: #8b9dc3;">Not running</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Ingestion Jobs
    st.markdown("### Ingestion Jobs")
    
    jobs = [
        {"name": "Docs Ingester", "schedule": "Every 6 hours", "last_run": datetime.now(), "status": "success", "messages": 156},
        {"name": "GitHub Ingester", "schedule": "Every 4 hours", "last_run": datetime.now(), "status": "success", "messages": 89},
        {"name": "Reddit Ingester", "schedule": "Every 4 hours", "last_run": datetime.now(), "status": "success", "messages": 234},
        {"name": "HackerNews Ingester", "schedule": "Every 4 hours", "last_run": datetime.now(), "status": "warning", "messages": 12},
        {"name": "Twitter Ingester", "schedule": "Every 4 hours", "last_run": datetime.now(), "status": "error", "messages": 0},
    ]
    
    for job in jobs:
        status_color = {
            "success": "#00c853",
            "warning": "#ff9800",
            "error": "#ff5252",
        }.get(job["status"], "#666")
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: {status_color}; font-size: 16px;">●</span>
                <strong style="color: #e0e0e0;">{job['name']}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.caption(job["schedule"])
        
        with col3:
            st.caption(f"Last: {job['last_run'].strftime('%H:%M')} ({job['messages']} msgs)")
        
        with col4:
            if st.button("Run", key=f"run_{job['name']}", type="secondary"):
                st.info(f"Would trigger {job['name']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Processing Jobs
    st.markdown("### Processing Jobs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #16213e; border-radius: 8px; padding: 16px;">
            <strong style="color: #00d4ff;">Embedding Generation</strong>
            <p style="color: #8b9dc3; margin: 8px 0;">Generate embeddings for new messages</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Embeddings", use_container_width=True):
            st.info("Would trigger embedding generation")
    
    with col2:
        st.markdown("""
        <div style="background: #16213e; border-radius: 8px; padding: 16px;">
            <strong style="color: #00d4ff;">Sentiment Analysis</strong>
            <p style="color: #8b9dc3; margin: 8px 0;">Analyze sentiment of unprocessed messages</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Sentiment", use_container_width=True):
            st.info("Would trigger sentiment analysis")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fine-tuning
    st.markdown("### Fine-tuning")
    
    with st.expander("🎯 Training Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            base_model = st.selectbox(
                "Base Model",
                ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
            )
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                value=2e-4,
                format_func=lambda x: f"{x:.0e}",
            )
        
        with col2:
            lora_r = st.number_input("LoRA Rank", min_value=8, max_value=128, value=64)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=16, value=4)
            max_seq_length = st.number_input("Max Seq Length", min_value=512, max_value=4096, value=2048)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📦 Prepare Data", use_container_width=True):
                st.info("Would prepare training dataset")
        
        with col2:
            if st.button("🚀 Start Training", use_container_width=True, type="primary"):
                st.info("Would start QLoRA fine-tuning on A100")
        
        with col3:
            if st.button("🔗 Merge Weights", use_container_width=True):
                st.info("Would merge LoRA weights")
    
    # Database stats
    st.markdown("### Database Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", "15,847")
    with col2:
        st.metric("Processed", "14,523")
    with col3:
        st.metric("Embeddings", "14,523")
    with col4:
        st.metric("Training Examples", "3,241")
