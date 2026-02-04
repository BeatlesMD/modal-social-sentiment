"""
Modal Support Assistant page.

Chat interface for asking questions about Modal.
"""

import streamlit as st


def render_assistant():
    """Render the assistant chat interface."""
    st.title("🤖 Modal Support Assistant")
    st.caption("Ask questions about Modal - powered by fine-tuned Qwen3-4B + RAG")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm your Modal support assistant. I can help you with questions about Modal's features, best practices, and troubleshooting. What would you like to know?",
            }
        ]
    
    # Settings sidebar
    with st.sidebar:
        st.markdown("### Assistant Settings")
        use_rag = st.toggle("Use RAG (recommended)", value=True)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max tokens", 128, 1024, 512, 64)
        
        st.markdown("---")
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{source.get('source', 'Unknown')}</strong><br>
                            <small>{source.get('text', '')[:150]}...</small><br>
                            <a href="{source.get('url', '#')}" target="_blank">View →</a>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about Modal..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # In production, this would call the Modal inference service
                # response = call_assistant_service(prompt, use_rag, temperature, max_tokens)
                
                # Mock response for development
                response = generate_mock_response(prompt, use_rag)
                
                st.markdown(response["answer"])
                
                if response.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in response["sources"]:
                            st.markdown(f"""
                            <div style="background: #16213e; border-radius: 8px; padding: 12px; margin: 4px 0;">
                                <strong style="color: #00d4ff;">{source.get('source', 'Unknown')}</strong><br>
                                <small style="color: #8b9dc3;">{source.get('text', '')[:150]}...</small><br>
                                <a href="{source.get('url', '#')}" target="_blank" style="color: #00ff88;">View →</a>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
        })
    
    # Example questions
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    
    examples = [
        "How do I create a Modal function with GPU access?",
        "What's the difference between @app.function and @app.cls?",
        "How do I persist data between function calls?",
        "How do I deploy a web endpoint with Modal?",
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Trigger the chat input programmatically
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


def generate_mock_response(question: str, use_rag: bool) -> dict:
    """Generate a mock response for development."""
    # Simple keyword-based responses for demo
    question_lower = question.lower()
    
    if "gpu" in question_lower:
        answer = """To create a Modal function with GPU access, use the `gpu` parameter in the `@app.function` decorator:

```python
import modal

app = modal.App()

@app.function(gpu="A10G")  # or "T4", "A100", "H100"
def my_gpu_function():
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    return torch.cuda.get_device_name(0)
```

Available GPU options:
- `"T4"` - Entry-level, good for inference
- `"A10G"` - Great balance of cost/performance
- `"A100"` - High-end, best for training
- `"H100"` - Latest generation, maximum performance

You can also request specific GPU memory with `gpu="A100-80GB"`."""
        
        sources = [
            {"source": "Docs", "text": "Modal supports NVIDIA GPUs including T4, A10G, A100, and H100...", "url": "https://modal.com/docs/guide/gpu"},
        ]
    
    elif "volume" in question_lower or "persist" in question_lower:
        answer = """To persist data between function calls, use Modal Volumes:

```python
import modal

app = modal.App()
volume = modal.Volume.from_name("my-volume", create_if_missing=True)

@app.function(volumes={"/data": volume})
def save_data():
    with open("/data/myfile.txt", "w") as f:
        f.write("Hello, World!")
    volume.commit()  # Important: commit changes!

@app.function(volumes={"/data": volume})
def read_data():
    with open("/data/myfile.txt", "r") as f:
        return f.read()
```

Key points:
- Use `volume.commit()` to persist changes
- Volumes are shared across all functions that mount them
- Data persists even after functions terminate"""
        
        sources = [
            {"source": "Docs", "text": "Modal Volumes provide persistent, shared storage across functions...", "url": "https://modal.com/docs/guide/volumes"},
        ]
    
    elif "web" in question_lower or "endpoint" in question_lower:
        answer = """To deploy a web endpoint with Modal, use the `@modal.web_endpoint` decorator:

```python
import modal

app = modal.App()

@app.function()
@modal.web_endpoint(method="GET")
def hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}

# For POST endpoints with request body:
@app.function()
@modal.web_endpoint(method="POST")
def process(request: dict):
    return {"received": request}
```

Deploy with `modal deploy app.py` and you'll get a URL like:
`https://your-app--hello.modal.run`

For ASGI apps (FastAPI, etc.), use `@modal.asgi_app()`."""
        
        sources = [
            {"source": "Docs", "text": "Web endpoints allow you to expose Modal functions as HTTP APIs...", "url": "https://modal.com/docs/guide/webhooks"},
        ]
    
    else:
        answer = f"""I'd be happy to help with your question about "{question[:50]}..."

For the most accurate answer, I recommend checking the Modal documentation at https://modal.com/docs or asking in the Modal Slack community.

Some helpful resources:
- [Modal Guide](https://modal.com/docs/guide) - Getting started
- [Modal Examples](https://modal.com/docs/examples) - Real-world use cases
- [Modal Reference](https://modal.com/docs/reference) - API documentation

Is there something more specific I can help you with?"""
        
        sources = []
    
    return {
        "answer": answer,
        "sources": sources if use_rag else [],
        "model": "mock",
    }
