"""
Modal Support Assistant page.

Chat interface for asking questions about Modal.
"""

import os
import streamlit as st
import httpx

# Modal inference endpoint (deployed via modal deploy app.py)
ASSISTANT_ENDPOINT = os.environ.get(
    "ASSISTANT_ENDPOINT",
    "https://masondudas--modal-social-sentiment-ask.modal.run"
)


def call_assistant_service(question: str, use_rag: bool = True, timeout: float = 60.0) -> dict:
    """Call the Modal inference service to get a real answer."""
    try:
        response = httpx.post(
            ASSISTANT_ENDPOINT,
            json={"question": question, "use_rag": use_rag},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        return {
            "answer": "⏱️ The assistant took too long to respond. The GPU may be warming up - please try again in a moment.",
            "sources": [],
            "error": "timeout",
        }
    except httpx.HTTPStatusError as e:
        return {
            "answer": f"❌ Error from assistant service: {e.response.status_code}",
            "sources": [],
            "error": str(e),
        }
    except Exception as e:
        return {
            "answer": f"❌ Could not reach assistant service: {str(e)}",
            "sources": [],
            "error": str(e),
        }


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
    
    # Track if we need to generate a response (for example button clicks)
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    
    # Settings sidebar
    with st.sidebar:
        st.markdown("### Assistant Settings")
        
        use_live = st.toggle("🔴 Live Mode (Qwen + RAG)", value=True, 
                             help="Use the real Modal inference service with Qwen LLM and RAG")
        use_rag = st.toggle("Use RAG context", value=True, 
                            help="Retrieve relevant docs to augment responses")
        
        st.markdown("---")
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
        
        if use_live:
            st.success("🟢 Connected to Modal")
            st.caption(f"Endpoint: `...{ASSISTANT_ENDPOINT[-30:]}`")
        else:
            st.warning("🟡 Mock mode")
            st.caption("Using pre-built responses")
    
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
    
    # Handle pending question (from example buttons)
    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None  # Clear it
        process_question(pending, use_rag, use_live)
    
    # Chat input
    if prompt := st.chat_input("Ask about Modal..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        process_question(prompt, use_rag, use_live)
    
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
                # Add user message and set pending question for processing
                st.session_state.messages.append({"role": "user", "content": example})
                st.session_state.pending_question = example
                st.rerun()


def process_question(question: str, use_rag: bool, use_live: bool):
    """Process a question and generate a response."""
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        if use_live:
            with st.spinner("🧠 Thinking with Qwen + RAG..."):
                response = call_assistant_service(question, use_rag)
        else:
            with st.spinner("Generating mock response..."):
                response = generate_mock_response(question, use_rag)
        
        # Display the answer
        st.markdown(response["answer"])
        
        # Show error info if present
        if response.get("error"):
            st.error(f"Service error: {response['error']}")
        
        # Show model info badge
        if use_live and not response.get("error"):
            model_type = response.get("model", "unknown")
            used_rag = response.get("used_rag", False)
            st.caption(f"Model: `{model_type}` | RAG: {'✅' if used_rag else '❌'}")
        
        # Show sources if available
        if response.get("sources"):
            with st.expander(f"📚 Sources ({len(response['sources'])})"):
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
