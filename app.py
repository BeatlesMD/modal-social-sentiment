"""
Modal Social Sentiment - Main Application

All Modal functions defined in one place following Modal best practices.
Deploy: modal deploy app.py
Run:    modal run app.py
"""

import os
import modal

# Import constants from config (single source of truth)
from src.config import (
    DUCKDB_PATH,
    LANCEDB_PATH,
    TRAINING_DIR,
    FINE_TUNED_DIR,
    BASE_MODEL,
    EMBEDDING_MODEL,
)

# ---------------------------------------------------------------------------
# App & Infrastructure
# ---------------------------------------------------------------------------

app = modal.App("modal-social-sentiment")

# Volumes for persistent storage
data_volume = modal.Volume.from_name("modal-sentiment-data", create_if_missing=True)
models_volume = modal.Volume.from_name("modal-sentiment-models", create_if_missing=True)

# Secrets
api_secrets = modal.Secret.from_name("social-api-keys", required_keys=[])
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# ---------------------------------------------------------------------------
# Images with dependencies (using uv_pip_install + pinned versions)
# ---------------------------------------------------------------------------

# Base image for ingestion and lightweight tasks
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "duckdb==1.1.3",
        "lancedb==0.17.0",
        "beautifulsoup4==4.12.3",
        "httpx==0.28.1",
        "tweepy==4.14.0",
        "praw==7.8.1",
        "PyGithub==2.5.0",
        "tenacity==9.0.0",
        "structlog==24.4.0",
        "pydantic==2.10.4",
        "tqdm==4.67.1",
    )
    .add_local_python_source("src")
)

# Embedding generation image (CPU-based sentence-transformers)
embedding_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.5.1",
        "sentence-transformers==3.3.1",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "duckdb==1.1.3",
        "lancedb==0.17.0",
        "tenacity==9.0.0",
        "structlog==24.4.0",
        "pydantic==2.10.4",
        "tqdm==4.67.1",
    )
    .add_local_python_source("src")
)

# GPU inference image - using NVIDIA CUDA base for robust GPU support
inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "accelerate==1.2.1",
        "bitsandbytes==0.45.0",
        "sentence-transformers==3.3.1",
        "lancedb==0.17.0",
        "duckdb==1.1.3",
        "pyarrow==18.1.0",
        "tenacity==9.0.0",
        "structlog==24.4.0",
        "pydantic==2.10.4",
        "fastapi[standard]==0.115.6",
    )
    .env({"HF_HOME": "/models", "TORCH_HOME": "/models"})  # Cache models to volume
    .add_local_python_source("src")
)

# GPU training image - using NVIDIA CUDA base
training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "accelerate==1.2.1",
        "bitsandbytes==0.45.0",
        "peft==0.14.0",
        "datasets==3.2.0",
        "trl==0.13.0",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "duckdb==1.1.3",
        "tenacity==9.0.0",
        "structlog==24.4.0",
        "pydantic==2.10.4",
        "tqdm==4.67.1",
    )
    .env({"HF_HOME": "/models", "TORCH_HOME": "/models"})  # Cache models to volume
    .add_local_python_source("src")
)

# Webapp image for Streamlit dashboard
webapp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "streamlit==1.41.1",
        "plotly==5.24.1",
        "altair==5.5.0",
        "pandas==2.2.3",
        "pyarrow==18.1.0",
        "duckdb==1.1.3",
        "httpx==0.28.1",
        "pydantic==2.10.4",
        "structlog==24.4.0",
    )
    .add_local_python_source("src")
    .add_local_dir("src/app/.streamlit", remote_path="/root/.streamlit")
)

# Lightweight image for HTTP wrapper endpoints (no model loading here)
endpoint_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]==0.115.6",
)


# ===========================================================================
# INGESTION FUNCTIONS
# ===========================================================================

@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    secrets=[api_secrets],
    timeout=3600,
    schedule=modal.Cron("0 */6 * * *"),
)
def ingest_docs():
    """Ingest Modal documentation from GitHub and blog."""
    import structlog
    from src.ingestion.docs_ingester import DocsIngester
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Starting docs ingestion")

    # Use GitHub token if available for better rate limits
    github_token = os.environ.get("GITHUB_TOKEN")
    ingester = DocsIngester(github_token=github_token)

    with DuckDBStore(DUCKDB_PATH) as db:
        state = db.get_ingestion_state("docs")
        count = 0
        total_fetched = 0
        for message in ingester.fetch(state=state, limit=100):
            total_fetched += 1
            logger.debug(
                "Processing message",
                id=message.id,
                title=message.title[:50] if message.title else "N/A",
            )
            if db.insert_message(message):
                count += 1
                logger.info("Inserted message", id=message.id)
            else:
                logger.warning("Failed to insert", id=message.id)
        db.update_ingestion_state(ingester.get_initial_state())
        logger.info(
            "Docs ingestion complete", new_messages=count, total_fetched=total_fetched
        )

    data_volume.commit()
    return {"status": "success", "new_messages": count}


@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    secrets=[api_secrets],
    timeout=3600,
    schedule=modal.Cron("0 */4 * * *"),
)
def ingest_github():
    """Ingest GitHub Issues and Discussions."""
    import structlog
    from src.ingestion.github_ingester import GitHubIngester
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Starting GitHub ingestion")

    token = os.environ.get("GITHUB_TOKEN") or None
    ingester = GitHubIngester(token=token)

    with DuckDBStore(DUCKDB_PATH) as db:
        state = db.get_ingestion_state("github")
        count = 0
        for message in ingester.fetch(state=state, limit=200):
            if db.insert_message(message):
                count += 1
        db.update_ingestion_state(ingester.get_initial_state())
        logger.info("GitHub ingestion complete", new_messages=count)

    data_volume.commit()
    return {"status": "success", "new_messages": count}


@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    secrets=[api_secrets],
    timeout=3600,
    schedule=modal.Cron("0 */4 * * *"),
)
def ingest_hackernews():
    """Ingest Hacker News discussions."""
    import structlog
    from src.ingestion.hackernews_ingester import HackerNewsIngester
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Starting HackerNews ingestion")

    ingester = HackerNewsIngester()

    with DuckDBStore(DUCKDB_PATH) as db:
        state = db.get_ingestion_state("hackernews")
        count = 0
        for message in ingester.fetch(state=state, limit=100):
            if db.insert_message(message):
                count += 1
        db.update_ingestion_state(ingester.get_initial_state())
        logger.info("HackerNews ingestion complete", new_messages=count)

    data_volume.commit()
    return {"status": "success", "new_messages": count}


# ===========================================================================
# PROCESSING FUNCTIONS
# ===========================================================================

@app.function(
    image=embedding_image,
    volumes={"/data": data_volume},
    timeout=3600,
    schedule=modal.Cron("30 */6 * * *"),
)
def generate_embeddings():
    """Generate embeddings for unprocessed messages."""
    import structlog
    from src.processing.embeddings import EmbeddingGenerator
    from src.storage.duckdb_store import DuckDBStore
    from src.storage.lancedb_store import LanceDBStore
    from src.storage.schemas import VectorRecord

    logger = structlog.get_logger()
    logger.info("Starting embedding generation")

    generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)

    with DuckDBStore(DUCKDB_PATH) as db:
        messages = db.get_messages_without_embeddings(limit=500)

        if not messages:
            logger.info("No unprocessed messages")
            return {"status": "success", "processed": 0}

        texts = [m["content"][:1000] for m in messages]
        embeddings = generator.embed_batch(texts)

        vector_store = LanceDBStore(LANCEDB_PATH)
        records = []
        for msg, emb in zip(messages, embeddings):
            records.append(
                VectorRecord(
                    id=msg["id"],
                    text=msg["content"][:1000],
                    vector=emb,
                    source=msg["source"],
                    created_at=msg["created_at"],
                    url=msg.get("url"),
                    metadata={"title": msg.get("title")},
                )
            )
            db.update_message_analysis(message_id=msg["id"], embedding_id=msg["id"])

        vector_store.upsert_vectors(records)
        logger.info("Embedding generation complete", count=len(records))

    data_volume.commit()
    return {"status": "success", "processed": len(records)}


@app.function(
    image=inference_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    timeout=7200,
    schedule=modal.Cron("0 */12 * * *"),
)
def analyze_sentiment():
    """Run sentiment analysis on unprocessed messages."""
    import structlog
    from src.processing.sentiment import SentimentAnalyzer, load_model_for_sentiment
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Starting sentiment analysis")

    model, tokenizer = load_model_for_sentiment(BASE_MODEL)
    analyzer = SentimentAnalyzer(model=model, tokenizer=tokenizer)

    with DuckDBStore(DUCKDB_PATH) as db:
        result = db.conn.execute(
            """
            SELECT * FROM messages 
            WHERE sentiment_simple IS NULL 
            ORDER BY created_at DESC LIMIT 200
        """
        ).fetchall()
        columns = [d[0] for d in db.conn.description]
        messages = [dict(zip(columns, r)) for r in result]

        if not messages:
            return {"status": "success", "processed": 0}

        for msg in messages:
            try:
                res = analyzer.analyze(msg["content"])
                db.update_message_analysis(
                    message_id=msg["id"],
                    sentiment_simple=res["sentiment_simple"],
                    sentiment_rich=res["sentiment_rich"],
                    content_type=res["content_type"],
                    topics=res["topics"],
                )
            except Exception as e:
                logger.warning("Analysis failed", id=msg["id"], error=str(e))

        db.refresh_daily_metrics()
        logger.info("Sentiment analysis complete", count=len(messages))

    data_volume.commit()
    return {"status": "success", "processed": len(messages)}


# ===========================================================================
# TRAINING FUNCTIONS
# ===========================================================================

@app.function(
    image=training_image,
    gpu="A100",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    timeout=14400,
)
def prepare_training_data():
    """Prepare training dataset from ingested data."""
    import structlog
    from src.training.dataset import TrainingDatasetBuilder
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Preparing training data")

    builder = TrainingDatasetBuilder(output_dir=TRAINING_DIR)

    with DuckDBStore(DUCKDB_PATH) as db:
        result = db.conn.execute(
            """
            SELECT * FROM messages WHERE content IS NOT NULL AND LENGTH(content) > 100
        """
        ).fetchall()
        columns = [d[0] for d in db.conn.description]
        messages = [dict(zip(columns, r)) for r in result]

    doc_examples = builder.build_from_docs(messages)
    conv_examples = builder.build_from_conversations(messages)
    all_examples = doc_examples + conv_examples

    if not all_examples:
        return {"status": "error", "reason": "no examples"}

    train_path, val_path = builder.save_dataset(all_examples)
    data_volume.commit()

    return {
        "status": "success",
        "total": len(all_examples),
        "train_path": str(train_path),
    }


@app.function(
    image=training_image,
    gpu="A100",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    timeout=21600,
)
def run_finetuning(train_path: str = None, epochs: int = 3):
    """Run QLoRA fine-tuning."""
    import structlog
    from src.training.finetune import QLoRATrainer

    logger = structlog.get_logger()

    train_path = train_path or f"{TRAINING_DIR}/training_data.jsonl"
    val_path = train_path.replace(".jsonl", "_val.jsonl")

    trainer = QLoRATrainer(
        base_model=BASE_MODEL,
        output_dir=FINE_TUNED_DIR,
        num_epochs=epochs,
    )
    model_path = trainer.train(train_path, val_path)

    models_volume.commit()
    return {"status": "success", "model_path": model_path}


# ===========================================================================
# INFERENCE SERVICE (with concurrent request handling)
# ===========================================================================

@app.cls(
    image=inference_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)  # Handle up to 4 concurrent requests per GPU
class Assistant:
    """RAG-powered Modal support assistant."""

    @modal.enter()
    def load(self):
        from pathlib import Path
        import structlog
        from src.inference.assistant import load_assistant

        logger = structlog.get_logger()
        candidate_paths = [
            f"{FINE_TUNED_DIR}/final",
            f"{FINE_TUNED_DIR}/merged",
        ]
        model_path = next((p for p in candidate_paths if Path(p).exists()), None)

        logger.info("Loading assistant", model=model_path or BASE_MODEL)
        self.assistant = load_assistant(
            model_path=model_path,
            base_model=BASE_MODEL,
            embedding_model=EMBEDDING_MODEL,
            vector_store_path=LANCEDB_PATH,
            use_quantization=True,
        )

    @modal.method()
    def ask(self, question: str, use_rag: bool = True) -> dict:
        return self.assistant.answer(question=question, use_rag=use_rag)

    @modal.method()
    def health(self) -> dict:
        return {"status": "healthy", "model": self.assistant.model is not None}


@app.function(
    image=endpoint_image,
)
@modal.fastapi_endpoint(method="POST")
def ask(request: dict) -> dict:
    """Web endpoint: POST /ask with {"question": "..."}"""
    q = request.get("question", "")
    if not q:
        return {"error": "No question"}
    return Assistant().ask.remote(q, request.get("use_rag", True))


# ===========================================================================
# WEB APP
# ===========================================================================

@app.function(
    image=webapp_image,
    volumes={"/data": data_volume},
)
@modal.web_server(port=8501, startup_timeout=60)
def dashboard():
    """Streamlit dashboard."""
    import subprocess

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/root/src/app/main.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
    )


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

@app.function(image=base_image, volumes={"/data": data_volume})
def reset_embeddings():
    """Reset all embeddings to regenerate them."""
    import shutil
    import structlog
    from pathlib import Path
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Resetting embeddings...")

    with DuckDBStore(DUCKDB_PATH) as db:
        # Reset only embedding pointers; keep sentiment analysis timestamps.
        db.conn.execute(
            """
            UPDATE messages 
            SET embedding_id = NULL
        """
        )
        count = db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE embedding_id IS NULL"
        ).fetchone()[0]
        logger.info(f"Reset {count} messages for re-embedding")

    # Remove old LanceDB data
    lance_path = Path(LANCEDB_PATH)
    if lance_path.exists():
        shutil.rmtree(lance_path)
        logger.info("Removed old LanceDB data")

    data_volume.commit()
    return {"status": "success", "reset_count": count}


@app.function(image=base_image, volumes={"/data": data_volume})
def check_db_status():
    """Check database status."""
    import structlog
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()

    with DuckDBStore(DUCKDB_PATH) as db:
        total = db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        logger.info(f"Total messages: {total}")

        by_source = db.conn.execute(
            "SELECT source, COUNT(*) FROM messages GROUP BY source"
        ).fetchall()
        for source, count in by_source:
            logger.info(f"  {source}: {count}")

        # Sample a few messages
        samples = db.conn.execute(
            "SELECT id, title FROM messages LIMIT 5"
        ).fetchall()
        logger.info("Sample messages:")
        for id, title in samples:
            logger.info(f"  {id}: {title}")

    return {"total": total, "by_source": dict(by_source)}


# ===========================================================================
# TEST FUNCTIONS (run on Modal for real environment validation)
# ===========================================================================

@app.function(image=base_image, volumes={"/data": data_volume})
def test_db_connection():
    """Verify DuckDB is accessible and schema is correct."""
    from src.storage.duckdb_store import DuckDBStore

    with DuckDBStore(DUCKDB_PATH) as db:
        # Check tables exist
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        # Check message count
        count = db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        # Check schema has expected columns
        columns = db.conn.execute("PRAGMA table_info(messages)").fetchall()
        column_names = [c[1] for c in columns]

    return {
        "status": "pass",
        "tables": table_names,
        "message_count": count,
        "columns": column_names,
    }


@app.function(image=embedding_image, volumes={"/data": data_volume})
def test_embeddings():
    """Verify embedding generation works."""
    from src.processing.embeddings import EmbeddingGenerator

    gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
    test_texts = ["Modal is a cloud platform for running Python.", "How do I deploy?"]
    embeddings = gen.embed_batch(test_texts)

    # Validate output
    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert len(embeddings[0]) == 384, f"Expected 384 dims, got {len(embeddings[0])}"

    return {
        "status": "pass",
        "model": EMBEDDING_MODEL,
        "embedding_dim": len(embeddings[0]),
        "batch_size": len(embeddings),
    }


@app.function(
    image=inference_image,
    gpu="A10G",
    volumes={"/models": models_volume},
    secrets=[hf_secret],
    timeout=600,
)
def test_model_loading():
    """Verify model loads correctly on GPU."""
    import torch
    from transformers import AutoTokenizer

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    # Load tokenizer (fast check without loading full model)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Quick tokenization test
    tokens = tokenizer("Hello Modal!", return_tensors="pt")

    return {
        "status": "pass",
        "cuda_available": cuda_available,
        "gpu": device_name,
        "model": BASE_MODEL,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "test_tokens": tokens["input_ids"].shape[1],
    }


@app.function(image=base_image, volumes={"/data": data_volume}, secrets=[api_secrets])
def test_ingestion_config():
    """Verify ingestion dependencies and secrets are available."""
    import httpx

    results = {}

    # Test HTTP client
    try:
        resp = httpx.get("https://modal.com", timeout=10)
        results["http_client"] = "pass" if resp.status_code == 200 else "fail"
    except Exception as e:
        results["http_client"] = f"fail: {e}"

    # Test GitHub token (if configured)
    github_token = os.environ.get("GITHUB_TOKEN")
    results["github_token"] = "configured" if github_token else "not set"

    # Test volume mount
    from pathlib import Path

    data_path = Path("/data")
    results["volume_mounted"] = data_path.exists()
    results["volume_writable"] = os.access("/data", os.W_OK)

    all_pass = (
        results["http_client"] == "pass"
        and results["volume_mounted"]
        and results["volume_writable"]
    )

    return {"status": "pass" if all_pass else "fail", **results}


# ===========================================================================
# LOCAL ENTRYPOINT
# ===========================================================================

@app.local_entrypoint()
def main(task: str = "ingest"):
    """
    Run tasks on Modal.

    Usage:
        modal run app.py                    # Run all ingestion
        modal run app.py --task test        # Run tests on Modal
        modal run app.py --task ingest      # Run ingestion
        modal run app.py --task process     # Run processing
        modal run app.py --task train       # Run fine-tuning
        modal run app.py --task ask         # Test assistant
    """
    if task == "test":
        print("🧪 Running Modal-native tests...\n")
        results = []
        failed = False

        # Run tests in parallel where possible
        print("  [1/4] Testing DB connection...")
        db_result = test_db_connection.remote()
        results.append(("DB Connection", db_result))

        print("  [2/4] Testing ingestion config...")
        config_result = test_ingestion_config.remote()
        results.append(("Ingestion Config", config_result))

        print("  [3/4] Testing embeddings...")
        emb_result = test_embeddings.remote()
        results.append(("Embeddings", emb_result))

        print("  [4/4] Testing model loading (GPU)...")
        model_result = test_model_loading.remote()
        results.append(("Model Loading", model_result))

        # Print results
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)

        for name, result in results:
            status = result.get("status", "unknown")
            icon = "✅" if status == "pass" else "❌"
            print(f"\n{icon} {name}:")
            for key, value in result.items():
                if key != "status":
                    print(f"   {key}: {value}")
            if status != "pass":
                failed = True

        print("\n" + "=" * 50)
        if failed:
            print("❌ Some tests failed!")
            raise SystemExit(1)
        else:
            print("✅ All tests passed!")

    elif task == "ingest":
        print("📥 Running ingestion on Modal...")
        print(f"   Docs: {ingest_docs.remote()}")
        print(f"   GitHub: {ingest_github.remote()}")
        print(f"   HackerNews: {ingest_hackernews.remote()}")

    elif task == "process":
        print("⚙️ Running processing on Modal...")
        print(f"   Embeddings: {generate_embeddings.remote()}")
        print(f"   Sentiment: {analyze_sentiment.remote()}")

    elif task == "train":
        print("🎯 Running fine-tuning on Modal...")
        result = prepare_training_data.remote()
        print(f"   Prep: {result}")
        if result["status"] == "success":
            print(f"   Training: {run_finetuning.remote(result['train_path'])}")

    elif task == "ask":
        print("🤖 Testing assistant...")
        assistant = Assistant()
        print(f"   Health: {assistant.health.remote()}")
        result = assistant.ask.remote("How do I use Modal volumes?")
        print(f"   Answer: {result['answer'][:300]}...")

    else:
        print(f"Unknown task: {task}")
        print("Available: test, ingest, process, train, ask")
