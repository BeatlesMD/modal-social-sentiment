"""
Modal Social Sentiment - Main Application

All Modal functions defined in one place following Modal best practices.
Deploy: modal deploy app.py
Run:    modal run app.py
"""

import os
import modal

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
# Images with dependencies
# ---------------------------------------------------------------------------

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas", "pyarrow", "duckdb", "lancedb",
        "beautifulsoup4", "httpx",
        "tweepy", "praw", "PyGithub",
        "tenacity", "structlog", "pydantic", "tqdm",
    )
    .add_local_python_source("src")  # Proper way to include local Python code
)

embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "sentence-transformers",
        "pandas", "pyarrow", "duckdb", "lancedb",
        "tenacity", "structlog", "pydantic", "tqdm",
    )
    .add_local_python_source("src")
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "bitsandbytes",
        "sentence-transformers", "lancedb", "duckdb", "pyarrow",
        "tenacity", "structlog", "pydantic",
        "fastapi[standard]",  # Required for web endpoints
    )
    .add_local_python_source("src")
)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "bitsandbytes",
        "peft", "datasets", "trl",
        "pandas", "pyarrow", "duckdb",
        "tenacity", "structlog", "pydantic", "tqdm",
    )
    .add_local_python_source("src")
)

webapp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit", "plotly", "altair",
        "pandas", "pyarrow", "duckdb",
        "httpx", "pydantic", "structlog",
    )
    .add_local_python_source("src")
    .add_local_dir("src/app/.streamlit", remote_path="/root/.streamlit")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUCKDB_PATH = "/data/processed/sentiment.duckdb"
LANCEDB_PATH = "/data/vectors/embeddings.lance"
TRAINING_DIR = "/data/training"
FINE_TUNED_DIR = "/models/fine-tuned"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


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
            logger.debug("Processing message", id=message.id, title=message.title[:50] if message.title else "N/A")
            if db.insert_message(message):
                count += 1
                logger.info("Inserted message", id=message.id)
            else:
                logger.warning("Failed to insert", id=message.id)
        db.update_ingestion_state(ingester.get_initial_state())
        logger.info("Docs ingestion complete", new_messages=count, total_fetched=total_fetched)
    
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
    import shutil
    import structlog
    from pathlib import Path
    from src.processing.embeddings import EmbeddingGenerator
    from src.storage.duckdb_store import DuckDBStore
    from src.storage.lancedb_store import LanceDBStore
    from src.storage.schemas import VectorRecord
    
    logger = structlog.get_logger()
    logger.info("Starting embedding generation")
    
    generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
    
    with DuckDBStore(DUCKDB_PATH) as db:
        messages = db.get_unprocessed_messages(limit=500)
        
        if not messages:
            logger.info("No unprocessed messages")
            return {"status": "success", "processed": 0}
        
        texts = [m["content"][:1000] for m in messages]
        embeddings = generator.embed_batch(texts)
        
        # Use temp directory for LanceDB (avoids Volume file lock issues)
        tmp_lance_path = "/tmp/embeddings.lance"
        final_lance_path = LANCEDB_PATH
        
        # Start fresh in temp (we'll merge with existing later if needed)
        if Path(tmp_lance_path).exists():
            shutil.rmtree(tmp_lance_path)
        
        vector_store = LanceDBStore(tmp_lance_path)
        records = []
        for msg, emb in zip(messages, embeddings):
            records.append(VectorRecord(
                id=msg["id"],
                text=msg["content"][:1000],
                vector=emb,
                source=msg["source"],
                created_at=msg["created_at"],
                url=msg.get("url"),
                metadata={"title": msg.get("title")},
            ))
            db.update_message_analysis(message_id=msg["id"], embedding_id=msg["id"])
        
        vector_store.add_vectors(records)
        logger.info("Embedding generation complete", count=len(records))
        
        # Copy back to Volume
        Path(final_lance_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(final_lance_path).exists():
            shutil.rmtree(final_lance_path)
        shutil.copytree(tmp_lance_path, final_lance_path)
    
    data_volume.commit()
    return {"status": "success", "processed": len(records)}


@app.function(
    image=inference_image,
    gpu="A10G",
    volumes={"/data": data_volume},
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
        result = db.conn.execute("""
            SELECT * FROM messages 
            WHERE sentiment_simple IS NULL 
            ORDER BY created_at DESC LIMIT 200
        """).fetchall()
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
        result = db.conn.execute("""
            SELECT * FROM messages WHERE content IS NOT NULL AND LENGTH(content) > 100
        """).fetchall()
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
# INFERENCE SERVICE
# ===========================================================================

@app.cls(
    image=inference_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
class Assistant:
    """RAG-powered Modal support assistant."""
    
    @modal.enter()
    def load(self):
        from pathlib import Path
        import structlog
        from src.inference.assistant import load_assistant
        
        logger = structlog.get_logger()
        model_path = f"{FINE_TUNED_DIR}/merged"
        model_path = model_path if Path(model_path).exists() else None
        
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


@app.function(image=inference_image, gpu="A10G", volumes={"/data": data_volume, "/models": models_volume}, secrets=[hf_secret])
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
    subprocess.Popen([
        "streamlit", "run", "/root/src/app/main.py",
        "--server.port=8501", "--server.address=0.0.0.0",
        "--server.headless=true", "--browser.gatherUsageStats=false",
    ])


# ===========================================================================
# ENTRYPOINT
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
        # Reset processed_at and embedding_id for all messages
        db.conn.execute("""
            UPDATE messages 
            SET processed_at = NULL, embedding_id = NULL
        """)
        count = db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed_at IS NULL").fetchone()[0]
        logger.info(f"Reset {count} messages for reprocessing")
    
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
        total = db.conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
        logger.info(f"Total messages: {total}")
        
        by_source = db.conn.execute('SELECT source, COUNT(*) FROM messages GROUP BY source').fetchall()
        for source, count in by_source:
            logger.info(f"  {source}: {count}")
        
        # Sample a few messages
        samples = db.conn.execute('SELECT id, title FROM messages LIMIT 5').fetchall()
        logger.info("Sample messages:")
        for id, title in samples:
            logger.info(f"  {id}: {title}")
    
    return {"total": total, "by_source": dict(by_source)}


@app.local_entrypoint()
def main(task: str = "ingest"):
    """
    Run tasks on Modal.
    
    Usage:
        modal run app.py                    # Run all ingestion
        modal run app.py --task ingest      # Run ingestion
        modal run app.py --task process     # Run processing
        modal run app.py --task train       # Run fine-tuning
        modal run app.py --task ask         # Test assistant
    """
    if task == "ingest":
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
        print("Available: ingest, process, train, ask")
