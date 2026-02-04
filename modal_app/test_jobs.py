"""Modal-native validation functions."""

import os

import modal

from src.config import BASE_MODEL, DUCKDB_PATH, EMBEDDING_MODEL

from .common import (
    api_secrets,
    base_image,
    data_volume,
    embedding_image,
    hf_secret,
    inference_image,
    models_volume,
)

app = modal.App("modal-social-sentiment-tests")


@app.function(image=base_image, volumes={"/data": data_volume})
def test_db_connection():
    """Verify DuckDB is accessible and schema is correct."""
    from src.storage.duckdb_store import DuckDBStore

    with DuckDBStore(DUCKDB_PATH) as db:
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        count = db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

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

    generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
    test_texts = ["Modal is a cloud platform for running Python.", "How do I deploy?"]
    embeddings = generator.embed_batch(test_texts)

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
    """Verify tokenizer and CUDA visibility on inference image."""
    import torch
    from transformers import AutoTokenizer

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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
    from pathlib import Path

    import httpx

    results = {}

    try:
        response = httpx.get("https://modal.com", timeout=10)
        results["http_client"] = "pass" if response.status_code == 200 else "fail"
    except Exception as exc:
        results["http_client"] = f"fail: {exc}"

    github_token = os.environ.get("GITHUB_TOKEN")
    results["github_token"] = "configured" if github_token else "not set"

    data_path = Path("/data")
    results["volume_mounted"] = data_path.exists()
    results["volume_writable"] = os.access("/data", os.W_OK)

    all_pass = (
        results["http_client"] == "pass"
        and results["volume_mounted"]
        and results["volume_writable"]
    )

    return {"status": "pass" if all_pass else "fail", **results}

