"""Shared Modal resources (images, volumes, secrets, constants)."""

import modal

from src.config import (
    BASE_MODEL,
    DUCKDB_PATH,
    EMBEDDING_MODEL,
    FINE_TUNED_DIR,
    LANCEDB_PATH,
    TRAINING_DIR,
)

APP_NAME = "modal-social-sentiment"

# Volumes for persistent storage
data_volume = modal.Volume.from_name("modal-sentiment-data", create_if_missing=True)
models_volume = modal.Volume.from_name("modal-sentiment-models", create_if_missing=True)

# Secrets
api_secrets = modal.Secret.from_name("social-api-keys", required_keys=[])
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

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
    .env({"HF_HOME": "/models", "TORCH_HOME": "/models"})
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
    .env({"HF_HOME": "/models", "TORCH_HOME": "/models"})
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
        "modal==1.3.2",
    )
    .add_local_python_source("src")
    .add_local_dir("src/app/.streamlit", remote_path="/root/.streamlit")
)

# Lightweight image for HTTP wrapper endpoints
# Needs src for common.py imports (via inference_service.py -> common.py -> src.config)
endpoint_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi[standard]==0.115.6")
    .add_local_python_source("src")
)

