"""
Shared configuration constants.

This module is the single source of truth for all configuration.
Imported by Modal functions running in containers.
"""

# ---------------------------------------------------------------------------
# Path Constants (inside Modal containers)
# ---------------------------------------------------------------------------
DATA_DIR = "/data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
VECTORS_DIR = f"{DATA_DIR}/vectors"
TRAINING_DIR = f"{DATA_DIR}/training"

MODELS_DIR = "/models"
CHECKPOINTS_DIR = f"{MODELS_DIR}/checkpoints"
FINE_TUNED_DIR = f"{MODELS_DIR}/fine-tuned"

# Database paths
DUCKDB_PATH = f"{PROCESSED_DIR}/sentiment.duckdb"
LANCEDB_PATH = f"{VECTORS_DIR}/embeddings.lance"

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Source Groups
# ---------------------------------------------------------------------------
# Knowledge sources are useful for retrieval/context, but should be excluded
# from product sentiment KPIs by default.
KNOWLEDGE_SOURCES = ["docs", "blog"]
VOICE_SOURCES = ["github", "hackernews", "reddit", "twitter", "slack"]

# ---------------------------------------------------------------------------
# Sentiment Categories
# ---------------------------------------------------------------------------
SENTIMENT_SIMPLE = ["positive", "negative", "neutral"]
SENTIMENT_RICH = [
    "frustration",
    "confusion",
    "delight",
    "gratitude",
    "curiosity",
    "complaint",
    "neutral",
]

TOPICS = [
    "gpu_availability",
    "pricing",
    "performance",
    "documentation",
    "ease_of_use",
    "reliability",
    "support",
    "functions",
    "volumes",
    "secrets",
    "images",
    "scheduling",
    "web_endpoints",
    "other",
]
