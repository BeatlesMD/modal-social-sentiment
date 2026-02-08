"""Processing jobs for embeddings and sentiment analysis."""

import modal

from src.config import (
    BASE_MODEL,
    DUCKDB_PATH,
    EMBEDDING_MODEL,
    KNOWLEDGE_SOURCES,
    LANCEDB_PATH,
)

from .common import data_volume, embedding_image, hf_secret, inference_image, models_volume

app = modal.App("modal-social-sentiment-processing")


@app.function(
    image=embedding_image,
    volumes={"/data": data_volume},
    timeout=3600,
    schedule=modal.Cron("30 */6 * * *"),
)
def generate_embeddings():
    """Generate embeddings for messages without embeddings."""
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
            logger.info("No messages need embeddings")
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

        # Write vectors to LanceDB FIRST, then mark in DuckDB.
        # This prevents DuckDB marking embeddings as done when LanceDB fails.
        vector_store.upsert_vectors(records)
        vector_store.sync_to_volume()

        for msg in messages:
            db.update_message_analysis(message_id=msg["id"], embedding_id=msg["id"])

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
    """Run sentiment analysis on messages without sentiment labels."""
    import structlog
    from src.processing.sentiment import SentimentAnalyzer, load_model_for_sentiment
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Starting sentiment analysis")

    model, tokenizer = load_model_for_sentiment(BASE_MODEL)
    analyzer = SentimentAnalyzer(model=model, tokenizer=tokenizer)

    with DuckDBStore(DUCKDB_PATH) as db:
        knowledge_placeholders = ", ".join(["?"] * len(KNOWLEDGE_SOURCES))
        result = db.conn.execute(
            f"""
            SELECT * FROM messages
            WHERE sentiment_simple IS NULL
            AND source NOT IN ({knowledge_placeholders})
            ORDER BY created_at DESC LIMIT 200
            """,
            KNOWLEDGE_SOURCES,
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
            except Exception as exc:
                logger.warning("Analysis failed", id=msg["id"], error=str(exc))

        db.refresh_daily_metrics()
        logger.info("Sentiment analysis complete", count=len(messages))

    data_volume.commit()
    return {"status": "success", "processed": len(messages)}


@app.function(image=embedding_image, volumes={"/data": data_volume})
def reset_embeddings():
    """Reset embedding pointers and remove vector store for full re-embedding."""
    import shutil
    import structlog
    from pathlib import Path
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Resetting embeddings")

    with DuckDBStore(DUCKDB_PATH) as db:
        db.conn.execute(
            """
            UPDATE messages
            SET embedding_id = NULL
            """
        )
        count = db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE embedding_id IS NULL"
        ).fetchone()[0]
        logger.info("Reset messages for re-embedding", count=count)

    lance_path = Path(LANCEDB_PATH)
    if lance_path.exists():
        shutil.rmtree(lance_path)
        logger.info("Removed old LanceDB data")

    data_volume.commit()
    return {"status": "success", "reset_count": count}


@app.function(image=embedding_image, volumes={"/data": data_volume})
def clear_source_data(source: str):
    """Clear all messages from a specific source for re-ingestion."""
    import structlog
    from src.storage.duckdb_store import DuckDBStore
    from src.storage.lancedb_store import LanceDBStore

    logger = structlog.get_logger()
    
    with DuckDBStore(DUCKDB_PATH) as db:
        count_before = db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE source = ?", [source]
        ).fetchone()[0]
        
        db.conn.execute("DELETE FROM messages WHERE source = ?", [source])
        db.conn.execute("DELETE FROM ingestion_state WHERE source = ?", [source])
        
        logger.info("Cleared source data", source=source, deleted_count=count_before)
    
    vector_store = LanceDBStore(LANCEDB_PATH)
    deleted_vectors = vector_store.delete_by_source(source)
    vector_store.sync_to_volume()
    logger.info("Cleared vector data", source=source, deleted_vectors=deleted_vectors)
    
    data_volume.commit()
    return {
        "status": "success",
        "source": source,
        "deleted_count": count_before,
        "deleted_vectors": deleted_vectors,
    }


@app.function(image=embedding_image, volumes={"/data": data_volume})
def check_db_status():
    """Check current database counts and coverage."""
    import structlog
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()

    with DuckDBStore(DUCKDB_PATH) as db:
        total = db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        logger.info("Total messages", total=total)

        by_source = db.conn.execute(
            "SELECT source, COUNT(*) FROM messages GROUP BY source"
        ).fetchall()
        for source, count in by_source:
            logger.info("Source count", source=source, count=count)

        knowledge_placeholders = ", ".join(["?"] * len(KNOWLEDGE_SOURCES))
        split_counts = db.conn.execute(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE source IN ({knowledge_placeholders})) AS knowledge_messages,
                COUNT(*) FILTER (WHERE source NOT IN ({knowledge_placeholders})) AS voice_messages
            FROM messages
            """,
            KNOWLEDGE_SOURCES + KNOWLEDGE_SOURCES,
        ).fetchone()
        knowledge_count = split_counts[0] or 0
        voice_count = split_counts[1] or 0

        sentiment_counts = db.conn.execute(
            """
            SELECT COALESCE(sentiment_simple, 'unlabeled') AS sentiment, COUNT(*)
            FROM messages
            GROUP BY 1
            ORDER BY 2 DESC
            """
        ).fetchall()
        by_sentiment = dict(sentiment_counts)
        logger.info(
            "Corpus split",
            knowledge_messages=knowledge_count,
            voice_messages=voice_count,
        )
        logger.info("Sentiment coverage", counts=by_sentiment)

        samples = db.conn.execute(
            "SELECT id, title FROM messages LIMIT 5"
        ).fetchall()
        for msg_id, title in samples:
            logger.info("Sample message", id=msg_id, title=title)

    return {
        "total": total,
        "voice_messages": voice_count,
        "knowledge_messages": knowledge_count,
        "by_source": dict(by_source),
        "by_sentiment": by_sentiment,
    }
