"""Scheduled ingestion jobs."""

import os

import modal

from src.config import DUCKDB_PATH

from .common import api_secrets, base_image, data_volume

app = modal.App("modal-social-sentiment-ingestion")


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


@app.function(
    image=base_image,
    volumes={"/data": data_volume},
    secrets=[api_secrets],
    timeout=3600,
    # No schedule - run manually or add schedule after upgrading Modal plan
    # schedule=modal.Cron("0 */6 * * *"),
)
def ingest_reddit():
    """Ingest Reddit discussions about Modal.
    
    Uses API if credentials are available, otherwise falls back to RSS feeds.
    """
    import structlog
    from src.storage.duckdb_store import DuckDBStore

    logger = structlog.get_logger()
    logger.info("Starting Reddit ingestion")

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

    # Use API if credentials available, otherwise RSS fallback
    if client_id and client_secret:
        from src.ingestion.reddit_ingester import RedditIngester
        ingester = RedditIngester(client_id=client_id, client_secret=client_secret)
        mode = "api"
        limit = 200
    else:
        from src.ingestion.reddit_rss_ingester import RedditRSSIngester
        ingester = RedditRSSIngester()
        mode = "rss"
        limit = 100  # RSS is more limited
        logger.info("No Reddit API credentials, using RSS fallback")

    with DuckDBStore(DUCKDB_PATH) as db:
        state = db.get_ingestion_state("reddit")
        count = 0
        for message in ingester.fetch(state=state, limit=limit):
            if db.insert_message(message):
                count += 1
        db.update_ingestion_state(ingester.get_initial_state())
        logger.info("Reddit ingestion complete", new_messages=count, mode=mode)

    data_volume.commit()
    return {"status": "success", "new_messages": count, "mode": mode}

