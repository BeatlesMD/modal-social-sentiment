"""
DuckDB storage layer for processed messages and analytics.

DuckDB is excellent for:
- Analytics queries (aggregations, time series)
- Parquet file integration
- Running directly on Modal Volumes
"""

import json
from datetime import datetime
from pathlib import Path

import duckdb
import structlog

from .schemas import Message, IngestionState, DailyMetrics, Source

logger = structlog.get_logger()


class DuckDBStore:
    """
    DuckDB-based storage for messages and analytics.
    
    Designed to work with Modal Volumes - the database file
    persists between function invocations.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_parent_dir()
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def _ensure_parent_dir(self):
        """Ensure the parent directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                content TEXT NOT NULL,
                title VARCHAR,
                author VARCHAR,
                url VARCHAR,
                parent_id VARCHAR,
                thread_id VARCHAR,
                created_at TIMESTAMP NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                processed_at TIMESTAMP,
                metadata JSON,
                sentiment_simple VARCHAR,
                sentiment_rich VARCHAR,
                content_type VARCHAR,
                topics JSON,
                embedding_id VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_state (
                source VARCHAR PRIMARY KEY,
                last_fetched_at TIMESTAMP NOT NULL,
                last_item_id VARCHAR,
                cursor VARCHAR,
                metadata JSON
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date DATE NOT NULL,
                source VARCHAR NOT NULL,
                total_messages INTEGER DEFAULT 0,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                frustration_count INTEGER DEFAULT 0,
                confusion_count INTEGER DEFAULT 0,
                delight_count INTEGER DEFAULT 0,
                gratitude_count INTEGER DEFAULT 0,
                questions_count INTEGER DEFAULT 0,
                bug_reports_count INTEGER DEFAULT 0,
                feature_requests_count INTEGER DEFAULT 0,
                top_topics JSON,
                PRIMARY KEY (date, source)
            )
        """)
        
        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_source 
            ON messages(source)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_created 
            ON messages(created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_sentiment 
            ON messages(sentiment_simple)
        """)
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # -------------------------------------------------------------------------
    # Message operations
    # -------------------------------------------------------------------------
    
    def insert_message(self, message: Message) -> bool:
        """
        Insert a message, returning True if inserted (vs already exists).
        Uses INSERT ... ON CONFLICT DO NOTHING semantics (DuckDB syntax).
        """
        try:
            # Check if already exists first (DuckDB's ON CONFLICT can be finicky)
            exists = self.conn.execute(
                "SELECT 1 FROM messages WHERE id = ?", [message.id]
            ).fetchone()
            
            if exists:
                logger.debug("Message already exists", id=message.id)
                return False
            
            logger.debug("Inserting message", id=message.id, source=str(message.source))
            
            self.conn.execute("""
                INSERT INTO messages (
                    id, source, source_id, content, title, author, url,
                    parent_id, thread_id, created_at, fetched_at, processed_at,
                    metadata, sentiment_simple, sentiment_rich, content_type,
                    topics, embedding_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                message.id,
                message.source.value if hasattr(message.source, 'value') else message.source,
                message.source_id,
                message.content,
                message.title,
                message.author,
                message.url,
                message.parent_id,
                message.thread_id,
                message.created_at,
                message.fetched_at,
                message.processed_at,
                json.dumps(message.metadata) if message.metadata else "{}",
                message.sentiment_simple.value if message.sentiment_simple and hasattr(message.sentiment_simple, 'value') else message.sentiment_simple,
                message.sentiment_rich.value if message.sentiment_rich and hasattr(message.sentiment_rich, 'value') else message.sentiment_rich,
                message.content_type.value if message.content_type and hasattr(message.content_type, 'value') else message.content_type,
                json.dumps(message.topics) if message.topics else "[]",
                message.embedding_id,
            ])
            logger.info("Inserted message successfully", id=message.id)
            return True
        except Exception as e:
            logger.error("Failed to insert message", id=message.id, error=str(e), exc_info=True)
            return False
    
    def insert_messages_batch(self, messages: list[Message]) -> int:
        """Insert multiple messages, returning count of new insertions."""
        inserted = 0
        for msg in messages:
            if self.insert_message(msg):
                inserted += 1
        return inserted
    
    def update_message_analysis(
        self,
        message_id: str,
        sentiment_simple: str | None = None,
        sentiment_rich: str | None = None,
        content_type: str | None = None,
        topics: list[str] | None = None,
        embedding_id: str | None = None,
    ):
        """Update analysis fields for a message.
        
        Uses a workaround for DuckDB's known index limitation with UPDATE
        on tables with ART indexes. We fetch the row, delete it, and reinsert.
        """
        # Fetch existing row
        existing = self.conn.execute(
            "SELECT * FROM messages WHERE id = ?", [message_id]
        ).fetchone()
        
        if not existing:
            logger.warning("Message not found for update", message_id=message_id)
            return
        
        columns = [desc[0] for desc in self.conn.description]
        row = dict(zip(columns, existing))
        
        # Apply updates to the row
        if sentiment_simple is not None:
            row["sentiment_simple"] = sentiment_simple
        if sentiment_rich is not None:
            row["sentiment_rich"] = sentiment_rich
        if content_type is not None:
            row["content_type"] = content_type
        if topics is not None:
            row["topics"] = json.dumps(topics)
        if embedding_id is not None:
            row["embedding_id"] = embedding_id
        
        # Mark as processed if we're doing analysis
        if any([sentiment_simple, sentiment_rich, content_type, topics]):
            row["processed_at"] = datetime.utcnow()
        
        # Delete and reinsert (workaround for DuckDB ART index bug)
        self.conn.execute("DELETE FROM messages WHERE id = ?", [message_id])
        self.conn.execute("""
            INSERT INTO messages (
                id, source, source_id, content, title, author, url,
                parent_id, thread_id, created_at, fetched_at, processed_at,
                metadata, sentiment_simple, sentiment_rich, content_type,
                topics, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["id"],
            row["source"],
            row["source_id"],
            row["content"],
            row["title"],
            row["author"],
            row["url"],
            row["parent_id"],
            row["thread_id"],
            row["created_at"],
            row["fetched_at"],
            row["processed_at"],
            row["metadata"] if isinstance(row["metadata"], str) else json.dumps(row["metadata"] or {}),
            row["sentiment_simple"],
            row["sentiment_rich"],
            row["content_type"],
            row["topics"] if isinstance(row["topics"], str) else json.dumps(row["topics"] or []),
            row["embedding_id"],
        ])
    
    def get_messages_without_embeddings(self, limit: int = 100) -> list[dict]:
        """Get messages that do not have embeddings yet."""
        result = self.conn.execute("""
            SELECT * FROM messages 
            WHERE embedding_id IS NULL
            ORDER BY created_at DESC 
            LIMIT ?
        """, [limit]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def get_unprocessed_messages(self, limit: int = 100) -> list[dict]:
        """Backward-compatible alias for embedding work queue."""
        return self.get_messages_without_embeddings(limit=limit)
    
    def get_messages(
        self,
        source: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        sentiment: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query messages with filters."""
        conditions = []
        params = []
        
        if source:
            conditions.append("source = ?")
            params.append(source)
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)
        if sentiment:
            conditions.append("sentiment_simple = ?")
            params.append(sentiment)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        result = self.conn.execute(f"""
            SELECT * FROM messages 
            {where_clause}
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    # -------------------------------------------------------------------------
    # Ingestion state operations
    # -------------------------------------------------------------------------
    
    def get_ingestion_state(self, source: Source | str) -> IngestionState | None:
        """Get the current ingestion state for a source."""
        source_val = source.value if isinstance(source, Source) else source
        result = self.conn.execute(
            "SELECT * FROM ingestion_state WHERE source = ?",
            [source_val]
        ).fetchone()
        
        if result:
            columns = [desc[0] for desc in self.conn.description]
            row = dict(zip(columns, result))
            row['metadata'] = json.loads(row['metadata']) if row['metadata'] else {}
            return IngestionState(**row)
        return None
    
    def update_ingestion_state(self, state: IngestionState):
        """Update ingestion state for a source."""
        source_val = state.source.value if hasattr(state.source, 'value') else state.source
        
        # Delete existing and insert (DuckDB doesn't support INSERT OR REPLACE)
        self.conn.execute("DELETE FROM ingestion_state WHERE source = ?", [source_val])
        self.conn.execute("""
            INSERT INTO ingestion_state 
            (source, last_fetched_at, last_item_id, cursor, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, [
            source_val,
            state.last_fetched_at,
            state.last_item_id,
            state.cursor,
            json.dumps(state.metadata) if state.metadata else "{}",
        ])
    
    # -------------------------------------------------------------------------
    # Analytics operations
    # -------------------------------------------------------------------------
    
    def get_sentiment_over_time(
        self,
        source: str | None = None,
        days: int = 30,
    ) -> list[dict]:
        """Get sentiment trends over time."""
        source_filter = "AND source = ?" if source else ""
        params = [days]
        if source:
            params.append(source)
        
        result = self.conn.execute(f"""
            SELECT 
                DATE_TRUNC('day', created_at) as date,
                sentiment_simple,
                COUNT(*) as count
            FROM messages
            WHERE created_at >= CURRENT_DATE - (? * INTERVAL '1 day')
            {source_filter}
            GROUP BY DATE_TRUNC('day', created_at), sentiment_simple
            ORDER BY date
        """, params).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_topic_distribution(
        self,
        source: str | None = None,
        days: int = 30,
    ) -> list[dict]:
        """Get topic distribution."""
        # This requires unnesting the topics JSON array
        source_filter = "AND source = ?" if source else ""
        params = [days]
        if source:
            params.append(source)
        
        result = self.conn.execute(f"""
            SELECT 
                topic,
                COUNT(*) as count
            FROM messages,
            UNNEST(from_json(topics, '["VARCHAR"]')) AS t(topic)
            WHERE created_at >= CURRENT_DATE - (? * INTERVAL '1 day')
            AND topics IS NOT NULL
            {source_filter}
            GROUP BY topic
            ORDER BY count DESC
            LIMIT 20
        """, params).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_summary_stats(self, days: int = 30) -> dict:
        """Get summary statistics for the dashboard."""
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT source) as sources,
                COUNT(CASE WHEN sentiment_simple = 'positive' THEN 1 END) as positive,
                COUNT(CASE WHEN sentiment_simple = 'negative' THEN 1 END) as negative,
                COUNT(CASE WHEN sentiment_simple = 'neutral' THEN 1 END) as neutral,
                COUNT(CASE WHEN content_type = 'bug_report' THEN 1 END) as bug_reports,
                COUNT(CASE WHEN content_type = 'feature_request' THEN 1 END) as feature_requests,
                COUNT(CASE WHEN content_type = 'question' THEN 1 END) as questions
            FROM messages
            WHERE created_at >= CURRENT_DATE - (? * INTERVAL '1 day')
        """, [days]).fetchone()
        
        columns = [desc[0] for desc in self.conn.description]
        return dict(zip(columns, result))
    
    def refresh_daily_metrics(self):
        """Refresh the daily_metrics table from messages."""
        # Delete existing metrics and recompute (DuckDB doesn't support INSERT OR REPLACE)
        self.conn.execute("DELETE FROM daily_metrics")
        self.conn.execute("""
            INSERT INTO daily_metrics
            SELECT 
                DATE_TRUNC('day', created_at)::DATE as date,
                source,
                COUNT(*) as total_messages,
                COUNT(CASE WHEN sentiment_simple = 'positive' THEN 1 END),
                COUNT(CASE WHEN sentiment_simple = 'negative' THEN 1 END),
                COUNT(CASE WHEN sentiment_simple = 'neutral' THEN 1 END),
                COUNT(CASE WHEN sentiment_rich = 'frustration' THEN 1 END),
                COUNT(CASE WHEN sentiment_rich = 'confusion' THEN 1 END),
                COUNT(CASE WHEN sentiment_rich = 'delight' THEN 1 END),
                COUNT(CASE WHEN sentiment_rich = 'gratitude' THEN 1 END),
                COUNT(CASE WHEN content_type = 'question' THEN 1 END),
                COUNT(CASE WHEN content_type = 'bug_report' THEN 1 END),
                COUNT(CASE WHEN content_type = 'feature_request' THEN 1 END),
                NULL  -- top_topics computed separately
            FROM messages
            WHERE processed_at IS NOT NULL
            GROUP BY DATE_TRUNC('day', created_at), source
        """)
