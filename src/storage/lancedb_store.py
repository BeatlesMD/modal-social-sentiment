"""
LanceDB storage layer for vector embeddings and semantic search.

LanceDB is excellent for:
- Serverless vector search (no separate server needed)
- Direct file storage on Modal Volumes
- Fast similarity search with IVF-PQ indexes
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import structlog

from .schemas import VectorRecord

logger = structlog.get_logger()


class LanceDBStore:
    """
    LanceDB-based vector storage for semantic search.
    
    Stores embeddings alongside metadata for RAG retrieval.
    Works directly on Modal Volumes.
    """
    
    TABLE_NAME = "embeddings"
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_parent_dir()
        self.db = lancedb.connect(db_path)
        self._init_table()
    
    def _ensure_parent_dir(self):
        """Ensure the parent directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_table(self):
        """Initialize the embeddings table if it doesn't exist."""
        try:
            table_names = self.db.table_names()
            if self.TABLE_NAME not in table_names:
                # Create empty table with schema
                # LanceDB infers schema from first insert
                logger.info("Embeddings table will be created on first insert")
                self._table = None
            else:
                self._table = self.db.open_table(self.TABLE_NAME)
        except Exception as e:
            logger.warning(f"Error checking tables: {e}, will create on first insert")
            self._table = None
    
    @property
    def table(self):
        """Get or create the table."""
        if self._table is None:
            if self.TABLE_NAME in self.db.table_names():
                self._table = self.db.open_table(self.TABLE_NAME)
        return self._table
    
    def add_vectors(self, records: list[VectorRecord]) -> int:
        """
        Backward-compatible insert API.
        """
        return self.upsert_vectors(records)

    def upsert_vectors(self, records: list[VectorRecord]) -> int:
        """
        Upsert vector records by id.
        Returns count of records written.
        """
        if not records:
            return 0
        
        data = [
            {
                "id": r.id,
                "text": r.text,
                "vector": r.vector,
                "source": r.source,
                "created_at": r.created_at.isoformat(),
                "url": r.url,
                "metadata": r.metadata,
            }
            for r in records
        ]
        
        if self._table is None:
            # Create table with first batch
            self._table = self.db.create_table(self.TABLE_NAME, data)
            logger.info("Created embeddings table", count=len(data))
        else:
            self._delete_existing_ids([r.id for r in records])
            self._table.add(data)
            logger.info("Upserted vectors", count=len(data))
        
        return len(data)

    def _delete_existing_ids(self, record_ids: list[str], chunk_size: int = 100) -> None:
        """Delete existing rows by id before re-inserting updated vectors."""
        if not record_ids or self._table is None:
            return

        for i in range(0, len(record_ids), chunk_size):
            chunk = record_ids[i : i + chunk_size]
            predicates = []
            for record_id in chunk:
                safe_id = record_id.replace("'", "''")
                predicates.append(f"id = '{safe_id}'")

            if predicates:
                self._table.delete(" OR ".join(predicates))
    
    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_source: str | None = None,
        filter_after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query embedding
            limit: Maximum results to return
            filter_source: Optional source filter
            filter_after: Optional date filter
        
        Returns:
            List of matching records with distances
        """
        if self.table is None:
            logger.warning("No embeddings table exists yet")
            return []
        
        # Build filter string
        filters = []
        if filter_source:
            filters.append(f"source = '{filter_source}'")
        if filter_after:
            filters.append(f"created_at >= '{filter_after.isoformat()}'")
        
        where_clause = " AND ".join(filters) if filters else None
        
        # Perform search
        query = self.table.search(query_vector).limit(limit)
        if where_clause:
            query = query.where(where_clause)
        
        results = query.to_list()
        
        return results
    
    def search_text(
        self,
        query_text: str,
        embedding_fn,
        limit: int = 10,
        **filter_kwargs,
    ) -> list[dict[str, Any]]:
        """
        Convenience method: embed query text and search.
        
        Args:
            query_text: The text to search for
            embedding_fn: Function that takes text and returns embedding
            limit: Maximum results
            **filter_kwargs: Passed to search()
        """
        query_vector = embedding_fn(query_text)
        return self.search(query_vector, limit=limit, **filter_kwargs)
    
    def get_by_id(self, record_id: str) -> dict | None:
        """Get a specific record by ID."""
        if self.table is None:
            return None
        
        results = self.table.search().where(f"id = '{record_id}'").limit(1).to_list()
        return results[0] if results else None
    
    def delete_by_source(self, source: str) -> int:
        """Delete all records from a specific source."""
        if self.table is None:
            return 0
        
        # LanceDB delete syntax
        self.table.delete(f"source = '{source}'")
        logger.info("Deleted vectors", source=source)
        return -1  # LanceDB doesn't return count
    
    def count(self, source: str | None = None) -> int:
        """Count records, optionally filtered by source."""
        if self.table is None:
            return 0
        
        if source:
            return len(self.table.search().where(f"source = '{source}'").limit(1000000).to_list())
        return self.table.count_rows()
    
    def create_index(self, num_partitions: int = 256, num_sub_vectors: int = 96):
        """
        Create an IVF-PQ index for faster search on large datasets.
        
        Call this after adding a significant amount of data (10k+ vectors).
        """
        if self.table is None:
            logger.warning("Cannot create index: no table exists")
            return
        
        self.table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
        logger.info(
            "Created vector index",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
