"""
Base ingester class that all source-specific ingesters inherit from.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator

import structlog

from src.storage.schemas import Message, IngestionState, Source

logger = structlog.get_logger()


class BaseIngester(ABC):
    """
    Abstract base class for data ingesters.
    
    Each ingester is responsible for:
    1. Fetching data from a specific source
    2. Normalizing it into the Message schema
    3. Tracking ingestion state for incremental fetches
    """
    
    source: Source  # Must be defined by subclass
    
    def __init__(self):
        self.logger = logger.bind(source=self.source.value)
    
    @abstractmethod
    def fetch(
        self,
        state: IngestionState | None = None,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """
        Fetch messages from the source.
        
        Args:
            state: Previous ingestion state for incremental fetching
            limit: Maximum number of messages to fetch (for testing)
        
        Yields:
            Normalized Message objects
        """
        pass
    
    @abstractmethod
    def get_initial_state(self) -> IngestionState:
        """
        Get the initial ingestion state for this source.
        Used when starting fresh (no previous state).
        """
        pass
    
    def create_message_id(self, source_id: str) -> str:
        """Create a unique message ID combining source and source_id."""
        return f"{self.source.value}:{source_id}"
    
    def log_progress(self, count: int, **extra):
        """Log ingestion progress."""
        self.logger.info("Ingestion progress", count=count, **extra)


class RateLimitedIngester(BaseIngester):
    """
    Base class for ingesters that need rate limiting.
    Adds backoff and retry logic.
    """
    
    # Override in subclass
    requests_per_minute: int = 60
    
    def __init__(self):
        super().__init__()
        self._last_request_time: datetime | None = None
        self._request_count = 0
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        import asyncio
        
        if self._last_request_time is None:
            self._last_request_time = datetime.utcnow()
            self._request_count = 1
            return
        
        self._request_count += 1
        elapsed = (datetime.utcnow() - self._last_request_time).total_seconds()
        
        # Reset counter every minute
        if elapsed >= 60:
            self._last_request_time = datetime.utcnow()
            self._request_count = 1
            return
        
        # If we've hit the limit, wait
        if self._request_count >= self.requests_per_minute:
            wait_time = 60 - elapsed
            self.logger.info("Rate limiting", wait_seconds=wait_time)
            await asyncio.sleep(wait_time)
            self._last_request_time = datetime.utcnow()
            self._request_count = 1
