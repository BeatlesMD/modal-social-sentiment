"""
Data schemas for the social sentiment platform.

Using Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Source(str, Enum):
    """Data source identifiers."""
    DOCS = "docs"
    BLOG = "blog"
    GITHUB = "github"
    TWITTER = "twitter"
    REDDIT = "reddit"
    HACKERNEWS = "hackernews"
    SLACK = "slack"


class SentimentSimple(str, Enum):
    """Simple sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentRich(str, Enum):
    """Rich sentiment classification."""
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    DELIGHT = "delight"
    GRATITUDE = "gratitude"
    CURIOSITY = "curiosity"
    COMPLAINT = "complaint"
    NEUTRAL = "neutral"


class ContentType(str, Enum):
    """Content type classification."""
    QUESTION = "question"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    PRAISE = "praise"
    COMPLAINT = "complaint"
    DISCUSSION = "discussion"
    DOCUMENTATION = "documentation"
    ANNOUNCEMENT = "announcement"


class Message(BaseModel):
    """
    Core message schema - normalized representation of content from any source.
    """
    id: str = Field(..., description="Unique identifier (source:source_id)")
    source: Source
    source_id: str = Field(..., description="Original ID from the source platform")
    
    # Content
    content: str
    title: str | None = None  # For Reddit posts, GitHub issues, etc.
    author: str | None = None
    url: str | None = None
    
    # Threading
    parent_id: str | None = None  # For replies/comments
    thread_id: str | None = None  # Root of the conversation
    
    # Timestamps
    created_at: datetime
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None
    
    # Source-specific metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Enriched fields (populated by processing pipeline)
    sentiment_simple: SentimentSimple | None = None
    sentiment_rich: SentimentRich | None = None
    content_type: ContentType | None = None
    topics: list[str] = Field(default_factory=list)
    
    # Vector reference
    embedding_id: str | None = None
    
    class Config:
        use_enum_values = True


class IngestionState(BaseModel):
    """
    Tracks ingestion progress for each source.
    Used to implement incremental fetching.
    """
    source: Source
    last_fetched_at: datetime
    last_item_id: str | None = None
    cursor: str | None = None  # API pagination cursor
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class DailyMetrics(BaseModel):
    """
    Aggregated daily metrics for dashboard performance.
    Pre-computed to avoid expensive queries.
    """
    date: datetime
    source: Source
    
    # Counts
    total_messages: int = 0
    
    # Simple sentiment
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    
    # Rich sentiment
    frustration_count: int = 0
    confusion_count: int = 0
    delight_count: int = 0
    gratitude_count: int = 0
    
    # Content types
    questions_count: int = 0
    bug_reports_count: int = 0
    feature_requests_count: int = 0
    
    # Topics (JSON blob)
    top_topics: dict[str, int] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class TrainingExample(BaseModel):
    """
    Training example for fine-tuning.
    Follows Alpaca/instruction format.
    """
    instruction: str
    input: str  # User query
    output: str  # Expected response
    source: str | None = None  # Where this example came from
    
    def to_chat_format(self) -> list[dict[str, str]]:
        """Convert to chat format for training."""
        messages = [{"role": "system", "content": self.instruction}]
        if self.input:
            messages.append({"role": "user", "content": self.input})
        messages.append({"role": "assistant", "content": self.output})
        return messages


class VectorRecord(BaseModel):
    """
    Schema for LanceDB vector storage.
    """
    id: str
    text: str
    vector: list[float]
    source: str
    created_at: datetime
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
