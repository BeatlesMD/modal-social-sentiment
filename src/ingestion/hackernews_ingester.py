"""
Hacker News ingester for Modal-related discussions.

Uses the HN Algolia API for search and the Firebase API for items.
"""

from datetime import datetime
from typing import Iterator

import httpx
import structlog

from .base import BaseIngester
from src.storage.schemas import Message, IngestionState, Source, ContentType

logger = structlog.get_logger()


class HackerNewsIngester(BaseIngester):
    """
    Ingests Hacker News posts and comments mentioning Modal.
    
    Uses:
    - Algolia HN Search API for finding stories
    - HN Firebase API for fetching items and comments
    """
    
    source = Source.HACKERNEWS
    
    SEARCH_URL = "https://hn.algolia.com/api/v1/search"
    ITEM_URL = "https://hacker-news.firebaseio.com/v0/item"
    
    SEARCH_TERMS = ["modal.com", "modal labs"]
    
    def __init__(self):
        super().__init__()
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "ModalSentimentBot/1.0"},
        )
    
    def fetch(
        self,
        state: IngestionState | None = None,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """Fetch HN stories and comments about Modal."""
        count = 0
        seen_story_ids: set[int] = set()
        
        for term in self.SEARCH_TERMS:
            for msg in self._search_stories(term, seen_story_ids):
                yield msg
                count += 1
                if limit and count >= limit:
                    return
        
        self.log_progress(count)
    
    def _search_stories(
        self,
        query: str,
        seen_ids: set[int],
    ) -> Iterator[Message]:
        """Search for stories matching the query."""
        page = 0
        
        while True:
            try:
                response = self.client.get(
                    self.SEARCH_URL,
                    params={
                        "query": query,
                        "tags": "story",
                        "page": page,
                        "hitsPerPage": 50,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                self.logger.error("Failed to search HN", error=str(e))
                break
            
            data = response.json()
            hits = data.get("hits", [])
            
            if not hits:
                break
            
            for hit in hits:
                story_id = hit.get("objectID")
                if story_id and int(story_id) not in seen_ids:
                    seen_ids.add(int(story_id))
                    
                    # Yield the story
                    yield self._hit_to_message(hit)
                    
                    # Fetch and yield comments
                    for comment_msg in self._fetch_comments(int(story_id)):
                        yield comment_msg
            
            # Check if more pages
            if page >= data.get("nbPages", 1) - 1:
                break
            
            page += 1
    
    def _fetch_comments(
        self,
        story_id: int,
        max_depth: int = 3,
    ) -> Iterator[Message]:
        """Recursively fetch comments for a story."""
        try:
            response = self.client.get(f"{self.ITEM_URL}/{story_id}.json")
            response.raise_for_status()
            story = response.json()
        except httpx.HTTPError as e:
            self.logger.warning("Failed to fetch story", story_id=story_id, error=str(e))
            return
        
        if not story:
            return
        
        kids = story.get("kids", [])
        for comment_id in kids[:50]:  # Limit comments per story
            for msg in self._fetch_comment_tree(
                comment_id,
                story_id,
                depth=0,
                max_depth=max_depth,
            ):
                yield msg
    
    def _fetch_comment_tree(
        self,
        comment_id: int,
        story_id: int,
        depth: int,
        max_depth: int,
    ) -> Iterator[Message]:
        """Recursively fetch a comment and its children."""
        if depth > max_depth:
            return
        
        try:
            response = self.client.get(f"{self.ITEM_URL}/{comment_id}.json")
            response.raise_for_status()
            comment = response.json()
        except httpx.HTTPError:
            return
        
        if not comment or comment.get("deleted") or comment.get("dead"):
            return
        
        yield self._comment_to_message(comment, story_id)
        
        # Recursively fetch child comments
        for child_id in comment.get("kids", [])[:20]:
            for msg in self._fetch_comment_tree(
                child_id,
                story_id,
                depth + 1,
                max_depth,
            ):
                yield msg
    
    def _hit_to_message(self, hit: dict) -> Message:
        """Convert an Algolia search hit to a Message."""
        source_id = f"story_{hit['objectID']}"
        
        content = hit.get("title", "")
        if hit.get("story_text"):
            content += f"\n\n{hit['story_text']}"
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=content,
            title=hit.get("title"),
            author=hit.get("author"),
            url=hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
            created_at=datetime.fromtimestamp(hit.get("created_at_i", 0)),
            content_type=ContentType.DISCUSSION,
            metadata={
                "points": hit.get("points", 0),
                "num_comments": hit.get("num_comments", 0),
                "hn_url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
            },
        )
    
    def _comment_to_message(self, comment: dict, story_id: int) -> Message:
        """Convert an HN comment to a Message."""
        source_id = f"comment_{comment['id']}"
        thread_id = self.create_message_id(f"story_{story_id}")
        parent_id = thread_id
        if comment.get("parent") and comment["parent"] != story_id:
            parent_id = self.create_message_id(f"comment_{comment['parent']}")
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=comment.get("text", ""),
            author=comment.get("by"),
            url=f"https://news.ycombinator.com/item?id={comment['id']}",
            parent_id=parent_id,
            thread_id=thread_id,
            created_at=datetime.fromtimestamp(comment.get("time", 0)),
            metadata={
                "story_id": story_id,
            },
        )
    
    def get_initial_state(self) -> IngestionState:
        """Get initial state for HN ingestion."""
        return IngestionState(
            source=self.source,
            last_fetched_at=datetime.utcnow(),
            metadata={},
        )
