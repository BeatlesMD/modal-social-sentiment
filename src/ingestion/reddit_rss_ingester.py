"""
Reddit RSS ingester - no API key required.

Uses Reddit's public RSS feeds to search for Modal mentions.
Limited compared to API but works without authentication.
"""

from datetime import datetime
from typing import Iterator
from xml.etree import ElementTree
import re

import httpx
import structlog

from .base import BaseIngester
from src.storage.schemas import Message, IngestionState, Source, ContentType

logger = structlog.get_logger()


class RedditRSSIngester(BaseIngester):
    """
    Ingests Reddit posts via public RSS feeds.
    
    No authentication required, but limited to:
    - 25 posts per feed
    - No comments (posts only)
    - May be rate limited if abused
    """
    
    source = Source.REDDIT
    
    # Subreddits to search
    SUBREDDITS = [
        "MachineLearning",
        "LocalLLaMA",
        "Python",
        "programming",
        "devops",
        "mlops",
        "learnmachinelearning",
        "deeplearning",
        "datascience",
        "singularity",
        "LanguageTechnology",
    ]
    
    # Search terms - Modal cloud platform specific
    SEARCH_TERMS = [
        "modal.com",
        "modal labs", 
        '"modal" serverless',
        '"modal" gpu cloud',
        '"modal" python cloud',
    ]
    
    # Keywords that must appear in title or content for relevance filtering
    MODAL_KEYWORDS = [
        "modal.com",
        "modal labs",
        "modal cloud",
        "modal serverless",
        "modal gpu",
        "@modal",  # decorator
        "modal.function",
        "modal.app",
        "modal run",
        "modal deploy",
    ]
    
    def __init__(self):
        super().__init__()
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ModalSentimentBot/1.0)"
            },
            follow_redirects=True,
        )
    
    def fetch(
        self,
        state: IngestionState | None = None,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """Fetch Reddit posts via RSS feeds."""
        count = 0
        seen_ids = set()
        
        for subreddit in self.SUBREDDITS:
            for term in self.SEARCH_TERMS:
                for msg in self._fetch_rss_feed(subreddit, term, seen_ids):
                    if msg.id not in seen_ids:
                        seen_ids.add(msg.id)
                        yield msg
                        count += 1
                        if limit and count >= limit:
                            self.log_progress(count)
                            return
        
        self.log_progress(count)
    
    def _fetch_rss_feed(
        self,
        subreddit: str,
        query: str,
        seen_ids: set,
    ) -> Iterator[Message]:
        """Fetch and parse a single RSS feed."""
        # Reddit search RSS URL
        url = f"https://www.reddit.com/r/{subreddit}/search.rss"
        params = {
            "q": query,
            "sort": "new",
            "restrict_sr": "on",
            "t": "month",  # Last month
        }
        
        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(
                "Failed to fetch RSS feed",
                subreddit=subreddit,
                query=query,
                error=str(e),
            )
            return
        
        # Parse RSS/Atom feed
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as e:
            self.logger.warning("Failed to parse RSS", error=str(e))
            return
        
        # Handle Atom namespace (Reddit uses Atom format)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        for entry in root.findall("atom:entry", ns):
            try:
                msg = self._entry_to_message(entry, ns, subreddit)
                if msg:
                    yield msg
            except Exception as e:
                self.logger.warning("Failed to parse entry", error=str(e))
    
    def _is_modal_related(self, title: str, content: str) -> bool:
        """Check if the post is actually about Modal cloud platform.
        
        We need to be careful to exclude:
        - "multimodal" (ML term)
        - "modal dialog" (UI term)
        - General uses of "modal" unrelated to the cloud platform
        """
        text = f"{title} {content}".lower()
        
        # Strong signals - definitely Modal cloud platform
        strong_keywords = [
            "modal.com",
            "modal labs",
            "modal cloud",
            "modal serverless",
            "modal.run",
            "modal deploy",
            "@modal",  # decorator
            "modal.function",
            "modal.app",
            "modal.cls",
            "modal.volume",
            "modal.image",
            "pip install modal",
        ]
        
        for keyword in strong_keywords:
            if keyword in text:
                return True
        
        # Exclude common false positives
        false_positive_patterns = [
            "multimodal",
            "multi-modal", 
            "modal dialog",
            "modal window",
            "modal popup",
            "modal verb",  # linguistic term
            "bimodal",
            "unimodal",
            "intermodal",
        ]
        
        for pattern in false_positive_patterns:
            if pattern in text:
                # If it contains a false positive pattern but no strong signals, skip
                return False
        
        # If we get here and "modal" appears with cloud context, include it
        if " modal " in f" {text} " or text.startswith("modal ") or text.endswith(" modal"):
            cloud_context = any(w in text for w in [
                "serverless", "gpu cloud", "deploy", "container",
                "python cloud", "inference api"
            ])
            if cloud_context:
                return True
        
        return False
    
    def _entry_to_message(
        self,
        entry: ElementTree.Element,
        ns: dict,
        subreddit: str,
    ) -> Message | None:
        """Convert an RSS entry to a Message."""
        # Extract fields
        title_elem = entry.find("atom:title", ns)
        link_elem = entry.find("atom:link", ns)
        content_elem = entry.find("atom:content", ns)
        updated_elem = entry.find("atom:updated", ns)
        author_elem = entry.find("atom:author/atom:name", ns)
        id_elem = entry.find("atom:id", ns)
        
        if title_elem is None or id_elem is None:
            return None
        
        title = title_elem.text or ""
        link = link_elem.get("href", "") if link_elem is not None else ""
        content = content_elem.text or "" if content_elem is not None else ""
        author = author_elem.text if author_elem is not None else None
        
        # Clean HTML from content
        content = self._strip_html(content)
        
        # Filter: Only include Modal-related posts
        if not self._is_modal_related(title, content):
            self.logger.debug(
                "Skipping non-Modal post",
                title=title[:50],
                subreddit=subreddit,
            )
            return None
        
        # Parse date
        updated = updated_elem.text if updated_elem is not None else None
        created_at = self._parse_date(updated) if updated else datetime.utcnow()
        
        # Extract post ID from Reddit URL or ID
        reddit_id = id_elem.text or ""
        # ID format: t3_xxxxx (posts) or URL
        match = re.search(r't3_(\w+)', reddit_id) or re.search(r'/comments/(\w+)/', link)
        source_id = f"post_{match.group(1)}" if match else f"post_{hash(reddit_id)}"
        
        # Determine content type
        content_type = ContentType.DISCUSSION
        title_lower = title.lower()
        if "?" in title or any(w in title_lower for w in ["how", "why", "what", "help", "can i"]):
            content_type = ContentType.QUESTION
        
        # Combine title and content
        full_content = title
        if content and content != title:
            full_content += f"\n\n{content}"
        
        self.logger.info("Found Modal-related post", title=title[:60], subreddit=subreddit)
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=full_content,
            title=title,
            author=author,
            url=link,
            created_at=created_at,
            content_type=content_type,
            metadata={
                "subreddit": subreddit,
                "via": "rss",
            },
        )
    
    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Simple HTML tag removal
        clean = re.sub(r'<[^>]+>', '', text)
        # Decode common entities
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&quot;', '"')
        clean = clean.replace('&#39;', "'")
        clean = clean.replace('&nbsp;', ' ')
        return clean.strip()
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse ISO date string."""
        try:
            # Reddit uses ISO format: 2024-01-15T12:00:00+00:00
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            return datetime.fromisoformat(date_str.replace('+00:00', ''))
        except Exception:
            return datetime.utcnow()
    
    def get_initial_state(self) -> IngestionState:
        """Get initial state for Reddit RSS ingestion."""
        return IngestionState(
            source=self.source,
            last_fetched_at=datetime.utcnow(),
            metadata={"via": "rss", "subreddits_searched": self.SUBREDDITS},
        )
    
    def log_progress(self, count: int):
        """Log ingestion progress."""
        self.logger.info("Reddit RSS ingestion progress", posts_fetched=count)
