"""
Reddit ingester for Modal-related discussions.

Searches for Modal mentions across relevant subreddits.
"""

from datetime import datetime
from typing import Iterator

import httpx
import structlog

from .base import RateLimitedIngester
from src.storage.schemas import Message, IngestionState, Source, ContentType

logger = structlog.get_logger()


class RedditIngester(RateLimitedIngester):
    """
    Ingests Reddit posts and comments mentioning Modal.
    
    Uses Reddit's API (requires app credentials).
    """
    
    source = Source.REDDIT
    requests_per_minute = 30  # Reddit's rate limit
    
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
    ]
    
    # Search terms
    SEARCH_TERMS = ["modal.com", "modal labs", "modal-labs"]
    
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"
    API_URL = "https://oauth.reddit.com"
    
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        """
        Initialize Reddit ingester.
        
        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
        """
        super().__init__()
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: str | None = None
        
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "ModalSentimentBot/1.0"},
        )
    
    def _authenticate(self) -> bool:
        """Get OAuth access token."""
        if not self.client_id or not self.client_secret:
            self.logger.warning("No Reddit credentials provided")
            return False
        
        try:
            response = self.client.post(
                self.AUTH_URL,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "ModalSentimentBot/1.0"},
            )
            response.raise_for_status()
            self._access_token = response.json()["access_token"]
            
            # Update client headers
            self.client.headers["Authorization"] = f"Bearer {self._access_token}"
            return True
        except httpx.HTTPError as e:
            self.logger.error("Failed to authenticate with Reddit", error=str(e))
            return False
    
    def fetch(
        self,
        state: IngestionState | None = None,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """Fetch Reddit posts and comments about Modal."""
        if not self._authenticate():
            return
        
        count = 0
        
        for subreddit in self.SUBREDDITS:
            for term in self.SEARCH_TERMS:
                for msg in self._search_subreddit(subreddit, term):
                    yield msg
                    count += 1
                    if limit and count >= limit:
                        return
        
        self.log_progress(count)
    
    def _search_subreddit(
        self,
        subreddit: str,
        query: str,
    ) -> Iterator[Message]:
        """Search a subreddit for Modal-related posts."""
        try:
            response = self.client.get(
                f"{self.API_URL}/r/{subreddit}/search",
                params={
                    "q": query,
                    "sort": "new",
                    "limit": 100,
                    "restrict_sr": True,
                    "type": "link",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(
                "Failed to search subreddit",
                subreddit=subreddit,
                error=str(e),
            )
            return
        
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        
        for post_data in posts:
            post = post_data.get("data", {})
            
            # Yield the post
            yield self._post_to_message(post)
            
            # Fetch comments
            if post.get("num_comments", 0) > 0:
                for comment_msg in self._fetch_comments(post["id"], post["subreddit"]):
                    yield comment_msg
    
    def _fetch_comments(
        self,
        post_id: str,
        subreddit: str,
    ) -> Iterator[Message]:
        """Fetch comments for a post."""
        try:
            response = self.client.get(
                f"{self.API_URL}/r/{subreddit}/comments/{post_id}",
                params={"limit": 100, "depth": 5},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.logger.warning(
                "Failed to fetch comments",
                post_id=post_id,
                error=str(e),
            )
            return
        
        # Reddit returns [post, comments] structure
        data = response.json()
        if len(data) < 2:
            return
        
        comments_data = data[1].get("data", {}).get("children", [])
        
        for comment_data in comments_data:
            comment = comment_data.get("data", {})
            if comment.get("body"):  # Skip deleted/removed
                yield self._comment_to_message(comment, post_id)
    
    def _post_to_message(self, post: dict) -> Message:
        """Convert a Reddit post to a Message."""
        source_id = f"post_{post['id']}"
        
        # Combine title and selftext
        content = post.get("title", "")
        if post.get("selftext"):
            content += f"\n\n{post['selftext']}"
        
        # Determine content type from flair or title
        content_type = ContentType.DISCUSSION
        title_lower = post.get("title", "").lower()
        if "?" in title_lower or any(w in title_lower for w in ["how", "why", "what", "help"]):
            content_type = ContentType.QUESTION
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=content,
            title=post.get("title"),
            author=post.get("author"),
            url=f"https://reddit.com{post.get('permalink', '')}",
            created_at=datetime.fromtimestamp(post.get("created_utc", 0)),
            content_type=content_type,
            metadata={
                "subreddit": post.get("subreddit"),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                "flair": post.get("link_flair_text"),
            },
        )
    
    def _comment_to_message(self, comment: dict, post_id: str) -> Message:
        """Convert a Reddit comment to a Message."""
        source_id = f"comment_{comment['id']}"
        parent_id = self.create_message_id(f"post_{post_id}")
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=comment.get("body", ""),
            author=comment.get("author"),
            url=f"https://reddit.com{comment.get('permalink', '')}",
            parent_id=parent_id,
            thread_id=parent_id,
            created_at=datetime.fromtimestamp(comment.get("created_utc", 0)),
            metadata={
                "subreddit": comment.get("subreddit"),
                "score": comment.get("score", 0),
                "post_id": post_id,
            },
        )
    
    def get_initial_state(self) -> IngestionState:
        """Get initial state for Reddit ingestion."""
        return IngestionState(
            source=self.source,
            last_fetched_at=datetime.utcnow(),
            metadata={"subreddits_searched": []},
        )
