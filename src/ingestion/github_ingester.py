"""
GitHub Issues and Discussions ingester.

Fetches issues and discussions from Modal's GitHub repositories
for sentiment analysis and Q&A extraction.
"""

from datetime import datetime
from typing import Iterator

import httpx
import structlog

from .base import BaseIngester, RateLimitedIngester
from src.storage.schemas import Message, IngestionState, Source, ContentType

logger = structlog.get_logger()


class GitHubIngester(RateLimitedIngester):
    """
    Ingests GitHub Issues and Discussions from Modal repositories.
    
    Uses GitHub's GraphQL API for efficient fetching.
    """
    
    source = Source.GITHUB
    requests_per_minute = 30  # Conservative rate limit
    
    # Modal's main repositories
    REPOS = [
        ("modal-labs", "modal-client"),
        ("modal-labs", "modal-examples"),
    ]
    
    GRAPHQL_URL = "https://api.github.com/graphql"
    REST_URL = "https://api.github.com"
    
    def __init__(self, token: str | None = None):
        """
        Initialize the GitHub ingester.
        
        Args:
            token: GitHub personal access token (optional but recommended)
        """
        super().__init__()
        self.token = token
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ModalSentimentBot/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        self.client = httpx.Client(
            timeout=30.0,
            headers=headers,
        )
    
    def fetch(
        self,
        state: IngestionState | None = None,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """
        Fetch issues and discussions from Modal repositories.
        """
        count = 0
        since = state.last_fetched_at if state else None
        
        for owner, repo in self.REPOS:
            self.logger.info("Fetching from repo", owner=owner, repo=repo)
            
            # Fetch issues
            for msg in self._fetch_issues(owner, repo, since):
                yield msg
                count += 1
                if limit and count >= limit:
                    return
            
            # Fetch discussions (if enabled on repo)
            for msg in self._fetch_discussions(owner, repo, since):
                yield msg
                count += 1
                if limit and count >= limit:
                    return
        
        self.log_progress(count)
    
    def _fetch_issues(
        self,
        owner: str,
        repo: str,
        since: datetime | None = None,
    ) -> Iterator[Message]:
        """Fetch issues from a repository using REST API."""
        page = 1
        per_page = 100
        
        while True:
            params = {
                "state": "all",
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
            if since:
                params["since"] = since.isoformat()
            
            try:
                response = self.client.get(
                    f"{self.REST_URL}/repos/{owner}/{repo}/issues",
                    params=params,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                self.logger.error("Failed to fetch issues", error=str(e))
                break
            
            issues = response.json()
            
            if not issues:
                break
            
            for issue in issues:
                # Skip pull requests (they appear in issues endpoint)
                if "pull_request" in issue:
                    continue
                
                # Yield the issue itself
                yield self._issue_to_message(owner, repo, issue)
                
                # Fetch and yield comments
                if issue.get("comments", 0) > 0:
                    for comment_msg in self._fetch_issue_comments(
                        owner, repo, issue["number"]
                    ):
                        yield comment_msg
            
            # Check if there are more pages
            if len(issues) < per_page:
                break
            
            page += 1
    
    def _fetch_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
    ) -> Iterator[Message]:
        """Fetch comments for a specific issue."""
        page = 1
        per_page = 100
        
        while True:
            try:
                response = self.client.get(
                    f"{self.REST_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments",
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                self.logger.warning(
                    "Failed to fetch comments",
                    issue=issue_number,
                    error=str(e),
                )
                break
            
            comments = response.json()
            
            if not comments:
                break
            
            for comment in comments:
                yield self._comment_to_message(owner, repo, issue_number, comment)
            
            if len(comments) < per_page:
                break
            
            page += 1
    
    def _fetch_discussions(
        self,
        owner: str,
        repo: str,
        since: datetime | None = None,
    ) -> Iterator[Message]:
        """Fetch discussions using GraphQL API."""
        if not self.token:
            self.logger.info("Skipping discussions (no token)")
            return
        
        query = """
        query($owner: String!, $repo: String!, $cursor: String) {
            repository(owner: $owner, name: $repo) {
                discussions(first: 50, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                    nodes {
                        id
                        number
                        title
                        body
                        author {
                            login
                        }
                        createdAt
                        updatedAt
                        url
                        category {
                            name
                        }
                        comments(first: 50) {
                            nodes {
                                id
                                body
                                author {
                                    login
                                }
                                createdAt
                                url
                            }
                        }
                    }
                }
            }
        }
        """
        
        cursor = None
        
        while True:
            try:
                response = self.client.post(
                    self.GRAPHQL_URL,
                    json={
                        "query": query,
                        "variables": {
                            "owner": owner,
                            "repo": repo,
                            "cursor": cursor,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as e:
                self.logger.error("Failed to fetch discussions", error=str(e))
                break
            
            if "errors" in data:
                # Discussions might not be enabled
                self.logger.info(
                    "GraphQL errors (discussions may not be enabled)",
                    errors=data["errors"],
                )
                break
            
            discussions_data = data.get("data", {}).get("repository", {}).get("discussions")
            if not discussions_data:
                break
            
            discussions = discussions_data.get("nodes", [])
            
            for discussion in discussions:
                # Check if updated since last fetch
                updated_at = datetime.fromisoformat(
                    discussion["updatedAt"].replace("Z", "+00:00")
                )
                if since and updated_at < since:
                    return
                
                # Yield the discussion
                yield self._discussion_to_message(owner, repo, discussion)
                
                # Yield comments
                for comment in discussion.get("comments", {}).get("nodes", []):
                    yield self._discussion_comment_to_message(
                        owner, repo, discussion["number"], comment
                    )
            
            page_info = discussions_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            
            cursor = page_info.get("endCursor")
    
    def _issue_to_message(self, owner: str, repo: str, issue: dict) -> Message:
        """Convert a GitHub issue to a Message."""
        source_id = f"{owner}_{repo}_issue_{issue['number']}"
        
        # Determine content type
        content_type = ContentType.DISCUSSION
        labels = [l["name"].lower() for l in issue.get("labels", [])]
        if any(l in ["bug", "bug-report"] for l in labels):
            content_type = ContentType.BUG_REPORT
        elif any(l in ["enhancement", "feature", "feature-request"] for l in labels):
            content_type = ContentType.FEATURE_REQUEST
        elif any(l in ["question", "help-wanted"] for l in labels):
            content_type = ContentType.QUESTION
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=f"{issue['title']}\n\n{issue.get('body', '')}",
            title=issue["title"],
            author=issue.get("user", {}).get("login"),
            url=issue["html_url"],
            created_at=datetime.fromisoformat(
                issue["created_at"].replace("Z", "+00:00")
            ),
            content_type=content_type,
            metadata={
                "repo": f"{owner}/{repo}",
                "type": "issue",
                "number": issue["number"],
                "state": issue["state"],
                "labels": labels,
                "reactions": issue.get("reactions", {}),
            },
        )
    
    def _comment_to_message(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        comment: dict,
    ) -> Message:
        """Convert a GitHub issue comment to a Message."""
        source_id = f"{owner}_{repo}_issue_{issue_number}_comment_{comment['id']}"
        parent_id = self.create_message_id(f"{owner}_{repo}_issue_{issue_number}")
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=comment.get("body", ""),
            author=comment.get("user", {}).get("login"),
            url=comment["html_url"],
            parent_id=parent_id,
            thread_id=parent_id,
            created_at=datetime.fromisoformat(
                comment["created_at"].replace("Z", "+00:00")
            ),
            metadata={
                "repo": f"{owner}/{repo}",
                "type": "issue_comment",
                "issue_number": issue_number,
            },
        )
    
    def _discussion_to_message(
        self,
        owner: str,
        repo: str,
        discussion: dict,
    ) -> Message:
        """Convert a GitHub discussion to a Message."""
        source_id = f"{owner}_{repo}_discussion_{discussion['number']}"
        
        # Map category to content type
        category = discussion.get("category", {}).get("name", "").lower()
        content_type = ContentType.DISCUSSION
        if "q&a" in category or "question" in category:
            content_type = ContentType.QUESTION
        elif "ideas" in category or "feature" in category:
            content_type = ContentType.FEATURE_REQUEST
        elif "announcement" in category:
            content_type = ContentType.ANNOUNCEMENT
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=f"{discussion['title']}\n\n{discussion.get('body', '')}",
            title=discussion["title"],
            author=discussion.get("author", {}).get("login"),
            url=discussion["url"],
            created_at=datetime.fromisoformat(
                discussion["createdAt"].replace("Z", "+00:00")
            ),
            content_type=content_type,
            metadata={
                "repo": f"{owner}/{repo}",
                "type": "discussion",
                "number": discussion["number"],
                "category": category,
            },
        )
    
    def _discussion_comment_to_message(
        self,
        owner: str,
        repo: str,
        discussion_number: int,
        comment: dict,
    ) -> Message:
        """Convert a GitHub discussion comment to a Message."""
        source_id = f"{owner}_{repo}_discussion_{discussion_number}_comment_{comment['id']}"
        parent_id = self.create_message_id(
            f"{owner}_{repo}_discussion_{discussion_number}"
        )
        
        return Message(
            id=self.create_message_id(source_id),
            source=self.source,
            source_id=source_id,
            content=comment.get("body", ""),
            author=comment.get("author", {}).get("login"),
            url=comment.get("url"),
            parent_id=parent_id,
            thread_id=parent_id,
            created_at=datetime.fromisoformat(
                comment["createdAt"].replace("Z", "+00:00")
            ),
            metadata={
                "repo": f"{owner}/{repo}",
                "type": "discussion_comment",
                "discussion_number": discussion_number,
            },
        )
    
    def get_initial_state(self) -> IngestionState:
        """Get initial state for GitHub ingestion."""
        return IngestionState(
            source=self.source,
            last_fetched_at=datetime.utcnow(),
            metadata={"repos_completed": []},
        )
