"""
Modal blog and GitHub documentation ingester.

Since modal.com/docs blocks scrapers, we fetch:
- Blog posts from modal.com/blog (works)
- Documentation from GitHub repos (modal-examples, modal-client)
"""

from datetime import datetime
from typing import Iterator
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import structlog

from .base import BaseIngester
from src.storage.schemas import Message, IngestionState, Source, ContentType

logger = structlog.get_logger()


class DocsIngester(BaseIngester):
    """
    Ingests Modal blog posts and documentation from GitHub.
    
    Sources:
    - https://modal.com/blog (blog posts)
    - GitHub: modal-labs/modal-examples (READMEs, Python files)
    """
    
    source = Source.DOCS
    
    BLOG_BASE_URL = "https://modal.com/blog"
    GITHUB_API = "https://api.github.com"
    
    # GitHub repos to fetch docs from
    GITHUB_REPOS = [
        ("modal-labs", "modal-examples"),
    ]
    
    def __init__(self, github_token: str | None = None):
        super().__init__()
        self.github_token = github_token
        
        # HTTP client for blog (browser-like headers)
        self.web_client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        
        # GitHub API client
        gh_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ModalSentimentBot/1.0",
        }
        if github_token:
            gh_headers["Authorization"] = f"Bearer {github_token}"
        self.github_client = httpx.Client(timeout=30.0, headers=gh_headers)
        
        self._visited_urls: set[str] = set()
    
    def fetch(
        self,
        state: IngestionState | None = None,
        limit: int | None = None,
    ) -> Iterator[Message]:
        """
        Fetch Modal docs from blog and GitHub.
        """
        count = 0
        
        # Fetch from GitHub repos (READMEs, Python files with docstrings)
        self.logger.info("Starting GitHub docs fetch")
        for owner, repo in self.GITHUB_REPOS:
            for msg in self._fetch_github_docs(owner, repo):
                yield msg
                count += 1
                if limit and count >= limit:
                    return
        
        # Fetch blog posts
        self.logger.info("Starting blog crawl")
        for msg in self._crawl_blog():
            yield msg
            count += 1
            if limit and count >= limit:
                return
        
        self.log_progress(count)
    
    def _fetch_github_docs(self, owner: str, repo: str) -> Iterator[Message]:
        """Fetch documentation files from a GitHub repository."""
        try:
            # Get repo contents (recursive) - this uses minimal API quota
            response = self.github_client.get(
                f"{self.GITHUB_API}/repos/{owner}/{repo}/git/trees/main?recursive=1"
            )
            response.raise_for_status()
            tree = response.json()
        except httpx.HTTPError as e:
            self.logger.warning("Failed to fetch GitHub tree", repo=f"{owner}/{repo}", error=str(e))
            return
        
        # Filter for interesting files
        doc_files = []
        for item in tree.get("tree", []):
            path = item.get("path", "")
            # README files
            if path.lower().endswith(("readme.md", "readme.rst", "readme.txt")):
                doc_files.append(path)
            # Python files in example directories (contain docstrings)
            elif path.endswith(".py") and "/" in path:
                # Skip test files, __init__.py, etc
                if not any(x in path.lower() for x in ["test_", "_test.py", "__init__", "__pycache__"]):
                    doc_files.append(path)
        
        self.logger.info("Found doc files", repo=f"{owner}/{repo}", count=len(doc_files))
        
        for path in doc_files[:50]:  # Limit to 50 files per repo
            try:
                # Use raw.githubusercontent.com - no API rate limits!
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
                response = self.web_client.get(raw_url)
                response.raise_for_status()
                content = response.text
                
                if len(content) < 100:
                    continue
                
                # For Python files, try to extract the module docstring
                if path.endswith(".py"):
                    content = self._extract_python_docs(content, path)
                    if not content:
                        continue
                
                source_id = f"github_{owner}_{repo}_{path.replace('/', '_')}"
                
                yield Message(
                    id=self.create_message_id(source_id),
                    source=Source.DOCS,
                    source_id=source_id,
                    content=content,
                    title=path.split("/")[-1],
                    url=f"https://github.com/{owner}/{repo}/blob/main/{path}",
                    created_at=datetime.utcnow(),
                    content_type=ContentType.DOCUMENTATION,
                    metadata={
                        "repo": f"{owner}/{repo}",
                        "path": path,
                        "type": "github_file",
                    }
                )
            except Exception as e:
                self.logger.warning("Failed to fetch file", path=path, error=str(e))
    
    def _extract_python_docs(self, content: str, path: str) -> str | None:
        """Extract documentation from a Python file."""
        lines = content.split("\n")
        result_parts = []
        
        # Get module docstring if exists
        in_docstring = False
        docstring_lines = []
        for i, line in enumerate(lines[:50]):  # Only check first 50 lines for module docstring
            stripped = line.strip()
            if not in_docstring and stripped.startswith('"""'):
                in_docstring = True
                # Check if single-line docstring
                if stripped.endswith('"""') and len(stripped) > 3:
                    docstring_lines.append(stripped[3:-3])
                    break
                docstring_lines.append(stripped[3:])
            elif in_docstring:
                if stripped.endswith('"""'):
                    docstring_lines.append(stripped[:-3])
                    break
                docstring_lines.append(line)
        
        if docstring_lines:
            result_parts.append("\n".join(docstring_lines).strip())
        
        # Also include the whole file if it's a good example (has Modal imports)
        if "import modal" in content or "from modal" in content:
            result_parts.append(f"\n\n--- Code Example: {path} ---\n\n{content}")
        
        return "\n".join(result_parts) if result_parts else None
    
    def _crawl_blog(self) -> Iterator[Message]:
        """Crawl the Modal blog."""
        try:
            response = self.web_client.get(self.BLOG_BASE_URL)
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.logger.error("Failed to fetch blog index", error=str(e))
            return
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find blog post links (adjust selector based on actual HTML structure)
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/blog/" in href and href != "/blog/":
                full_url = urljoin(self.BLOG_BASE_URL, href)
                if full_url not in self._visited_urls:
                    self._visited_urls.add(full_url)
                    
                    try:
                        post_response = self.web_client.get(full_url)
                        post_response.raise_for_status()
                        post_soup = BeautifulSoup(post_response.text, "html.parser")
                        
                        message = self._extract_page_content(full_url, post_soup, is_docs=False)
                        if message:
                            yield message
                    except httpx.HTTPError as e:
                        self.logger.warning("Failed to fetch blog post", url=full_url, error=str(e))
    
    def _extract_page_content(
        self,
        url: str,
        soup: BeautifulSoup,
        is_docs: bool,
    ) -> Message | None:
        """Extract content from a page."""
        # Try to find the main content area
        # These selectors may need adjustment based on Modal's actual HTML structure
        content_selectors = [
            "article",
            "main",
            ".content",
            ".docs-content",
            ".markdown-body",
            "[role='main']",
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            content_element = soup.body
        
        if not content_element:
            return None
        
        # Extract title
        title = None
        title_element = soup.find("h1") or soup.find("title")
        if title_element:
            title = title_element.get_text(strip=True)
        
        # Extract text content
        # Remove script and style elements
        for element in content_element.find_all(["script", "style", "nav", "footer"]):
            element.decompose()
        
        text = content_element.get_text(separator="\n", strip=True)
        
        # Skip if too short
        if len(text) < 100:
            return None
        
        # Create unique ID from URL path
        parsed = urlparse(url)
        source_id = parsed.path.strip("/").replace("/", "_") or "index"
        
        return Message(
            id=self.create_message_id(source_id),
            source=Source.DOCS if is_docs else Source.BLOG,
            source_id=source_id,
            content=text,
            title=title,
            url=url,
            created_at=datetime.utcnow(),  # Docs don't have dates, use fetch time
            content_type=ContentType.DOCUMENTATION,
            metadata={
                "is_docs": is_docs,
                "section": parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else None,
            }
        )
    
    def _should_crawl(self, url: str) -> bool:
        """Check if a URL should be crawled."""
        parsed = urlparse(url)
        
        # Only crawl modal.com
        if parsed.netloc and parsed.netloc != "modal.com":
            return False
        
        # Only crawl docs paths
        if not parsed.path.startswith("/docs"):
            return False
        
        # Skip anchors, query params
        if parsed.fragment:
            return False
        
        # Skip certain file types
        skip_extensions = [".png", ".jpg", ".gif", ".pdf", ".zip"]
        if any(parsed.path.endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    def get_initial_state(self) -> IngestionState:
        """Get initial state for docs ingestion."""
        return IngestionState(
            source=self.source,
            last_fetched_at=datetime.utcnow(),
            metadata={"sections_completed": []}
        )
