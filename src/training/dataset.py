"""
Training dataset preparation for fine-tuning.

Extracts Q&A pairs from:
- Documentation (doc content → instructional format)
- GitHub Issues/Discussions (questions and answers)
- Community conversations
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import structlog

from src.storage.schemas import TrainingExample, Source, ContentType

logger = structlog.get_logger()


# System instruction for the fine-tuned model
MODAL_ASSISTANT_INSTRUCTION = """You are a helpful assistant specializing in Modal, a cloud computing platform that makes it easy to run code in the cloud. You have deep knowledge of:

- Modal's Python SDK and decorators (@app.function, @app.cls, etc.)
- Container images and dependencies
- GPU compute and machine learning workloads
- Volumes for persistent storage
- Secrets management
- Web endpoints and APIs
- Scheduled functions (cron jobs)
- Best practices for serverless computing

Provide accurate, helpful, and concise answers about Modal. When relevant, include code examples. If you're unsure about something, say so rather than guessing."""


class TrainingDatasetBuilder:
    """
    Builds training datasets from ingested data.
    
    Creates instruction-tuned examples in chat format.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="dataset")
    
    def build_from_docs(self, messages: list[dict]) -> list[TrainingExample]:
        """
        Create training examples from documentation.
        
        Converts doc sections into Q&A format:
        - Extract headings as questions
        - Use content as answers
        """
        examples = []
        
        for msg in messages:
            if msg.get("source") not in [Source.DOCS.value, Source.BLOG.value, "docs", "blog"]:
                continue
            
            content = msg.get("content", "")
            title = msg.get("title", "")
            
            # Skip very short or very long content
            if len(content) < 200 or len(content) > 10000:
                continue
            
            # Create instructional examples
            doc_examples = self._doc_to_examples(title, content)
            examples.extend(doc_examples)
        
        self.logger.info("Created doc examples", count=len(examples))
        return examples
    
    def build_from_conversations(self, messages: list[dict]) -> list[TrainingExample]:
        """
        Create training examples from community conversations.
        
        Looks for question-answer patterns in threads.
        """
        examples = []
        
        # Group messages by thread
        threads: dict[str, list[dict]] = {}
        for msg in messages:
            thread_id = msg.get("thread_id") or msg.get("id")
            if thread_id not in threads:
                threads[thread_id] = []
            threads[thread_id].append(msg)
        
        # Extract Q&A pairs from threads
        for thread_id, thread_msgs in threads.items():
            # Sort by creation time
            thread_msgs.sort(key=lambda x: x.get("created_at", datetime.min))
            
            if len(thread_msgs) < 2:
                continue
            
            # Look for Q&A patterns
            qa_examples = self._extract_qa_from_thread(thread_msgs)
            examples.extend(qa_examples)
        
        self.logger.info("Created conversation examples", count=len(examples))
        return examples
    
    def _doc_to_examples(self, title: str, content: str) -> list[TrainingExample]:
        """Convert a documentation page to training examples."""
        examples = []
        
        # Split content by headers
        sections = re.split(r'\n#{1,3}\s+', content)
        headers = re.findall(r'\n(#{1,3}\s+.+)', content)
        
        if not headers:
            # No headers, use the whole doc
            if title:
                question = self._title_to_question(title)
                if question:
                    examples.append(TrainingExample(
                        instruction=MODAL_ASSISTANT_INSTRUCTION,
                        input=question,
                        output=self._clean_content(content[:3000]),
                        source="docs",
                    ))
            return examples
        
        # Create example for each section
        for header, section in zip(headers, sections[1:]):
            header = header.strip().lstrip('#').strip()
            section = section.strip()
            
            if len(section) < 100:
                continue
            
            question = self._title_to_question(header)
            if question:
                examples.append(TrainingExample(
                    instruction=MODAL_ASSISTANT_INSTRUCTION,
                    input=question,
                    output=self._clean_content(section[:2000]),
                    source="docs",
                ))
        
        return examples
    
    def _extract_qa_from_thread(self, messages: list[dict]) -> list[TrainingExample]:
        """Extract Q&A pairs from a conversation thread."""
        examples = []
        
        # First message is usually the question
        first_msg = messages[0]
        
        # Check if it looks like a question
        content = first_msg.get("content", "")
        content_type = first_msg.get("content_type", "")
        
        is_question = (
            content_type in [ContentType.QUESTION.value, "question"] or
            "?" in content or
            any(w in content.lower()[:100] for w in ["how", "why", "what", "when", "where", "can i", "is it possible"])
        )
        
        if not is_question:
            return examples
        
        # Look for good answers (replies with substantial content)
        for reply in messages[1:]:
            reply_content = reply.get("content", "")
            
            # Skip short replies
            if len(reply_content) < 100:
                continue
            
            # Skip if it's another question
            if reply_content.count("?") > 2:
                continue
            
            # Create Q&A pair
            examples.append(TrainingExample(
                instruction=MODAL_ASSISTANT_INSTRUCTION,
                input=self._clean_content(content[:1500]),
                output=self._clean_content(reply_content[:2000]),
                source=first_msg.get("source", "conversation"),
            ))
            
            # Only take first good answer per question
            break
        
        return examples
    
    def _title_to_question(self, title: str) -> str | None:
        """Convert a doc title/header to a question."""
        title = title.strip()
        
        if not title or len(title) < 5:
            return None
        
        # Already a question
        if title.endswith("?"):
            return title
        
        # Convert imperative to question
        title_lower = title.lower()
        
        if title_lower.startswith(("how to", "getting started")):
            return f"How do I {title_lower.replace('how to ', '').replace('getting started with ', 'get started with ')}?"
        
        if title_lower.startswith(("using", "working with")):
            return f"How do I {title_lower.replace('using ', 'use ').replace('working with ', 'work with ')}?"
        
        if title_lower.startswith(("create", "build", "deploy", "run", "install", "configure")):
            return f"How do I {title_lower}?"
        
        # Generic conversion
        if any(title_lower.startswith(w) for w in ["the", "a ", "an "]):
            return f"What is {title_lower}?"
        
        return f"What is {title} in Modal?"
    
    def _clean_content(self, content: str) -> str:
        """Clean content for training."""
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        # Remove HTML tags if any
        content = re.sub(r'<[^>]+>', '', content)
        
        return content.strip()
    
    def save_dataset(
        self,
        examples: list[TrainingExample],
        filename: str = "training_data.jsonl",
        train_split: float = 0.9,
    ):
        """
        Save training examples to JSONL files.
        
        Creates train and validation splits.
        """
        import random
        
        # Shuffle examples
        examples = examples.copy()
        random.shuffle(examples)
        
        # Split
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Save train
        train_path = self.output_dir / filename
        with open(train_path, 'w') as f:
            for ex in train_examples:
                f.write(json.dumps({
                    "messages": ex.to_chat_format()
                }) + '\n')
        
        # Save validation
        val_path = self.output_dir / filename.replace('.jsonl', '_val.jsonl')
        with open(val_path, 'w') as f:
            for ex in val_examples:
                f.write(json.dumps({
                    "messages": ex.to_chat_format()
                }) + '\n')
        
        self.logger.info(
            "Saved training dataset",
            train_count=len(train_examples),
            val_count=len(val_examples),
            train_path=str(train_path),
            val_path=str(val_path),
        )
        
        return train_path, val_path
