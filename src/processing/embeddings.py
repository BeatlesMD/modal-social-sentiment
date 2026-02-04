"""
Embedding generation for semantic search.

Uses sentence-transformers for fast, high-quality embeddings.
"""

from typing import Iterator

import structlog

logger = structlog.get_logger()


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers.
    
    Default model: BAAI/bge-small-en-v1.5
    - 384 dimensions
    - Fast inference
    - Good quality for retrieval
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model=None,
    ):
        self.model_name = model_name
        self._model = model
        self.logger = logger.bind(component="embeddings")
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer
        
        self.logger.info("Loading embedding model", model=self.model_name)
        return SentenceTransformer(self.model_name)
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for inference
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [e.tolist() for e in embeddings]
    
    def embed_batch_iter(
        self,
        texts: Iterator[str],
        batch_size: int = 32,
    ) -> Iterator[tuple[str, list[float]]]:
        """
        Generate embeddings for a stream of texts.
        
        Yields (text, embedding) tuples.
        """
        batch = []
        
        for text in texts:
            batch.append(text)
            
            if len(batch) >= batch_size:
                embeddings = self.embed_batch(batch, show_progress=False)
                for t, e in zip(batch, embeddings):
                    yield t, e
                batch = []
        
        # Process remaining
        if batch:
            embeddings = self.embed_batch(batch, show_progress=False)
            for t, e in zip(batch, embeddings):
                yield t, e
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


def load_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Helper to load embedding model in Modal function.
    """
    generator = EmbeddingGenerator(model_name)
    _ = generator.model  # Force load
    return generator
