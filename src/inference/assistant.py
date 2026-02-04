"""
RAG-powered Modal support assistant.

Combines:
- Vector search (LanceDB) for relevant context
- Fine-tuned Qwen3 for Modal-specific responses
"""

from typing import Iterator

import structlog

logger = structlog.get_logger()


class ModalAssistant:
    """
    RAG-powered assistant for answering Modal questions.
    
    Uses:
    - LanceDB for semantic search over docs/community content
    - Fine-tuned (or base) Qwen3 for generation
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant specializing in Modal, a cloud computing platform that makes it easy to run code in the cloud.

You have access to relevant documentation and community discussions to help answer questions.

When answering:
1. Be accurate and helpful
2. Include code examples when relevant
3. Reference the provided context when applicable
4. If you're unsure, say so rather than guessing
5. Keep responses concise but complete

Context from Modal documentation and community:
{context}
"""
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        embedding_model=None,
        vector_store=None,
    ):
        """
        Initialize the assistant.
        
        Args:
            model: Pre-loaded language model
            tokenizer: Pre-loaded tokenizer
            embedding_model: Embedding model for queries
            vector_store: LanceDB store for retrieval
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.logger = logger.bind(component="assistant")
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve relevant context for a query.
        """
        if self.embedding_model is None or self.vector_store is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Search for similar documents
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k,
        )
        
        return results
    
    def format_context(self, results: list[dict]) -> str:
        """Format retrieved results into context string."""
        if not results:
            return "No specific context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get("text", "")[:500]  # Truncate long texts
            source = result.get("source", "unknown")
            url = result.get("url", "")
            
            part = f"[{i}] Source: {source}"
            if url:
                part += f"\nURL: {url}"
            part += f"\n{text}"
            context_parts.append(part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def answer(
        self,
        question: str,
        use_rag: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        """
        Generate an answer to a question.
        
        Args:
            question: The user's question
            use_rag: Whether to use RAG for context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with 'answer', 'sources', and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Retrieve context if using RAG
        sources = []
        context = "No specific context available."
        
        if use_rag:
            results = self.retrieve_context(question)
            if results:
                context = self.format_context(results)
                sources = [
                    {
                        "text": r.get("text", "")[:200],
                        "source": r.get("source"),
                        "url": r.get("url"),
                    }
                    for r in results
                ]
        
        # Build prompt
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        # Generate response
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return {
            "answer": response.strip(),
            "sources": sources,
            "model": "fine-tuned" if hasattr(self.model, "peft_config") else "base",
            "used_rag": use_rag and bool(sources),
        }
    
    def answer_stream(
        self,
        question: str,
        use_rag: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """
        Stream answer tokens for real-time display.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        from transformers import TextIteratorStreamer
        import threading
        
        # Retrieve context
        context = "No specific context available."
        if use_rag:
            results = self.retrieve_context(question)
            if results:
                context = self.format_context(results)
        
        # Build prompt
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Generate in background thread
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }
        
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they're generated
        for token in streamer:
            yield token
        
        thread.join()


def load_assistant(
    model_path: str | None = None,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    vector_store_path: str | None = None,
    use_quantization: bool = True,
) -> ModalAssistant:
    """
    Load the Modal assistant with all components.
    
    Args:
        model_path: Path to fine-tuned model (or None for base)
        base_model: Base model name
        embedding_model: Embedding model name
        vector_store_path: Path to LanceDB
        use_quantization: Use 4-bit quantization for inference
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.processing.embeddings import EmbeddingGenerator
    from src.storage.lancedb_store import LanceDBStore
    
    logger.info("Loading Modal assistant")
    
    # Load tokenizer
    model_to_load = model_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    
    # Load model
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Load embedding model
    embedding_gen = None
    if embedding_model:
        embedding_gen = EmbeddingGenerator(model_name=embedding_model)
        _ = embedding_gen.model  # Force load
    
    # Load vector store
    vector_store = None
    if vector_store_path:
        vector_store = LanceDBStore(vector_store_path)
    
    return ModalAssistant(
        model=model,
        tokenizer=tokenizer,
        embedding_model=embedding_gen,
        vector_store=vector_store,
    )
