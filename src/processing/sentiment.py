"""
Sentiment analysis using Qwen3 for classification.

Classifies messages into:
- Simple sentiment: positive, negative, neutral
- Rich sentiment: frustration, confusion, delight, etc.
- Content type: question, bug_report, feature_request, etc.
- Topics: gpu_availability, pricing, documentation, etc.
"""

import json
import structlog

logger = structlog.get_logger()


# System prompt for sentiment analysis
SENTIMENT_SYSTEM_PROMPT = """You are an expert at analyzing user feedback and community discussions about Modal, a cloud computing platform for running code in the cloud.

Analyze the given message and classify it according to the following criteria:

1. **sentiment_simple**: Overall sentiment
   - "positive": Praise, satisfaction, excitement
   - "negative": Complaints, frustration, disappointment
   - "neutral": Factual, questions without strong emotion

2. **sentiment_rich**: More nuanced emotion (pick the most dominant)
   - "frustration": User is frustrated or annoyed
   - "confusion": User is confused or uncertain
   - "delight": User is happy or impressed
   - "gratitude": User is thankful
   - "curiosity": User is curious or interested
   - "complaint": User is complaining but not necessarily frustrated
   - "neutral": No strong emotion

3. **content_type**: What kind of message is this
   - "question": Asking for help or information
   - "bug_report": Reporting a bug or issue
   - "feature_request": Requesting a new feature
   - "praise": Positive feedback or testimonial
   - "complaint": Negative feedback
   - "discussion": General discussion
   - "documentation": Documentation content
   - "announcement": News or announcements

4. **topics**: Which topics does this message relate to (list all that apply)
   - "gpu_availability": GPU access, availability, quotas
   - "pricing": Costs, billing, pricing
   - "performance": Speed, latency, throughput
   - "documentation": Docs, examples, tutorials
   - "ease_of_use": Simplicity, developer experience
   - "reliability": Uptime, stability, errors
   - "support": Customer support, response times
   - "functions": Modal functions, decorators
   - "volumes": Data storage, volumes
   - "secrets": Secret management
   - "images": Container images, dependencies
   - "scheduling": Cron jobs, scheduling
   - "web_endpoints": Web endpoints, APIs
   - "other": Other topics

Respond with ONLY a valid JSON object, no other text:
{
  "sentiment_simple": "positive|negative|neutral",
  "sentiment_rich": "frustration|confusion|delight|gratitude|curiosity|complaint|neutral",
  "content_type": "question|bug_report|feature_request|praise|complaint|discussion|documentation|announcement",
  "topics": ["topic1", "topic2"]
}"""


class SentimentAnalyzer:
    """
    Sentiment analyzer using LLM for classification.
    
    Can use either:
    - Local inference with transformers
    - External API (for testing)
    """
    
    def __init__(self, model=None, tokenizer=None):
        """
        Initialize with optional pre-loaded model.
        
        Args:
            model: Pre-loaded transformers model
            tokenizer: Pre-loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger.bind(component="sentiment")
    
    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single message.
        
        Returns dict with:
            - sentiment_simple
            - sentiment_rich
            - content_type
            - topics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Build the prompt
        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this message:\n\n{text[:2000]}"},  # Truncate long texts
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
            max_new_tokens=200,
            temperature=0.1,  # Low temperature for consistent classification
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse JSON response
        return self._parse_response(response)
    
    def analyze_batch(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        """
        Analyze sentiment of multiple messages.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    result = self.analyze(text)
                    results.append(result)
                except Exception as e:
                    self.logger.warning("Failed to analyze message", error=str(e))
                    results.append(self._default_result())
        return results
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response into structured result."""
        try:
            # Try to find JSON in response
            response = response.strip()
            
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response)
            
            # Validate and normalize
            return {
                "sentiment_simple": result.get("sentiment_simple", "neutral"),
                "sentiment_rich": result.get("sentiment_rich", "neutral"),
                "content_type": result.get("content_type", "discussion"),
                "topics": result.get("topics", []),
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.logger.warning("Failed to parse sentiment response", error=str(e), response=response[:200])
            return self._default_result()
    
    def _default_result(self) -> dict:
        """Return default result when analysis fails."""
        return {
            "sentiment_simple": "neutral",
            "sentiment_rich": "neutral",
            "content_type": "discussion",
            "topics": [],
        }


def load_model_for_sentiment(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """
    Load model and tokenizer for sentiment analysis.
    
    This is a helper function to load the model in a Modal function.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info("Loading model for sentiment analysis", model=model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    return model, tokenizer
