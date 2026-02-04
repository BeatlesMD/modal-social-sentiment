"""Inference class and HTTP endpoint."""

import modal

from src.config import BASE_MODEL, EMBEDDING_MODEL, FINE_TUNED_DIR, LANCEDB_PATH

from .common import endpoint_image, hf_secret, inference_image, models_volume, data_volume

app = modal.App("modal-social-sentiment-inference")


@app.cls(
    image=inference_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
class Assistant:
    """RAG-powered Modal support assistant."""

    @modal.enter()
    def load(self):
        from pathlib import Path
        import structlog
        from src.inference.assistant import load_assistant

        logger = structlog.get_logger()
        candidate_paths = [
            f"{FINE_TUNED_DIR}/final",
            f"{FINE_TUNED_DIR}/merged",
        ]
        model_path = next((p for p in candidate_paths if Path(p).exists()), None)

        logger.info("Loading assistant", model=model_path or BASE_MODEL)
        self.assistant = load_assistant(
            model_path=model_path,
            base_model=BASE_MODEL,
            embedding_model=EMBEDDING_MODEL,
            vector_store_path=LANCEDB_PATH,
            use_quantization=True,
        )

    @modal.method()
    def ask(self, question: str, use_rag: bool = True) -> dict:
        return self.assistant.answer(question=question, use_rag=use_rag)

    @modal.method()
    def health(self) -> dict:
        return {"status": "healthy", "model": self.assistant.model is not None}


@app.function(image=endpoint_image)
@modal.fastapi_endpoint(method="POST")
def ask(request: dict) -> dict:
    """Web endpoint: POST /ask with {"question": "..."}."""
    question = request.get("question", "")
    if not question:
        return {"error": "No question"}
    return Assistant().ask.remote(question, request.get("use_rag", True))

