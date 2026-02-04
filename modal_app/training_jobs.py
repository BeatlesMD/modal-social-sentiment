"""Training and model packaging jobs."""

import modal

from src.config import BASE_MODEL, DUCKDB_PATH, FINE_TUNED_DIR, TRAINING_DIR

from .common import data_volume, hf_secret, models_volume, training_image

app = modal.App("modal-social-sentiment-training")


@app.function(
    image=training_image,
    gpu="A100",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    timeout=14400,
)
def prepare_training_data():
    """Prepare training dataset from ingested data."""
    import structlog
    from src.storage.duckdb_store import DuckDBStore
    from src.training.dataset import TrainingDatasetBuilder

    logger = structlog.get_logger()
    logger.info("Preparing training data")

    builder = TrainingDatasetBuilder(output_dir=TRAINING_DIR)

    with DuckDBStore(DUCKDB_PATH) as db:
        result = db.conn.execute(
            """
            SELECT * FROM messages WHERE content IS NOT NULL AND LENGTH(content) > 100
            """
        ).fetchall()
        columns = [d[0] for d in db.conn.description]
        messages = [dict(zip(columns, r)) for r in result]

    doc_examples = builder.build_from_docs(messages)
    conv_examples = builder.build_from_conversations(messages)
    all_examples = doc_examples + conv_examples

    if not all_examples:
        return {"status": "error", "reason": "no examples"}

    train_path, _val_path = builder.save_dataset(all_examples)
    data_volume.commit()

    return {
        "status": "success",
        "total": len(all_examples),
        "train_path": str(train_path),
    }


@app.function(
    image=training_image,
    gpu="A100",
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[hf_secret],
    timeout=21600,
)
def run_finetuning(train_path: str | None = None, epochs: int = 3):
    """Run QLoRA fine-tuning."""
    from src.training.finetune import QLoRATrainer

    train_path = train_path or f"{TRAINING_DIR}/training_data.jsonl"
    val_path = train_path.replace(".jsonl", "_val.jsonl")

    trainer = QLoRATrainer(
        base_model=BASE_MODEL,
        output_dir=FINE_TUNED_DIR,
        num_epochs=epochs,
    )
    model_path = trainer.train(train_path, val_path)

    models_volume.commit()
    return {"status": "success", "model_path": model_path}


@app.function(
    image=training_image,
    gpu="A10G",
    volumes={"/models": models_volume},
    secrets=[hf_secret],
    timeout=10800,
)
def merge_finetuned_weights(
    adapter_path: str | None = None,
    output_path: str | None = None,
):
    """Merge LoRA adapter into a standalone model for fast inference."""
    from src.training.finetune import merge_lora_weights

    adapter_path = adapter_path or f"{FINE_TUNED_DIR}/final"
    output_path = output_path or f"{FINE_TUNED_DIR}/merged"

    merged_path = merge_lora_weights(
        base_model=BASE_MODEL,
        adapter_path=adapter_path,
        output_path=output_path,
    )
    models_volume.commit()

    return {"status": "success", "model_path": merged_path}

