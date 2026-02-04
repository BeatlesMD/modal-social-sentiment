"""Modal Social Sentiment - App Orchestrator.

Deploy: modal deploy app.py
Run:    modal run app.py
"""

import modal

from modal_app.common import APP_NAME
from modal_app.ingestion_jobs import (
    app as ingestion_app,
    ingest_docs,
    ingest_github,
    ingest_hackernews,
)
from modal_app.inference_service import app as inference_app, Assistant
from modal_app.processing_jobs import (
    app as processing_app,
    analyze_sentiment,
    check_db_status,
    generate_embeddings,
    reset_embeddings,
)
from modal_app.test_jobs import (
    app as test_app,
    test_db_connection,
    test_embeddings,
    test_ingestion_config,
    test_model_loading,
)
from modal_app.training_jobs import (
    app as training_app,
    merge_finetuned_weights,
    prepare_training_data,
    run_finetuning,
)
from modal_app.web_dashboard import app as web_app

app = modal.App(APP_NAME)
app.include(ingestion_app)
app.include(processing_app)
app.include(training_app)
app.include(inference_app)
app.include(web_app)
app.include(test_app)


@app.local_entrypoint()
def main(task: str = "ingest"):
    """Run tasks on Modal.

    Usage:
        modal run app.py                    # Run all ingestion
        modal run app.py --task test        # Run tests on Modal
        modal run app.py --task ingest      # Run ingestion
        modal run app.py --task process     # Run processing
        modal run app.py --task train       # Prepare + run fine-tuning
        modal run app.py --task merge       # Merge LoRA weights
        modal run app.py --task ask         # Test assistant
        modal run app.py --task status      # Inspect DB counts
    """
    if task == "test":
        print("Running Modal-native tests...\n")

        calls = [
            ("DB Connection", test_db_connection.spawn()),
            ("Ingestion Config", test_ingestion_config.spawn()),
            ("Embeddings", test_embeddings.spawn()),
            ("Model Loading", test_model_loading.spawn()),
        ]

        failed = False
        print("=" * 50)
        print("TEST RESULTS")
        print("=" * 50)

        for name, call in calls:
            result = call.get()
            status = result.get("status", "unknown")
            icon = "PASS" if status == "pass" else "FAIL"
            print(f"\n[{icon}] {name}:")
            for key, value in result.items():
                if key != "status":
                    print(f"  {key}: {value}")
            if status != "pass":
                failed = True

        print("\n" + "=" * 50)
        if failed:
            print("Some tests failed")
            raise SystemExit(1)
        print("All tests passed")

    elif task == "ingest":
        print("Running ingestion on Modal...")
        calls = [
            ("Docs", ingest_docs.spawn()),
            ("GitHub", ingest_github.spawn()),
            ("HackerNews", ingest_hackernews.spawn()),
        ]
        for name, call in calls:
            print(f"  {name}: {call.get()}")

    elif task == "process":
        print("Running processing on Modal...")
        calls = [
            ("Embeddings", generate_embeddings.spawn()),
            ("Sentiment", analyze_sentiment.spawn()),
        ]
        for name, call in calls:
            print(f"  {name}: {call.get()}")

    elif task == "train":
        print("Preparing and fine-tuning on Modal...")
        prep = prepare_training_data.remote()
        print(f"  Prep: {prep}")
        if prep.get("status") == "success":
            print(f"  Training: {run_finetuning.remote(prep['train_path'])}")

    elif task == "merge":
        print("Merging fine-tuned adapter into standalone model...")
        print(f"  Merge: {merge_finetuned_weights.remote()}")

    elif task == "ask":
        print("Testing assistant...")
        assistant = Assistant()
        print(f"  Health: {assistant.health.remote()}")
        result = assistant.ask.remote("How do I use Modal volumes?")
        print(f"  Answer: {result['answer'][:300]}...")

    elif task == "status":
        print("Checking DB status...")
        print(f"  {check_db_status.remote()}")

    elif task == "reset-embeddings":
        print("Resetting embeddings...")
        print(f"  {reset_embeddings.remote()}")

    else:
        print(f"Unknown task: {task}")
        print("Available: test, ingest, process, train, merge, ask, status, reset-embeddings")
