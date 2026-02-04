# Modal Social Sentiment

Social listening + AI support assistant for Modal, **built entirely on Modal**.

## Overview

This project combines two workflows in one Modal-native app:

1. **Voice-of-customer analytics** for product and support signal  
   - Ingests community conversations (currently GitHub + HackerNews, extensible to Reddit/X/Slack/support tools)
   - Runs sentiment, topic, and content-type classification
   - Powers a dashboard optimized for product insights (bugs, unanswered questions, pain points, high-impact mentions)

2. **Knowledge retrieval for assistant quality**  
   - Ingests docs/examples/blog content for grounding and citations
   - Uses embeddings + LanceDB retrieval for RAG
   - Keeps docs/blog out of core sentiment KPIs to avoid neutral-noise skew

At runtime, the app provides:
- Scheduled ingestion + processing jobs
- A GPU-backed assistant endpoint (`POST /ask`)
- A Streamlit dashboard + explorer + admin UI
- Training utilities for optional fine-tuning on collected data

### Data Split (Important)

- **Voice sources**: used for sentiment analytics and product insights  
- **Knowledge sources** (`docs`, `blog`): used for retrieval/context, excluded from primary KPI views by default

## Quick Start

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate
modal setup

# 3. Create secrets
modal secret create social-api-keys GITHUB_TOKEN="" REDDIT_CLIENT_ID="" REDDIT_CLIENT_SECRET=""

# 4. Run ingestion
modal run app.py

# 5. Deploy (scheduled jobs + web endpoints)
modal deploy app.py
```

## Commands

```bash
# Run Modal-native tests (validates DB, GPU, embeddings, config)
modal run app.py --task test

# Ingest data from GitHub, HN, Modal docs
modal run app.py --task ingest

# Process: generate embeddings + sentiment analysis
modal run app.py --task process

# Fine-tune the model (requires training data)
modal run app.py --task train

# Test the assistant interactively
modal run app.py --task ask

# Deploy everything (scheduled jobs + web endpoints)
modal deploy app.py

# Stop the deployed app
modal app stop modal-social-sentiment
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
│  Modal Docs │ GitHub Issues │ HackerNews │ Reddit │ Twitter │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              INGESTION (Scheduled Cron Jobs)                │
│  ingest_docs() │ ingest_github() │ ingest_hackernews()      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODAL VOLUMES                            │
│  /data (DuckDB + LanceDB)  │  /models (Fine-tuned weights)  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                PROCESSING (Scheduled)                       │
│  generate_embeddings()  │  analyze_sentiment() (GPU)        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                INFERENCE (On-demand, A10G)                  │
│  Assistant.ask() │ POST /ask endpoint                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  WEB DASHBOARD (Streamlit)                  │
│  dashboard() → https://...modal.run                         │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
modal-social-sentiment/
├── app.py              # Main Modal app (all functions)
├── src/
│   ├── ingestion/      # Data collectors
│   ├── processing/     # Sentiment + embeddings
│   ├── training/       # Fine-tuning pipeline
│   ├── inference/      # RAG assistant
│   ├── storage/        # DuckDB + LanceDB
│   └── app/            # Streamlit UI
└── README.md
```

## Costs

| Task | GPU | Cost |
|------|-----|------|
| Ingestion | - | ~$0.01/run |
| Embeddings | - | ~$0.02/run |
| Sentiment | A10G | ~$0.20/run |
| Fine-tuning | A100 | ~$15 total |
| Inference | A10G | ~$1.10/hr |

## API

After `modal deploy app.py`:

```bash
# Ask a question
curl -X POST https://<your-url>/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I use Modal volumes?"}'
```

## Modal Best Practices

This project follows [Modal best practices](https://modal.com/docs/guide/project-structure):

- **`uv_pip_install`** - Fast, reliable package installation
- **Pinned dependencies** - Reproducible builds
- **CUDA base images** - `nvidia/cuda:12.4.0-devel-ubuntu22.04` for GPU workloads
- **`HF_HOME`/`TORCH_HOME`** - Model caching to volumes
- **`@modal.concurrent`** - Multiple requests per GPU container
- **`add_local_python_source("src")`** - Proper local code packaging
- **Modal-native tests** - Validate infra before deploying

## License

MIT
