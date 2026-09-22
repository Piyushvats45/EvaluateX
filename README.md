# LLM Evaluation & Prompt Optimization Framework

A modular framework for benchmarking prompts and comparing LLM configurations —
real local model inference, structured evaluation metrics, a REST API, an
interactive demo UI, automated tests, and CI. Every piece runs at **$0 cost**:
open-weight models on your own machine, SQLite for storage, free/open-source
metric libraries, and free-tier hosting for deployment.

## Highlights

- **Real local LLM inference** — DistilGPT-2 / GPT-2 (or any Hugging Face causal-LM) run entirely on your own CPU, no API key, no per-token billing
- **Structured evaluation metrics** — ROUGE-L, BERTScore-style semantic similarity, n-gram diversity, combined into a composite score
- **Modular experiment pipeline** — swap models, prompts, or metrics independently; every prompt × every model config is logged automatically
- **SQLite logging** — every prompt, response, and metric is persisted and queryable after the fact
- **Metric ↔ human agreement analysis** — Pearson/Spearman correlation between automated scores and human ratings, to validate whether cheap automated scoring is a trustworthy proxy for human review
- **REST API** (`api.py`, FastAPI) — deployable behind a URL on any free-tier host
- **Interactive demo UI** (`app.py`, Streamlit) — deployable for free, so you can link a *live, clickable demo* on your resume
- **Automated tests + CI** — `pytest` suite covering metrics, models, and the full pipeline; GitHub Actions runs it on every push

## Architecture

```
EvaluateX/
├── app.py                      # Streamlit interactive demo (deploy this for a live link)
├── api.py                      # FastAPI REST service (deploy this behind a URL)
├── run_experiment.py           # CLI entry point for batch experiments
├── human_agreement_demo.py     # demo: metric-vs-human-rating agreement analysis
├── Dockerfile                  # containerized deployment for api.py
├── requirements.txt            # full deps (real models + API + UI + tests)
├── requirements-minimal.txt    # fast install, mock backend only (CI)
├── config/
│   ├── models.yaml             # DEFAULT: real local HuggingFace models
│   └── models.mock.yaml        # offline/CI config, no downloads needed
├── data/
│   └── prompts.json            # structured NLP-task prompt set
├── src/
│   ├── models.py                # pluggable model runners (huggingface / mock)
│   ├── metrics.py                # ROUGE-L, semantic similarity, diversity, composite score
│   ├── database.py               # SQLite logging layer
│   ├── pipeline.py               # orchestrates prompt × model experiment runs
│   └── report.py                 # CSV + self-contained HTML report with charts
├── tests/
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_pipeline.py
├── .github/workflows/ci.yml    # automated test suite on every push
└── outputs/                    # generated DB, CSV, HTML report (gitignored)
```

