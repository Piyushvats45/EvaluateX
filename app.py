"""
app.py
------
Interactive Streamlit demo for the LLM Evaluation & Prompt Optimization
Framework. This is the piece you deploy for free (Streamlit Community
Cloud or a Hugging Face Space) and link to on your resume / LinkedIn so an
interviewer can click it and try it themselves, live, with no setup.

Run locally:
    streamlit run app.py

Deploy for free:
    - Streamlit Community Cloud: push this repo to GitHub, go to
      share.streamlit.io, point it at app.py. Free tier, $0 cost.
    - Hugging Face Spaces: create a Space with SDK "Streamlit", push this
      repo. Free CPU tier, $0 cost.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src import metrics as metrics_mod
from src.models import build_runner

st.set_page_config(page_title="LLM Evaluation & Prompt Optimization", layout="wide")

st.title("🧪 LLM Evaluation & Prompt Optimization Framework")
st.caption(
    "Benchmark prompts across model configurations using ROUGE-L, semantic "
    "similarity, and diversity metrics — running entirely locally, at $0 cost."
)

# ---------------------------------------------------------------------------
# Sidebar: model configuration
# ---------------------------------------------------------------------------
st.sidebar.header("Model configuration")

backend = st.sidebar.selectbox(
    "Backend",
    options=["huggingface", "mock"],
    index=0,
    help=(
        "huggingface = real local open-weight model (downloads once, then runs "
        "free forever). mock = offline, dependency-free, for fast demos/tests."
    ),
)

if backend == "huggingface":
    model_name = st.sidebar.text_input(
        "Hugging Face model id", value="distilgpt2",
        help="Any causal-LM model id on the HF Hub, e.g. distilgpt2, gpt2, gpt2-medium.",
    )
    max_new_tokens = st.sidebar.slider("Max new tokens", 10, 200, 60)
else:
    model_name = st.sidebar.selectbox(
        "Mock style", options=["mock-concise", "mock-verbose", "mock-creative"]
    )
    max_new_tokens = 60

n_candidates = st.sidebar.slider("Candidates to generate", 1, 5, 1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 First run with a Hugging Face backend downloads the model once "
    "(e.g. ~350MB for distilgpt2). Every run after that is instant and free — "
    "no API key, no per-call billing."
)


@st.cache_resource(show_spinner="Loading model (first run only)...")
def get_runner(backend: str, model_name: str):
    return build_runner(backend, model_name)


# ---------------------------------------------------------------------------
# Main panel: single prompt evaluation
# ---------------------------------------------------------------------------
st.subheader("1. Try a single prompt")

col1, col2 = st.columns(2)
with col1:
    prompt = st.text_area(
        "Prompt",
        value="Question: What is the capital of France? Answer:",
        height=100,
    )
with col2:
    reference = st.text_area(
        "Reference / expected answer (optional — enables ROUGE-L & semantic similarity)",
        value="Paris",
        height=100,
    )

if st.button("▶ Run evaluation", type="primary"):
    try:
        runner = get_runner(backend, model_name)
    except ImportError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Generating..."):
        start = time.time()
        generations = runner.generate(prompt, n=n_candidates, max_new_tokens=max_new_tokens)
        elapsed = time.time() - start

    st.success(f"Generated {len(generations)} candidate(s) in {elapsed:.2f}s")

    rows = []
    for i, gen in enumerate(generations):
        result = metrics_mod.score_response(reference or None, gen.text)
        rows.append(
            {
                "candidate": i + 1,
                "response": gen.text,
                "rouge_l": result.rouge_l,
                "semantic_sim": result.semantic_sim,
                "semantic_backend": result.semantic_backend,
                "distinct_2": result.distinct_2,
                "composite": result.composite,
                "latency_sec": round(gen.latency_sec, 4),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    best = df.sort_values("composite", ascending=False).iloc[0]
    st.info(f"**Best candidate:** #{int(best['candidate'])} — composite score {best['composite']:.3f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Batch comparison across the sample prompt set
# ---------------------------------------------------------------------------
st.subheader("2. Run the full sample benchmark")
st.caption(
    "Runs every prompt in data/prompts.json through the selected model and "
    "shows aggregate metrics — the same thing run_experiment.py does from the CLI."
)

if st.button("▶ Run full benchmark"):
    import json
    from pathlib import Path

    prompts = json.loads(Path("data/prompts.json").read_text())

    try:
        runner = get_runner(backend, model_name)
    except ImportError as e:
        st.error(str(e))
        st.stop()

    progress = st.progress(0.0)
    rows = []
    for idx, item in enumerate(prompts):
        gens = runner.generate(item["prompt"], n=1, max_new_tokens=max_new_tokens)
        gen = gens[0]
        result = metrics_mod.score_response(item.get("reference"), gen.text)
        rows.append(
            {
                "task": item.get("task", "general"),
                "prompt": item["prompt"][:60] + "...",
                "response": gen.text[:80] + ("..." if len(gen.text) > 80 else ""),
                "rouge_l": result.rouge_l,
                "semantic_sim": result.semantic_sim,
                "composite": result.composite,
                "latency_sec": round(gen.latency_sec, 4),
            }
        )
        progress.progress((idx + 1) / len(prompts))

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.subheader("Aggregate by task")
    st.bar_chart(df.groupby("task")["composite"].mean())

st.markdown("---")
st.caption(
    "Built with a modular pipeline (src/pipeline.py), pluggable model backends "
    "(src/models.py), structured metrics (src/metrics.py), and SQLite logging "
    "(src/database.py). See README.md for the CLI, API, and deployment options."
)
