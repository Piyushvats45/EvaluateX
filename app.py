from __future__ import annotations

import time
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import metrics as metrics_mod
from src.models import build_runner

# PAGE CONFIG
st.set_page_config(
    page_title="EvaluateX",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hero */
    .hero {
        padding: 3rem 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 100%
        );
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.15rem;
        color: #d1d5db;
        max-width: 800px;
    }

    /* Section heading */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Feature cards */
    .feature-card {
        padding: 1.4rem;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        background: white;
        min-height: 180px;
    }

    .feature-card h3 {
        margin-bottom: 0.5rem;
    }

    .feature-card p {
        color: #6b7280;
        line-height: 1.6;
    }

    /* Tech badges */
    .tech {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        margin: 0.25rem;
        border-radius: 20px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }

    /* Metric cards */
    .metric-card {
        text-align: center;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: white;
    }

    .metric-card .number {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-card .label {
        color: #6b7280;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: #15171c;
        border-right: 1px solid #292d35;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0.2rem 1rem 0.2rem;
    }

    .sidebar-logo {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #ffffff;
        color: #111827;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 800;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .sidebar-subtitle {
        font-size: 0.75rem;
        color: #8b93a1;
        margin-top: 2px;
    }

    .sidebar-section {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: #8b93a1;
        margin: 1rem 0 0.8rem 0;
    }

    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff;
    }

    .page-header {
        margin-bottom: 1.8rem;
    }

    .page-header h1 {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .page-header p {
        color: #9ca3af;
        font-size: 1rem;
    }

    .evaluation-card {
        padding: 1.5rem;
        border: 1px solid #2b3038;
        border-radius: 14px;
        background: #171a20;
        margin-bottom: 1.5rem;
    }

    div.stButton > button[kind="primary"] {
        height: 3rem;
        border-radius: 9px;
        font-size: 1rem;
        font-weight: 650;
        border: none;
    }

    .result-card {
        background: #171a20;
        border: 1px solid #2b3038;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .result-value {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .result-label {
        color: #8b93a1;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .workflow-note {
        border-left: 3px solid #64748b;
        background: #171a20;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# MODEL CONFIGURATION
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-logo">E</div>
        <div>
            <div class="sidebar-title">EvaluateX</div>
            <div class="sidebar-subtitle">LLM Evaluation</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    '<div class="sidebar-section">Experiment Configuration</div>',
    unsafe_allow_html=True,
)

backend = st.sidebar.selectbox(
    "Model Backend",
    options=["huggingface", "mock"],
    index=0,
)

if backend == "huggingface":

    model_name = st.sidebar.text_input(
        "Hugging Face Model",
        value="distilgpt2",
    )

    max_new_tokens = st.sidebar.slider(
        "Max New Tokens",
        10,
        200,
        60,
    )

else:

    model_name = st.sidebar.selectbox(
        "Mock Style",
        options=[
            "mock-concise",
            "mock-verbose",
            "mock-creative",
        ],
    )

    max_new_tokens = 60


n_candidates = st.sidebar.slider(
    "Candidate Responses",
    1,
    5,
    1,
)


@st.cache_resource(show_spinner="Loading model...")
def get_runner(backend: str, model_name: str):
    return build_runner(backend, model_name)


# NAVIGATION
home_tab, evaluate_tab, benchmark_tab, report_tab = st.tabs(
    [
        "Home",
        "Evaluate",
        "Benchmark",
        "Reports",
    ]
)

# HOME
with home_tab:

    st.markdown(
        """
        <div class="hero">

        <h1>EvaluateX</h1>

        <p>
        An evaluation and prompt optimization framework for systematically
        benchmarking LLM responses using automated quality, similarity,
        diversity, and performance metrics.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-title">
        What does this project do?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        """
        EvaluateX helps developers evaluate how different prompts and
        model configurations perform on the same NLP tasks. Instead of
        judging a generated response only by looking at it manually, the
        framework generates candidate responses and evaluates them using
        multiple automated metrics.
        """
    )

    st.write(
        """
        The system supports repeatable experiments, model comparison,
        metric-based scoring, and experiment tracking so that prompt
        changes can be evaluated using measurable results.
        """
    )
    # WHAT IT DOES

    st.markdown(
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
            <h3>1️⃣ Generate</h3>
            <p>
            Provide a prompt and generate one or more candidate responses
            using the selected LLM backend.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
            <h3>2️⃣ Evaluate</h3>
            <p>
            Measure response quality using ROUGE-L, semantic similarity,
            and diversity metrics.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
            <h3>3️⃣ Compare</h3>
            <p>
            Compare candidate responses and model configurations using
            a composite evaluation score.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="feature-card">
            <h3>4️⃣ Optimize</h3>
            <p>
            Use experiment results to identify better-performing prompt
            and model configurations.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TECHNICAL ARCHITECTURE
    st.markdown(
        '<div class="section-title">Technical Architecture</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "The evaluation pipeline follows a modular generate → score → "
        "compare → track workflow."
    )

    architecture = """
    digraph {

        graph [
            rankdir=LR,
            bgcolor="transparent",
            nodesep=0.5,
            ranksep=0.8
        ]

        node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=12,
            margin="0.2,0.12"
        ]

        edge [
            color="#64748b",
            arrowsize=0.8
        ]

        prompt [
            label="Prompt / Task",
            fillcolor="#e0f2fe"
        ]

        variants [
            label="Prompt Variants",
            fillcolor="#e0f2fe"
        ]

        model [
            label="LLM Backend\\nHuggingFace / Mock",
            fillcolor="#ede9fe"
        ]

        responses [
            label="Candidate Responses",
            fillcolor="#fef3c7"
        ]

        metrics [
            label="Evaluation Metrics\\nROUGE-L\\nSemantic Similarity\\nDistinct-2",
            fillcolor="#dcfce7"
        ]

        score [
            label="Composite Score",
            fillcolor="#dcfce7"
        ]

        tracking [
            label="Experiment Tracking\\nMLflow / SQLite",
            fillcolor="#fce7f3"
        ]

        prompt -> variants
        variants -> model
        model -> responses
        responses -> metrics
        metrics -> score
        score -> tracking
    }
    """

    st.graphviz_chart(
        architecture,
        use_container_width=True,
    )

    # METRICS

    st.markdown(
        '<div class="section-title">Evaluation Metrics</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            <div class="metric-card">
            <div class="number">ROUGE-L</div>
            <div class="label">
            Measures lexical overlap with the reference answer
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="metric-card">
            <div class="number">Semantic Similarity</div>
            <div class="label">
            Measures semantic closeness between response and reference
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            """
            <div class="metric-card">
            <div class="number">Distinct-2</div>
            <div class="label">
            Measures response diversity using unique bigrams
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TECH STACK

    st.markdown(
        '<div class="section-title">Technology Stack</div>',
        unsafe_allow_html=True,
    )

    technologies = [
        "Python",
        "Streamlit",
        "Hugging Face",
        "Pandas",
        "ROUGE-L",
        "BERTScore / Semantic Similarity",
        "MLflow",
        "SQLite",
    ]

    st.markdown(
        "".join(
            f'<span class="tech">{tech}</span>'
            for tech in technologies
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.info(
        "💡 Start with the Evaluate tab to test an individual prompt, "
        "or use Benchmark to evaluate the complete prompt dataset."
    )

# EVALUATE

with evaluate_tab:

    st.markdown(
        """
        <div class="page-header">

            Prompt Evaluation
          
            Generate candidate responses and measure their quality using
            automated evaluation metrics.
            

        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="workflow-note">

        <strong>Evaluation workflow</strong><br>

        Prompt → Model Generation → Candidate Responses
        → Metric Evaluation → Composite Score

        </div>
        """,
        unsafe_allow_html=True,
    )


    col1, col2 = st.columns(2)

    with col1:

        prompt = st.text_area(
            "Prompt",
            value="Question: What is the capital of France?",
            height=140,
            placeholder="Enter the prompt you want to evaluate...",
        )

    with col2:

        reference = st.text_area(
            "Reference Answer",
            value="Answer: Paris",
            height=140,
            placeholder="Optional reference answer...",
            help="Providing a reference enables reference-based metrics.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=True,
    ):

        try:
            runner = get_runner(
                backend,
                model_name,
            )

        except ImportError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Generating candidate responses..."):

            start = time.time()

            generations = runner.generate(
                prompt,
                n=n_candidates,
                max_new_tokens=max_new_tokens,
            )

            elapsed = time.time() - start

        st.success(
            f"Generated {len(generations)} candidate(s) "
            f"in {elapsed:.2f}s"
        )

        rows = []

        for i, gen in enumerate(generations):

            result = metrics_mod.score_response(
                reference or None,
                gen.text,
            )

            rows.append(
                {
                    "Candidate": i + 1,
                    "Response": gen.text,
                    "ROUGE-L": result.rouge_l,
                    "Semantic Similarity": result.semantic_sim,
                    "Distinct-2": result.distinct_2,
                    "Composite": result.composite,
                    "Latency (s)": round(
                        gen.latency_sec,
                        4,
                    ),
                }
            )

        df = pd.DataFrame(rows)

        st.subheader("Evaluation Results")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        best = df.sort_values(
            "Composite",
            ascending=False,
        ).iloc[0]

        st.success(
            f"Best candidate: #{int(best['Candidate'])} "
            f"with composite score "
            f"{best['Composite']:.3f}"
        )

# BENCHMARK

with benchmark_tab:

    st.header("Full Benchmark")

    st.write(
        """
        Run the complete prompt dataset through the selected model
        configuration and compare aggregate performance across tasks.
        """
    )

    if st.button(
        "Run Full Benchmark",
        type="primary",
        use_container_width=True,
    ):

        prompts_path = Path("data/prompts.json")

        if not prompts_path.exists():

            st.error(
                "data/prompts.json was not found."
            )

            st.stop()

        prompts = json.loads(
            prompts_path.read_text()
        )

        try:

            runner = get_runner(
                backend,
                model_name,
            )

        except ImportError as e:

            st.error(str(e))
            st.stop()

        progress = st.progress(0.0)

        rows = []

        for idx, item in enumerate(prompts):

            gens = runner.generate(
                item["prompt"],
                n=1,
                max_new_tokens=max_new_tokens,
            )

            gen = gens[0]

            result = metrics_mod.score_response(
                item.get("reference"),
                gen.text,
            )

            rows.append(
                {
                    "Task": item.get(
                        "task",
                        "general",
                    ),
                    "Prompt": item["prompt"][:70],
                    "Response": gen.text[:100],
                    "ROUGE-L": result.rouge_l,
                    "Semantic Similarity": result.semantic_sim,
                    "Composite": result.composite,
                    "Latency (s)": round(
                        gen.latency_sec,
                        4,
                    ),
                }
            )

            progress.progress(
                (idx + 1) / len(prompts)
            )

        df = pd.DataFrame(rows)

        st.success(
            f"Benchmark completed — {len(df)} prompts evaluated."
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Average Composite Score by Task")

        chart_df = (
            df.groupby("Task")["Composite"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(chart_df)


# REPORT

with report_tab:

    st.header("Experiment Reports")

    st.caption(
        "Detailed experiment results, model comparison, "
        "evaluation metrics, and performance analysis."
    )

    report_path = Path(__file__).resolve().parent / "outputs" / "report.html"

    if report_path.exists():

        report_html = report_path.read_text(
            encoding="utf-8"
        )

        components.html(
            report_html,
            height=1200,
            scrolling=True,
        )

    else:

        st.error(
            f"Report file not found: {report_path}"
        )