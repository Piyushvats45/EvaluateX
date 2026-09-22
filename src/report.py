"""
report.py
---------
Turns the raw logged experiment data into:
  1. A CSV summary (per-model aggregate metrics) for spreadsheets/tracking.
  2. A single self-contained HTML report (charts embedded as base64 PNGs,
     so it opens standalone with no server, no internet, $0 cost) that
     compares models side-by-side and — if human ratings were supplied —
     reports metric-human agreement.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics as metrics_mod

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; background: #fafafa; color: #1a1a1a; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.92rem; }}
  th {{ background: #f2f2f5; font-weight: 600; }}
  tr:hover {{ background: #f7f7fb; }}
  .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 28px; }}
  img {{ max-width: 100%; border-radius: 8px; }}
  .badge {{ display:inline-block; padding: 2px 10px; border-radius: 12px; background:#eef2ff; color:#3b4dd1; font-size:0.8rem; margin-left:8px;}}
  code {{ background:#f2f2f5; padding:2px 6px; border-radius:4px; font-size:0.85rem; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">Experiment ID {experiment_id} &middot; {n_prompts} prompts &middot; {n_models} model configs &middot; {n_responses} responses scored</div>

  <div class="card">
    <h2>Per-model summary</h2>
    {summary_table}
  </div>

  <div class="card">
    <h2>Composite score by model</h2>
    <img src="data:image/png;base64,{composite_chart}" />
  </div>

  <div class="card">
    <h2>Latency vs. quality trade-off</h2>
    <img src="data:image/png;base64,{latency_chart}" />
  </div>

  {agreement_block}

  <div class="card">
    <h2>Sample responses</h2>
    {sample_table}
  </div>
</body>
</html>
"""


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _composite_chart(df: pd.DataFrame) -> str:
    agg = df.groupby("model_name")["composite"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    agg.plot(kind="bar", ax=ax, color="#3b4dd1")
    ax.set_ylabel("Mean composite score")
    ax.set_xlabel("")
    ax.set_title("Average composite score per model")
    plt.xticks(rotation=20, ha="right")
    return _fig_to_base64(fig)


def _latency_chart(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, group in df.groupby("model_name"):
        ax.scatter(group["latency_sec"], group["composite"], label=name, alpha=0.7)
    ax.set_xlabel("Latency (sec)")
    ax.set_ylabel("Composite score")
    ax.set_title("Latency vs. composite score")
    ax.legend(fontsize=8)
    return _fig_to_base64(fig)


def generate_report(df: pd.DataFrame, experiment_id: int, out_path: str, title: str = "LLM Evaluation Report") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("model_name")
        .agg(
            avg_rouge_l=("rouge_l", "mean"),
            avg_semantic_sim=("semantic_sim", "mean"),
            avg_distinct_2=("distinct_2", "mean"),
            avg_composite=("composite", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            n=("response_id", "count"),
        )
        .round(4)
        .sort_values("avg_composite", ascending=False)
    )

    # CSV export
    csv_path = str(Path(out_path).with_suffix(".csv"))
    summary.to_csv(csv_path)

    agreement_block = ""
    if df["human_score"].notna().any():
        agreement = metrics_mod.metric_human_agreement(
            df["composite"].fillna(0).tolist(), df["human_score"].fillna(0).tolist()
        )
        agreement_block = f"""
        <div class="card">
          <h2>Metric &harr; human agreement</h2>
          <p>Pearson correlation: <code>{agreement['pearson']}</code>
             &nbsp; Spearman rank correlation: <code>{agreement['spearman']}</code>
             &nbsp; <span class="badge">n = {agreement['n']}</span></p>
          <p style="color:#666; font-size:0.85rem;">Measures how well the automated composite score
          tracks human preference ratings — the same reliability check used to validate
          whether cheap automated scoring can substitute for human review at scale.</p>
        </div>
        """

    sample = df[["task", "model_name", "prompt_text", "response_text", "composite"]].head(8)

    html = HTML_TEMPLATE.format(
        title=title,
        experiment_id=experiment_id,
        n_prompts=df["prompt_id"].nunique(),
        n_models=df["model_name"].nunique(),
        n_responses=len(df),
        summary_table=summary.to_html(classes="", border=0),
        composite_chart=_composite_chart(df),
        latency_chart=_latency_chart(df),
        agreement_block=agreement_block,
        sample_table=sample.to_html(classes="", border=0, escape=True, index=False),
    )

    Path(out_path).write_text(html)
    return csv_path
