from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import metrics as metrics_mod
from src.models import build_runner
from src.pipeline import ExperimentPipeline, load_model_configs, load_prompts

app = FastAPI(
    title="LLM Evaluation & Prompt Optimization API",
    description="Benchmark prompts across LLM configurations with structured metrics — free, local inference.",
    version="1.0.0",
)

# Runners are built lazily and cached per (backend, model_name) so a model is
# only loaded once per process, not once per request.
_runner_cache: dict[tuple[str, str], object] = {}


def get_runner(backend: str, model_name: str):
    key = (backend, model_name)
    if key not in _runner_cache:
        _runner_cache[key] = build_runner(backend, model_name)
    return _runner_cache[key]


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the model")
    reference: Optional[str] = Field(None, description="Optional gold answer for ROUGE-L / semantic similarity")
    backend: str = Field("huggingface", description="'huggingface' or 'mock'")
    model_name: str = Field("distilgpt2", description="HF model id (if backend=huggingface) or mock style name")
    n_candidates: int = Field(1, ge=1, le=5)
    max_new_tokens: int = Field(60, ge=1, le=512)


class CandidateResult(BaseModel):
    response: str
    rouge_l: float
    semantic_sim: float
    semantic_backend: str
    distinct_2: float
    composite: float
    latency_sec: float


class EvaluateResponse(BaseModel):
    prompt: str
    backend: str
    model_name: str
    candidates: list[CandidateResult]
    best_candidate_index: int


class RunExperimentRequest(BaseModel):
    experiment_name: str = "api-triggered-run"
    prompts_path: str = "data/prompts.json"
    models_path: str = "config/models.yaml"
    candidates_per_prompt: int = Field(1, ge=1, le=5)


class RunExperimentResponse(BaseModel):
    experiment_id: int
    n_prompts: int
    n_models: int
    n_responses: int
    summary: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    try:
        runner = get_runner(req.backend, req.model_name)
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    generations = runner.generate(req.prompt, n=req.n_candidates, max_new_tokens=req.max_new_tokens)

    candidates = []
    for gen in generations:
        result = metrics_mod.score_response(req.reference, gen.text)
        candidates.append(
            CandidateResult(
                response=gen.text,
                rouge_l=result.rouge_l,
                semantic_sim=result.semantic_sim,
                semantic_backend=result.semantic_backend,
                distinct_2=result.distinct_2,
                composite=result.composite,
                latency_sec=gen.latency_sec,
            )
        )

    best_idx = max(range(len(candidates)), key=lambda i: candidates[i].composite)

    return EvaluateResponse(
        prompt=req.prompt,
        backend=req.backend,
        model_name=req.model_name,
        candidates=candidates,
        best_candidate_index=best_idx,
    )


@app.post("/run-experiment", response_model=RunExperimentResponse)
def run_experiment(req: RunExperimentRequest):
    """Triggers a full batch experiment (every prompt x every configured
    model) and returns the aggregate summary. Equivalent to running
    run_experiment.py from the CLI, but over HTTP."""
    try:
        prompts = load_prompts(req.prompts_path)
        model_configs = load_model_configs(req.models_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    pipeline = ExperimentPipeline(db_path="outputs/experiments.db")
    exp_id = pipeline.run(
        experiment_name=req.experiment_name,
        prompts=prompts,
        model_configs=model_configs,
        candidates_per_prompt=req.candidates_per_prompt,
    )

    df = pipeline.get_dataframe(exp_id)
    summary = (
        df.groupby("model_name")[["rouge_l", "semantic_sim", "distinct_2", "composite", "latency_sec"]]
        .mean()
        .round(4)
        .to_dict(orient="index")
    )

    return RunExperimentResponse(
        experiment_id=exp_id,
        n_prompts=df["prompt_id"].nunique(),
        n_models=df["model_name"].nunique(),
        n_responses=len(df),
        summary=summary,
    )
