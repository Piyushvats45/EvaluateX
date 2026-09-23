from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from . import metrics as metrics_mod
from .database import ExperimentDB
from .models import build_runner

@dataclass
class ModelConfig:
    name: str
    backend: str
    model_name: str
    params: dict

@dataclass
class PromptItem:
    task: str
    prompt: str
    reference: Optional[str] = None

def load_prompts(path: str) -> list[PromptItem]:
    data = json.loads(Path(path).read_text())
    return [PromptItem(task=d.get("task", "general"), prompt=d["prompt"], reference=d.get("reference")) for d in data]

def load_model_configs(path: str) -> list[ModelConfig]:
    raw = yaml.safe_load(Path(path).read_text())
    return [
        ModelConfig(
            name=cfg["name"],
            backend=cfg["backend"],
            model_name=cfg["model_name"],
            params=cfg.get("params", {}),
        )
        for cfg in raw["models"]
    ]

class ExperimentPipeline:
    def __init__(self, db_path: str = "outputs/experiments.db"):
        self.db = ExperimentDB(db_path)

    def run(
        self,
        experiment_name: str,
        prompts: list[PromptItem],
        model_configs: list[ModelConfig],
        candidates_per_prompt: int = 1,
        metric_weights: Optional[dict] = None,
        human_scores: Optional[dict] = None,
        notes: str = "",
    ) -> int:
        """
        human_scores: optional dict of {(prompt_index, model_name): score}
        used to populate the human_score column for later metric-human
        agreement analysis (see report.py / metrics.metric_human_agreement).
        """
        exp_id = self.db.create_experiment(experiment_name, notes=notes)

        runners = {cfg.name: build_runner(cfg.backend, cfg.model_name, **cfg.params) for cfg in model_configs}

        for p_idx, item in enumerate(prompts):
            prompt_id = self.db.add_prompt(exp_id, item.task, item.prompt, item.reference)

            for cfg in model_configs:
                runner = runners[cfg.name]
                generations = runner.generate(item.prompt, n=candidates_per_prompt)

                for gen in generations:
                    response_id = self.db.add_response(
                        prompt_id=prompt_id,
                        model_name=cfg.name,
                        backend=gen.backend,
                        response_text=gen.text,
                        latency_sec=gen.latency_sec,
                    )
                    result = metrics_mod.score_response(item.reference, gen.text, weights=metric_weights)

                    human_score = None
                    if human_scores:
                        human_score = human_scores.get((p_idx, cfg.name))

                    self.db.add_metrics(response_id, result, human_score=human_score)

        return exp_id

    def get_dataframe(self, experiment_id: int):
        import pandas as pd

        rows = self.db.fetch_results(experiment_id)
        return pd.DataFrame(rows)
