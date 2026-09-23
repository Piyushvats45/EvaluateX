#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ExperimentPipeline, load_prompts, load_model_configs
from src.report import generate_report


def main():
    prompts = load_prompts("data/prompts.json")
    # Uses the mock config deliberately: this demo is about the agreement
    # *statistic* (Pearson/Spearman between automated and human scores),
    # not about model output quality, so it should run instantly with no
    # downloads. Swap to config/models.yaml to run the same analysis against
    # real model output.
    model_configs = load_model_configs("config/models.mock.yaml")

    # Simulated human ratings for the first few (prompt_index, model_name)
    # pairs -- in a real workflow these would come from a human review UI.
    human_scores = {
        (0, "concise-style"): 0.8,
        (0, "verbose-style"): 0.4,
        (0, "creative-style"): 0.5,
        (1, "concise-style"): 0.7,
        (1, "verbose-style"): 0.3,
        (1, "creative-style"): 0.6,
        (2, "concise-style"): 0.9,
        (2, "verbose-style"): 0.5,
        (2, "creative-style"): 0.4,
    }

    pipeline = ExperimentPipeline(db_path="outputs/human_agreement.db")
    exp_id = pipeline.run(
        experiment_name="human-agreement-demo",
        prompts=prompts,
        model_configs=model_configs,
        candidates_per_prompt=1,
        human_scores=human_scores,
    )

    df = pipeline.get_dataframe(exp_id)
    generate_report(df, exp_id, "outputs/human_agreement_report.html", title="Metric-Human Agreement Demo")
    print("Report written to outputs/human_agreement_report.html")
    print(df[["model_name", "composite", "human_score"]].dropna().to_string(index=False))


if __name__ == "__main__":
    main()
