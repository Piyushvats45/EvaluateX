#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ExperimentPipeline, load_prompts, load_model_configs
from src.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Run an LLM evaluation & prompt optimization experiment.")
    parser.add_argument("--prompts", default="data/prompts.json", help="Path to prompts JSON file")
    parser.add_argument("--models", default="config/models.yaml", help="Path to model configs YAML file")
    parser.add_argument("--candidates", type=int, default=1, help="Candidate responses generated per prompt")
    parser.add_argument("--name", default="experiment", help="Experiment name/tag")
    parser.add_argument("--db", default="outputs/experiments.db", help="SQLite DB path")
    parser.add_argument("--report", default="outputs/report.html", help="Output HTML report path")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    model_configs = load_model_configs(args.models)

    print(f"Loaded {len(prompts)} prompts and {len(model_configs)} model configurations.")

    pipeline = ExperimentPipeline(db_path=args.db)
    try:
        exp_id = pipeline.run(
            experiment_name=args.name,
            prompts=prompts,
            model_configs=model_configs,
            candidates_per_prompt=args.candidates,
        )
    except ImportError as e:
        print(f"\n❌ {e}\n")
        print(
            "Tip: config/models.yaml uses real local models (backend: huggingface), "
            "which need `pip install transformers torch`. To try the framework "
            "without any downloads, run with the mock config instead:\n"
            "  python run_experiment.py --models config/models.mock.yaml"
        )
        sys.exit(1)
    print(f"Experiment logged with id={exp_id} in {args.db}")

    df = pipeline.get_dataframe(exp_id)
    csv_path = generate_report(df, exp_id, args.report, title=f"LLM Evaluation Report — {args.name}")

    print(f"Report written to:\n  HTML: {args.report}\n  CSV : {csv_path}")
    print("\n=== Per-model summary ===")
    print(
        df.groupby("model_name")[["rouge_l", "semantic_sim", "distinct_2", "composite", "latency_sec"]]
        .mean()
        .round(4)
        .sort_values("composite", ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
