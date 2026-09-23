import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ExperimentPipeline, ModelConfig, PromptItem

def _sample_prompts():
    return [
        PromptItem(task="qa", prompt="Question: What is 2+2? Answer:", reference="4"),
        PromptItem(task="summarization", prompt="Summarize: the sky is blue.", reference="The sky is blue."),
    ]


def _sample_configs():
    return [
        ModelConfig(name="concise", backend="mock", model_name="mock-concise", params={}),
        ModelConfig(name="verbose", backend="mock", model_name="mock-verbose", params={}),
    ]

def test_pipeline_runs_and_logs_every_combination(tmp_path):
    db_path = str(tmp_path / "test.db")
    pipeline = ExperimentPipeline(db_path=db_path)

    exp_id = pipeline.run(
        experiment_name="test-run",
        prompts=_sample_prompts(),
        model_configs=_sample_configs(),
        candidates_per_prompt=2,
    )

    df = pipeline.get_dataframe(exp_id)

    # 2 prompts x 2 models x 2 candidates = 8 logged responses
    assert len(df) == 8
    assert set(df["model_name"].unique()) == {"concise", "verbose"}
    assert df["composite"].notna().all()

def test_pipeline_records_human_scores(tmp_path):
    db_path = str(tmp_path / "test_human.db")
    pipeline = ExperimentPipeline(db_path=db_path)

    human_scores = {(0, "concise"): 0.8, (0, "verbose"): 0.3}

    exp_id = pipeline.run(
        experiment_name="test-human-run",
        prompts=_sample_prompts(),
        model_configs=_sample_configs(),
        candidates_per_prompt=1,
        human_scores=human_scores,
    )

    df = pipeline.get_dataframe(exp_id)
    rated = df[df["human_score"].notna()]
    assert len(rated) == 2
    assert set(rated["human_score"]) == {0.8, 0.3}

def test_multiple_experiments_are_isolated(tmp_path):
    db_path = str(tmp_path / "multi.db")
    pipeline = ExperimentPipeline(db_path=db_path)

    exp_id_1 = pipeline.run("run-1", _sample_prompts(), _sample_configs(), candidates_per_prompt=1)
    exp_id_2 = pipeline.run("run-2", _sample_prompts()[:1], _sample_configs(), candidates_per_prompt=1)

    df1 = pipeline.get_dataframe(exp_id_1)
    df2 = pipeline.get_dataframe(exp_id_2)

    assert len(df1) == 4  # 2 prompts x 2 models
    assert len(df2) == 2  # 1 prompt x 2 models
