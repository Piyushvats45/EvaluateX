"""Tests for src/models.py — uses the mock backend so CI needs no network
or model downloads. Run with: pytest tests/test_models.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.models import MockRunner, build_runner


def test_mock_runner_generates_requested_count():
    runner = MockRunner(model_name="mock-concise", seed=42)
    results = runner.generate("What is the capital of France?", n=3)
    assert len(results) == 3
    for r in results:
        assert r.backend == "mock"
        assert isinstance(r.text, str) and len(r.text) > 0


def test_mock_runner_deterministic_given_same_prompt():
    runner = MockRunner(model_name="mock-concise", seed=1)
    r1 = runner.generate("hello world this is a test prompt", n=1)[0]
    r2 = runner.generate("hello world this is a test prompt", n=1)[0]
    assert r1.text == r2.text


def test_mock_runner_styles_differ():
    prompt = "Explain how photosynthesis works in plants today"
    concise = MockRunner(model_name="mock-concise").generate(prompt, n=1)[0].text
    verbose = MockRunner(model_name="mock-verbose").generate(prompt, n=1)[0].text
    assert concise != verbose


def test_build_runner_mock():
    runner = build_runner("mock", "mock-concise")
    assert runner.backend_name == "mock"


def test_build_runner_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_runner("not-a-real-backend", "whatever")


def test_build_runner_huggingface_missing_deps_raises_informative_error(monkeypatch):
    """If transformers/torch aren't installed, HuggingFaceRunner should raise
    a clear ImportError rather than a confusing stack trace."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("transformers", "torch"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="transformers/torch are not installed"):
        build_runner("huggingface", "distilgpt2")
