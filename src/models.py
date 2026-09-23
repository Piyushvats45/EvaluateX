from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationResult:
    text: str
    latency_sec: float
    model_name: str
    backend: str

class BaseRunner:
    backend_name = "base"

    def generate(self, prompt: str, n: int = 1, **gen_kwargs) -> list[GenerationResult]:
        raise NotImplementedError

class HuggingFaceRunner(BaseRunner):
    """Loads a local, open-weight causal LM once and reuses it for every
    prompt. Fully free — the model runs on your own CPU/GPU."""

    backend_name = "huggingface"

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers/torch are not installed. Run:\n"
                "  pip install transformers torch\n"
                "or use backend='mock' to try the framework without them."
            ) from e

        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, n: int = 1, max_new_tokens: int = 80, **gen_kwargs) -> list[GenerationResult]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        results = []
        for _ in range(n):
            start = time.time()
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
            latency = time.time() - start
            text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            results.append(
                GenerationResult(
                    text=text.strip(),
                    latency_sec=latency,
                    model_name=self.model_name,
                    backend=self.backend_name,
                )
            )
        return results

class MockRunner(BaseRunner):
    """Deterministic, offline, zero-dependency stand-in for a real LLM.

    Useful for: unit testing the pipeline, demoing the framework on a
    machine with no internet access, and reproducible CI runs where you
    don't want network variance in the test suite.

    Each "model_name" produces a distinctly-flavored response so that
    comparative evaluation across configurations still shows meaningful
    differences.
    """

    backend_name = "mock"

    _STYLE_TEMPLATES = {
        "mock-concise": "{summary}",
        "mock-verbose": (
            "To thoroughly address this, consider the following: {summary} "
            "In addition, it is worth noting several related points that "
            "provide broader context and nuance to the original question."
        ),
        "mock-creative": "Imagine it this way: {summary} — and there's more to explore.",
    }

    def __init__(self, model_name: str = "mock-concise", seed: Optional[int] = None):
        self.model_name = model_name
        self._rng = random.Random(seed)

    def _pseudo_summary(self, prompt: str) -> str:
        """Produces a short, deterministic-but-prompt-dependent 'answer' by
        hashing the prompt and picking words from it — simulates a model
        loosely reflecting the input without needing real inference."""
        words = [w for w in prompt.split() if len(w) > 3]
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        rng = random.Random(h)
        rng.shuffle(words)
        picked = words[: max(3, len(words) // 3)]
        return " ".join(picked) if picked else "a relevant, concise answer"

    def generate(self, prompt: str, n: int = 1, max_new_tokens: int = 80, **gen_kwargs) -> list[GenerationResult]:
        template = self._STYLE_TEMPLATES.get(self.model_name, "{summary}")
        results = []
        for i in range(n):
            start = time.time()
            time.sleep(0.001)  # simulate negligible latency
            summary = self._pseudo_summary(prompt + str(i))
            text = template.format(summary=summary)
            latency = time.time() - start
            results.append(
                GenerationResult(
                    text=text,
                    latency_sec=latency,
                    model_name=self.model_name,
                    backend=self.backend_name,
                )
            )
        return results

def build_runner(backend: str, model_name: str, **kwargs) -> BaseRunner:
    """Factory: build the right runner from a config entry."""
    if backend == "huggingface":
        return HuggingFaceRunner(model_name=model_name, **kwargs)
    if backend == "mock":
        return MockRunner(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown backend '{backend}'. Use 'huggingface' or 'mock'.")
