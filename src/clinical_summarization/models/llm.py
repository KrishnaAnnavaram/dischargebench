"""Local LLM adapter (Ollama) with deterministic decoding and versioned prompts."""
from __future__ import annotations

from importlib import resources

from ..text import strip_reasoning


def load_prompt(name: str) -> str:
    return resources.files("clinical_summarization").joinpath("prompts", f"{name}.txt").read_text(encoding="utf-8")


class OllamaSummarizer:
    def __init__(self, model: str, prompt: str = "bhc_one_paragraph", temperature: float = 0.0, seed: int = 42,
                 host: str = "http://localhost:11434", timeout: int = 600):
        self.name = f"ollama:{model}"
        self.model, self.temperature, self.seed = model, temperature, seed
        self.template = load_prompt(prompt)
        self.prompt_name = prompt
        self.url = host.rstrip("/") + "/api/generate"
        self.timeout = timeout

    def summarize(self, texts: list[str]) -> list[str]:
        import requests  # optional dependency: pip install ".[llm]"

        out = []
        for text in texts:
            resp = requests.post(self.url, timeout=self.timeout, json={
                "model": self.model,
                "prompt": self.template.format(note=text),
                "stream": False,
                "options": {"temperature": self.temperature, "seed": self.seed},
            })
            resp.raise_for_status()
            out.append(strip_reasoning(resp.json().get("response", "")))
        return out
