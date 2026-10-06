"""Hugging Face encoder-decoder adapter with explicit long-input handling (truncate or map-reduce chunking)."""
from __future__ import annotations


class HFSeq2SeqSummarizer:
    def __init__(self, model_name: str, max_input_tokens: int = 1024, max_new_tokens: int = 256, num_beams: int = 4,
                 long_input: str = "chunk", chunk_overlap: int = 64, prefix: str = "", device: str | None = None):
        if long_input not in {"truncate", "chunk"}:
            raise ValueError("long_input must be 'truncate' or 'chunk'")
        import torch  # optional dependency: pip install ".[hf]"
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = f"hf:{model_name}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self.max_in, self.max_new, self.beams = max_input_tokens, max_new_tokens, num_beams
        self.long_input, self.overlap, self.prefix = long_input, chunk_overlap, prefix
        self.truncated = 0      # how many inputs exceeded max_input_tokens (reported, never silent)

    def _generate(self, text: str) -> str:
        import torch

        enc = self.tok(self.prefix + text, return_tensors="pt", truncation=True, max_length=self.max_in).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**enc, num_beams=self.beams, max_new_tokens=self.max_new, early_stopping=True)
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()

    def _chunks(self, text: str) -> list[str]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        size = self.max_in - 32                                  # room for special tokens / prefix
        step = max(1, size - self.overlap)
        return [self.tok.decode(ids[i:i + size]) for i in range(0, len(ids), step)]

    def summarize(self, texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            n_tokens = len(self.tok(text, add_special_tokens=False)["input_ids"])
            if n_tokens <= self.max_in - 32:
                out.append(self._generate(text))
                continue
            self.truncated += 1
            if self.long_input == "truncate":
                out.append(self._generate(text))
                continue
            partial = [self._generate(c) for c in self._chunks(text)]      # map
            out.append(self._generate(" ".join(partial)))                  # reduce
        return out
