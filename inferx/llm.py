"""Client for the local LLM server.

Refuses non-loopback addresses, so this client cannot make a cloud call.
Treats an empty reply as a failure, because Foundry Local can return HTTP 200 with
no text when inference fails during streaming. Counts output chunks itself, because
Foundry's completion_tokens over-reports.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from tokenizers import Tokenizer

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# Prompt + max_tokens under 64 fails on the QNN model because of its 64-token window.
MIN_MAX_TOKENS = 64


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    ttft_s: float
    total_s: float
    chunks: int
    prompt_tokens: int


class LocalLLM:
    def __init__(self, base_url: str, model: str, tokenizer_path: str | Path, deadline_s: float = 180):
        if urlparse(base_url).hostname not in LOOPBACK:
            raise ValueError(f"refusing non-local LLM address: {base_url}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.deadline_s = deadline_s
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded.ids) <= max_tokens:
            return text
        return self.tokenizer.decode(encoded.ids[:max_tokens])

    def complete(self, system: str, user: str, max_tokens: int = 200) -> Completion:
        self.calls += 1
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
            "temperature": 0,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        first = None
        chunks = 0
        pieces: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=self.deadline_s) as response:
                for raw in response:
                    if time.perf_counter() - start > self.deadline_s:
                        raise LLMError(f"deadline of {self.deadline_s}s exceeded")
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    choices = json.loads(line[5:]).get("choices") or []
                    piece = (choices[0].get("delta") or {}).get("content") if choices else None
                    if piece:
                        first = first or time.perf_counter()
                        chunks += 1
                        pieces.append(piece)
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(f"{type(error).__name__}: {error}") from error
        text = "".join(pieces).strip()
        if not text:
            raise LLMError("empty reply")
        return Completion(
            text=text,
            ttft_s=first - start,
            total_s=time.perf_counter() - start,
            chunks=chunks,
            prompt_tokens=self.count_tokens(system) + self.count_tokens(user),
        )
