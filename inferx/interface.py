"""The interface shared by the proposed design and every baseline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    id: str
    speaker: str
    text: str
    date: str = ""
    session: int = 0

    @property
    def rendered(self) -> str:
        prefix = f"[{self.date}] " if self.date else ""
        return f"{prefix}{self.speaker}: {self.text}"


@dataclass
class Answer:
    text: str
    ttft_s: float | None = None
    total_s: float = 0.0
    retrieval_s: float = 0.0
    prompt_tokens: int = 0
    output_chunks: int = 0
    top_score: float | None = None
    sources: list[str] = field(default_factory=list)
    cloud_calls: int = 0
    cloud_bytes: int = 0
    sufficient: bool | None = None
    cloud_reply: str | None = None
    error: str | None = None


class MemorySystem(ABC):
    name: str
    cloud = None  # a CloudPolicy, or None for a fully local run

    @abstractmethod
    def ingest(self, message: Message) -> None:
        """Handle one incoming message during a live turn."""

    @abstractmethod
    def on_idle(self) -> dict:
        """Run deferred work and return counters describing it."""

    @abstractmethod
    def answer(self, question: str) -> Answer:
        """Answer a question from memory."""
