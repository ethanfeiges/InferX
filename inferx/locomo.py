"""LoCoMo data loading and scoring. Scoring follows the official task_eval/evaluation.py."""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer

from .interface import Message

_stem = PorterStemmer().stem


def load(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sessions(sample: dict) -> list[tuple[int, str, list[Message]]]:
    """Return (session number, date, messages) in order."""
    conversation = sample["conversation"]
    numbers = sorted(
        int(key.split("_")[1])
        for key in conversation
        if re.fullmatch(r"session_\d+", key)
    )
    result = []
    for number in numbers:
        date = conversation.get(f"session_{number}_date_time", "")
        messages = []
        for turn in conversation[f"session_{number}"]:
            text = turn["text"]
            if turn.get("blip_caption"):
                text += f" [shared image: {turn['blip_caption']}]"
            messages.append(Message(turn["dia_id"], turn["speaker"], text, date, number))
        result.append((number, date, messages))
    return result


def _normalize(text: str) -> str:
    text = text.replace(",", "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, truth: str) -> float:
    predicted = [_stem(word) for word in _normalize(prediction).split()]
    expected = [_stem(word) for word in _normalize(truth).split()]
    same = sum((Counter(predicted) & Counter(expected)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(predicted), same / len(expected)
    return 2 * precision * recall / (precision + recall)


def _multi_f1(prediction: str, truth: str) -> float:
    predictions = [part.strip() for part in prediction.split(",")]
    truths = [part.strip() for part in truth.split(",")]
    return sum(max(f1_score(p, t) for p in predictions) for t in truths) / len(truths)


def score(qa: dict, prediction: str) -> float:
    category = qa["category"]
    if category == 5:
        lowered = prediction.lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    answer = str(qa["answer"])
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return _multi_f1(prediction, answer)
    return f1_score(prediction, answer)


def evidence_recall(qa: dict, sources: list[str]) -> float | None:
    """Fraction of evidence messages that reached the prompt. None if the question has no evidence."""
    evidence = qa.get("evidence") or []
    if not evidence:
        return None
    return sum(item in sources for item in evidence) / len(evidence)
