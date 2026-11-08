"""Small trained models for the memory controller's decisions.

Each decision is a linear head on top of the shared MiniLM embedding. Linear heads are cheap,
and easy to compile for an NPU later. Training happens off the device, on simulated task data.

Decisions:
  store:     is this message worth remembering?
  kind:      state, event or core (only asked of messages worth remembering)
  sensitive: is this a secret, such as a one-time code?
  replaces:  does this new message replace that stored one?
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

KINDS = ["state", "event", "core"]
WORD = re.compile(r"[a-z0-9$']+")
# "my sister moved" is about someone else, so it can't replace the sender's own address.
THIRD_PARTY = re.compile(r"\bmy (sister|brother|mum|mom|dad|mother|father|friend|boss|partner|wife|husband|son|daughter|"
                         r"cousin|aunt|uncle|neighbou?r|colleague|parents|grandma|grandad|flatmate|roommate)\b", re.I)
STOP = {"the", "a", "an", "is", "at", "on", "to", "for", "of", "in", "and", "your", "my", "i", "it", "now", "s"}


class LinearHead:
    """Softmax regression with balanced classes, fitted by full-batch gradient descent."""

    def __init__(self, classes: list, l2: float = 1e-3, lr: float = 0.5, steps: int = 1500):
        self.classes, self.l2, self.lr, self.steps = list(classes), l2, lr, steps
        self.mean = self.std = self.weights = self.bias = None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit(self, x: np.ndarray, labels: list) -> "LinearHead":
        index = {label: i for i, label in enumerate(self.classes)}
        y = np.array([index[label] for label in labels])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        x = self._scale(x)
        counts = np.bincount(y, minlength=len(self.classes))
        sample_weight = (len(y) / (len(self.classes) * np.maximum(counts, 1)))[y]
        self.weights = np.zeros((x.shape[1], len(self.classes)), dtype=np.float64)
        self.bias = np.zeros(len(self.classes), dtype=np.float64)
        one_hot = np.eye(len(self.classes))[y]
        velocity_w, velocity_b = np.zeros_like(self.weights), np.zeros_like(self.bias)
        for _ in range(self.steps):
            error = (self._softmax(x) - one_hot) * sample_weight[:, None]
            velocity_w = 0.9 * velocity_w + x.T @ error / len(y) + self.l2 * self.weights
            velocity_b = 0.9 * velocity_b + error.mean(0)
            self.weights -= self.lr * 0.1 * velocity_w
            self.bias -= self.lr * 0.1 * velocity_b
        return self

    def _softmax(self, scaled: np.ndarray) -> np.ndarray:
        logits = scaled @ self.weights + self.bias
        logits -= logits.max(1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(1, keepdims=True)

    def proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(self._scale(np.atleast_2d(x)))

    def predict(self, x: np.ndarray) -> list:
        return [self.classes[i] for i in self.proba(x).argmax(1)]

    def state(self, name: str) -> dict:
        return {f"{name}_weights": self.weights, f"{name}_bias": self.bias, f"{name}_mean": self.mean,
                f"{name}_std": self.std, f"{name}_classes": np.array(self.classes)}

    @classmethod
    def load(cls, data, name: str) -> "LinearHead":
        head = cls([str(value) for value in data[f"{name}_classes"]])
        head.weights, head.bias = data[f"{name}_weights"], data[f"{name}_bias"]
        head.mean, head.std = data[f"{name}_mean"], data[f"{name}_std"]
        return head


def render(speaker: str, text: str) -> str:
    return f"{speaker}: {text}"


def _words(text: str) -> set[str]:
    return {word for word in WORD.findall(text.lower()) if word not in STOP}


def pair_features(new_vector: np.ndarray, old_vector: np.ndarray, new_text: str, old_text: str,
                  new_speaker: str, old_speaker: str) -> np.ndarray:
    """Describe how a new message relates to a stored one, and whether it reads like a change."""
    new_words, old_words = _words(new_text), _words(old_text)
    overlap = len(new_words & old_words) / max(len(new_words | old_words), 1)
    extra = np.array([float(new_vector @ old_vector), float(new_speaker == old_speaker), overlap])
    return np.concatenate([new_vector, new_vector * old_vector, np.abs(new_vector - old_vector), extra])


@dataclass
class Item:
    id: str
    speaker: str
    text: str
    vector: np.ndarray
    kind: str = "state"


@dataclass
class Decision:
    store: bool
    kind: str | None
    sensitive: bool
    replaces: str | None


class MemoryClassifier:
    """The trained controller: one call per incoming message."""

    def __init__(self, embed: Callable[[str], np.ndarray], store: LinearHead, kind: LinearHead, sensitive: LinearHead,
                 replaces: LinearHead, replace_threshold: float = 0.5, candidates: int = 5,
                 store_threshold: float = 0.5, sensitive_threshold: float = 0.5):
        self.embed, self.store, self.kind, self.sensitive = embed, store, kind, sensitive
        self.replaces, self.replace_threshold, self.candidates = replaces, replace_threshold, candidates
        self.store_threshold, self.sensitive_threshold = store_threshold, sensitive_threshold

    @staticmethod
    def _yes(head, vector: np.ndarray) -> float:
        return float(head.proba(vector)[0, head.classes.index("yes")])

    def nearest(self, vector: np.ndarray, active: list[Item]) -> list[Item]:
        scored = sorted(active, key=lambda item: -float(vector @ item.vector))
        return scored[: self.candidates]

    def decide(self, speaker: str, text: str, active: list[Item]) -> tuple[Decision, np.ndarray]:
        vector = self.embed(render(speaker, text))
        sensitive = self._yes(self.sensitive, vector) >= self.sensitive_threshold
        if sensitive or self._yes(self.store, vector) < self.store_threshold:
            return Decision(False, None, sensitive, None), vector
        kind = self.kind.predict(vector)[0]
        best, best_score = None, self.replace_threshold
        # An event stays true once it has happened, and a fact can only replace a fact of its own kind.
        replaceable = [] if kind == "event" else [item for item in active if item.kind == kind]
        if THIRD_PARTY.search(text):
            replaceable = [item for item in replaceable if THIRD_PARTY.search(item.text)]
        for item in self.nearest(vector, replaceable):
            features = pair_features(vector, item.vector, text, item.text, speaker, item.speaker)
            score = self._yes(self.replaces, features)
            if score >= best_score:
                best, best_score = item.id, score
        return Decision(True, kind, False, best), vector

    def save(self, path: str | Path) -> None:
        np.savez(path, replace_threshold=self.replace_threshold, candidates=self.candidates,
                 store_threshold=self.store_threshold, sensitive_threshold=self.sensitive_threshold, **self.store.state("store"),
                 **self.kind.state("kind"), **self.sensitive.state("sensitive"), **self.replaces.state("replaces"))

    @classmethod
    def load(cls, path: str | Path, embed: Callable[[str], np.ndarray]) -> "MemoryClassifier":
        data = np.load(path)
        return cls(embed, *(LinearHead.load(data, name) for name in ("store", "kind", "sensitive", "replaces")),
                   float(data["replace_threshold"]), int(data["candidates"]), float(data["store_threshold"]),
                   float(data["sensitive_threshold"]))


def evaluate(classifier: MemoryClassifier, timeline: list[dict]) -> dict:
    """Replay a timeline in order, letting the model's own decisions shape what is stored.

    Each entry has id, speaker, text, store, kind, replaces and sensitive.
    """
    active: list[Item] = []
    counts = {name: 0 for name in ("store_tp", "store_fp", "store_fn", "kind_right", "kind_total", "update_tp", "update_fp",
                                   "update_fn", "wrong_target", "sensitive_total", "sensitive_kept", "sensitive_flagged",
                                   "false_sensitive")}
    errors = []
    for entry in timeline:
        decision, vector = classifier.decide(entry["speaker"], entry["text"], active)
        if entry["store"] and decision.store:
            counts["store_tp"] += 1
            counts["kind_total"] += 1
            counts["kind_right"] += decision.kind == entry["kind"]
        elif decision.store:
            counts["store_fp"] += 1
        elif entry["store"]:
            counts["store_fn"] += 1
        if entry["sensitive"]:
            counts["sensitive_total"] += 1
            counts["sensitive_kept"] += decision.store
            counts["sensitive_flagged"] += decision.sensitive
        elif decision.sensitive:
            counts["false_sensitive"] += 1
        if entry["replaces"] and decision.replaces == entry["replaces"]:
            counts["update_tp"] += 1
        elif entry["replaces"]:
            counts["update_fn"] += 1
            counts["wrong_target"] += decision.replaces is not None
        elif decision.replaces:
            counts["update_fp"] += 1
        if decision.store != entry["store"] or (decision.replaces or None) != (entry["replaces"] or None):
            errors.append({"id": entry["id"], "text": entry["text"], "wanted_store": entry["store"], "got_store": decision.store,
                           "wanted_replaces": entry["replaces"], "got_replaces": decision.replaces})
        if decision.store:
            if decision.replaces:
                active = [item for item in active if item.id != decision.replaces]
            active.append(Item(entry["id"], entry["speaker"], entry["text"], vector, decision.kind))
    return {**summarize(counts), "counts": counts, "errors": errors}


def _f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(score, 3), "n": tp + fn}


def summarize(counts: dict) -> dict:
    return {
        "store": _f1(counts["store_tp"], counts["store_fp"], counts["store_fn"]),
        "update": _f1(counts["update_tp"], counts["update_fp"], counts["update_fn"]),
        "kind_accuracy": round(counts["kind_right"] / max(counts["kind_total"], 1), 3),
        "sensitive_kept": f"{counts['sensitive_kept']} of {counts['sensitive_total']}",
        "false_sensitive": counts["false_sensitive"],
    }
