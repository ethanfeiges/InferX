import hashlib
import json
from pathlib import Path

import numpy as np

from inferx.memory_classifier import Item, LinearHead, MemoryClassifier, evaluate, pair_features

ROOT = Path(__file__).resolve().parent.parent


def fake_embed(text):
    vector = np.zeros(64, dtype=np.float32)
    for word in text.lower().split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
    return vector / max(np.linalg.norm(vector), 1e-9)


class FixedHead:
    """Stands in for a trained head, so replay logic can be tested on its own."""

    def __init__(self, classes, rule):
        self.classes, self.rule = classes, rule

    def predict(self, vector):
        return [self.rule(vector)]

    def proba(self, features):
        if self.rule:
            return np.array([[0.0, 1.0] if self.rule(features) == "yes" else [1.0, 0.0]])
        return np.array([[0.0, 1.0 if features[-1] > 0.3 else 0.0]])


def always(value):
    return FixedHead(["no", "yes"], lambda _: value)


def entry(id_, text, store=True, replaces=None, sensitive=False, kind="state"):
    return {"id": id_, "speaker": "Ana", "text": text, "store": store, "kind": kind if store else "none",
            "replaces": replaces, "sensitive": sensitive}


def test_head_learns_a_separable_problem_and_survives_saving(tmp_path):
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-1, 0.3, (60, 4)), rng.normal(1, 0.3, (60, 4))])
    labels = ["no"] * 60 + ["yes"] * 60
    head = LinearHead(["no", "yes"]).fit(x, labels)
    assert head.predict(x) == labels
    path = tmp_path / "head.npz"
    np.savez(path, **head.state("h"))
    assert LinearHead.load(np.load(path), "h").predict(x) == labels


def test_balanced_classes_still_find_a_rare_class():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(-1, 0.3, (300, 4)), rng.normal(1, 0.3, (6, 4))])
    head = LinearHead(["no", "yes"]).fit(x, ["no"] * 300 + ["yes"] * 6)
    assert head.predict(x[-6:]) == ["yes"] * 6


def test_pair_features_see_shared_words_and_speaker():
    a, b = fake_embed("dentist moved friday"), fake_embed("dentist booked monday")
    same = pair_features(a, b, "dentist moved friday", "dentist booked monday", "Cal", "Cal")
    other = pair_features(a, b, "dentist moved friday", "gym booked monday", "Cal", "Sam")
    assert same[-1] > other[-1]
    assert same[-2] == 1.0 and other[-2] == 0.0


def test_sensitive_message_is_never_stored_even_if_marked_worth_keeping():
    model = MemoryClassifier(fake_embed, always("yes"), FixedHead(["state"], lambda _: "state"), always("yes"), always("no"))
    decision, _ = model.decide("Ana", "code 123456", [])
    assert decision.sensitive and not decision.store


def test_replay_scores_updates_and_removes_replaced_items():
    model = MemoryClassifier(fake_embed, always("yes"), FixedHead(["state"], lambda _: "state"), always("no"), FixedHead(["no", "yes"], None))
    timeline = [entry("a", "dentist monday clinic"), entry("b", "dentist friday clinic", replaces="a")]
    result = evaluate(model, timeline)
    assert result["update"]["f1"] == 1.0
    assert result["store"]["f1"] == 1.0


def test_replay_counts_a_wrong_target_as_a_miss():
    model = MemoryClassifier(fake_embed, always("yes"), FixedHead(["state"], lambda _: "state"), always("no"), FixedHead(["no", "yes"], None))
    timeline = [entry("x", "dentist monday clinic"), entry("y", "gym tuesday"), entry("z", "dentist friday clinic", replaces="y")]
    result = evaluate(model, timeline)
    assert result["counts"]["update_tp"] == 0 and result["counts"]["wrong_target"] == 1


def test_handwritten_set_is_consistent():
    rows = json.loads((ROOT / "data/assistant/handwritten.json").read_text(encoding="utf-8"))
    seen = set()
    for row in rows:
        if row["replaces"]:
            assert row["replaces"] in seen and row["store"]
        if row["sensitive"]:
            assert not row["store"]
        assert (row["kind"] == "none") == (not row["store"])
        seen.add(row["id"])


def test_thresholds_survive_saving(tmp_path):
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-1, 0.3, (40, 64)), rng.normal(1, 0.3, (40, 64))])
    head = LinearHead(["no", "yes"]).fit(x, ["no"] * 40 + ["yes"] * 40)
    kind = LinearHead(["state", "event", "core"]).fit(x, (["state", "event"] * 20) + (["core"] * 40))
    model = MemoryClassifier(fake_embed, head, kind, head, head, 0.6, 3, 0.2, 0.9)
    path = tmp_path / "heads.npz"
    model.save(path)
    loaded = MemoryClassifier.load(path, fake_embed)
    assert (loaded.replace_threshold, loaded.candidates, loaded.store_threshold, loaded.sensitive_threshold) == (0.6, 3, 0.2, 0.9)


def test_events_and_other_kinds_are_never_replaced():
    def model(kind):
        return MemoryClassifier(fake_embed, always("yes"), FixedHead(["state", "event", "core"], lambda _: kind), always("no"),
                                FixedHead(["no", "yes"], None))

    text = "dentist friday clinic"
    stored = [Item("old", "Ana", "dentist monday clinic", fake_embed("Ana: dentist monday clinic"), "event")]
    assert model("state").decide("Ana", text, stored)[0].replaces is None
    assert model("event").decide("Ana", text, stored)[0].replaces is None
    stored[0].kind = "state"
    assert model("core").decide("Ana", text, stored)[0].replaces is None
    assert model("event").decide("Ana", text, stored)[0].replaces is None
    assert model("state").decide("Ana", text, stored)[0].replaces == "old"


def test_message_about_someone_else_cannot_replace_the_senders_fact():
    model = MemoryClassifier(fake_embed, always("yes"), FixedHead(["state"], lambda _: "state"), always("no"),
                             FixedHead(["no", "yes"], None))
    stored = [Item("own", "Lena", "our new place is 5 Fisher Row", fake_embed("Lena: our new place is 5 Fisher Row"))]
    assert model.decide("Lena", "my sister moved to 40 Quay Street place", stored)[0].replaces is None
    stored.append(Item("sis", "Lena", "my sister moved to Quay Street place", fake_embed("Lena: my sister moved to Quay Street place")))
    assert model.decide("Lena", "my sister moved to 40 Quay Street place", stored)[0].replaces == "sis"
