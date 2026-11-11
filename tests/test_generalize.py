"""The additions meant to generalize past the simulated training data: secret rules, the owner guard,
and handing unsure memory decisions to the main LLM."""
import numpy as np
import pytest

from inferx.interface import Message
from inferx.memory_classifier import Decision, Item, MemoryClassifier, different_item, different_owner, names
from inferx.sensitive import iban_valid, luhn, secret_kind
from inferx.systems import TrainedMemory, TrainedPlus
from test_core import FakeLLM, fake_embed
from test_memory_classifier import FixedHead, always


@pytest.mark.parametrize("text, kind", [
    ("Card 4111 1111 1111 1111 exp 04/27", "card number"),
    ("IBAN GB82 WEST 1234 5698 7654 32", "bank account"),
    ("wifi password is sunflower22", "password or PIN"),
    ("Temporary password for your account: 9412-77", "password or PIN"),
    ("My PIN is 4821", "password or PIN"),
    ("Security code: 118557. It expires soon.", "one-time code"),
    ("334442 is your authentication number", "one-time code"),
    ("G-208841 is your account confirmation code", "one-time code"),
    ("Confirm your payment with 7674", "one-time code"),
    ("Enter 804 to finish signing in on your new phone", "one-time code"),
    ("SSN 123-45-6789", "social security number"),
    ("NI number AB 12 34 56 C", "national insurance number"),
    ("here's the key sk-proj-abcdefghijklmnopqrstu", "API key"),
])
def test_secret_formats_are_caught(text, kind):
    assert secret_kind(text) == kind


@pytest.mark.parametrize("text", [
    "Card 4111 1111 1111 1112",  # fails the Luhn check
    "01232 381102 is my new cell",
    "Security deposit is 500 pounds",
    "The code review is at 1500",
    "Dress code is 1920s",
    "Area code is 020",
    "Confirm your appointment on 12 May 2025 at 10:30",
    "Your order 1234 shipped",
    "Use 2 eggs",
    "I'm allergic to peanuts",  # health facts are kept on purpose
])
def test_ordinary_messages_are_not_secrets(text):
    assert secret_kind(text) is None


def test_checksums():
    assert luhn("4111111111111111") and not luhn("4111111111111112")
    assert iban_valid("GB82 WEST 1234 5698 7654 32") and not iban_valid("GB82 WEST 1234 5698 7654 33")


def item(id_, speaker, text, kind="state"):
    return Item(id_, speaker, text, fake_embed(f"{speaker}: {text}"), kind)


def test_owner_guard_only_blocks_first_person_messages_about_someone_else():
    dara = item("d", "Dara", "03984 940839 is my new cell")
    assert different_owner("Ivan", "01232 381102 is my new cell", dara)
    assert not different_owner("Dara", "01232 381102 is my new cell", dara)
    assert not different_owner("Omar", "They're full, so dinner at the Old Mill now", dara)


def test_names_skip_dates_sentence_starts_and_the_people_talking():
    assert names("Booked: LH514 to Prague, Thursday 19 June.", ("Grace's calendar",)) == {"lh514", "prague"}
    assert names("Dentist moved to 9 May", ("Ana",)) == set()
    assert names("lunch with Grace at Cafe Mira", ("Grace's calendar",)) == {"cafe mira"}


def test_different_item_needs_names_on_both_sides():
    flight = item("f", "Grace's calendar", "Booked: LH514 to Prague, Thursday 19 June.")
    assert different_item("Grace's calendar", "Booked: FR211 to Vienna, Sun 22 Jun.", flight)
    assert not different_item("Grace's calendar", "LH514 now leaves at 2 pm.", flight)
    assert not different_item("Grace's calendar", "Flight moved to 21 June.", flight)


def model(**kwargs):
    return MemoryClassifier(fake_embed, always("yes"), FixedHead(["state"], lambda _: "state"), always("no"),
                            always("yes"), **kwargs)


def test_guard_removes_a_replacement_target():
    stored = [item("d", "Dara", "03984 940839 is my new cell")]
    assert model().decide("Ivan", "01232 381102 is my new cell", stored)[0].replaces == "d"
    guarded = model(guards=(different_owner,)).decide("Ivan", "01232 381102 is my new cell", stored)[0]
    assert guarded.replaces is None and guarded.candidate is None


def test_decision_reports_the_best_candidate_even_below_the_threshold():
    stored = [item("old", "Ana", "dentist monday clinic")]
    decision = model(replace_threshold=1.1).decide("Ana", "dentist friday clinic", stored)[0]
    assert decision.replaces is None and decision.candidate == "old" and decision.replace_score == 1.0


class ScriptedClassifier:
    """A classifier with fixed scores, as TrainedMemory sees it."""

    def __init__(self, decisions, store_threshold=0.5, replace_threshold=0.7):
        self.decisions, self.guards = decisions, ()
        self.store_threshold, self.replace_threshold = store_threshold, replace_threshold

    def decide(self, speaker, text, active, store_threshold=None):
        decision = self.decisions.get(text, Decision(False, None, False, None, 0.0))
        threshold = self.store_threshold if store_threshold is None else store_threshold
        if decision.store_score is not None and decision.store_score < threshold:
            decision = Decision(False, None, decision.sensitive, None, decision.store_score)
        return decision, fake_embed(text)


def msg(i, text, speaker="Ana"):
    return Message(f"D1:{i}", speaker, text, "8 May 2023", 1)


def test_secret_rules_drop_what_the_head_missed():
    system = TrainedMemory(FakeLLM(), fake_embed, ScriptedClassifier({}), secret_rules=True)
    system.ingest(msg(1, "My PIN is 4821"))
    assert system.store.counts()["recall"] == 0
    stats = system.on_idle()
    assert stats["secrets_dropped"] == 1 and stats["secrets_by_rule"] == 1


def test_secret_rules_are_off_by_default():
    system = TrainedMemory(FakeLLM(), fake_embed, ScriptedClassifier({}))
    system.ingest(msg(1, "My PIN is 4821"))
    assert system.store.counts()["recall"] == 1


def keep(score=0.9, candidate=None, replace_score=None, replaces=None):
    return Decision(True, "state", False, replaces, score, candidate, replace_score)


def test_unsure_update_waits_for_the_llm_and_happens_on_yes():
    llm = FakeLLM({"Answer:": "YES", "Earlier: Ana: dentist Monday": "Ana's dentist visit is on Friday."})
    decisions = {"dentist Monday": keep(), "dentist Friday": keep(candidate="D1:1", replace_score=0.6)}
    system = TrainedMemory(llm, fake_embed, ScriptedClassifier(decisions), ask_when_unsure=True)
    system.ingest(msg(1, "dentist Monday"))
    system.ingest(msg(2, "dentist Friday"))
    assert system.store.outdated() == set()  # not replaced until phi agrees
    stats = system.on_idle()
    assert stats["asked_replaces"] == 1 and stats["llm_said_replaces"] == 1
    assert system.store.outdated() == {"D1:1"} and stats["update"] == 1


def test_unsure_update_keeps_both_on_no():
    llm = FakeLLM({"Answer:": "NO", "dentist": "Ana has a dentist visit."})
    decisions = {"dentist Monday": keep(), "dentist Friday": keep(candidate="D1:1", replace_score=0.75, replaces="D1:1")}
    system = TrainedMemory(llm, fake_embed, ScriptedClassifier(decisions), ask_when_unsure=True)
    system.ingest(msg(1, "dentist Monday"))
    system.ingest(msg(2, "dentist Friday"))
    stats = system.on_idle()
    assert stats["asked_replaces"] == 1 and stats["llm_said_replaces"] == 0
    assert system.store.outdated() == set() and stats["update"] == 0


def test_sure_update_is_not_asked():
    llm = FakeLLM({"dentist": "Ana has a dentist visit."})
    decisions = {"dentist Monday": keep(), "dentist Friday": keep(candidate="D1:1", replace_score=0.99, replaces="D1:1")}
    system = TrainedMemory(llm, fake_embed, ScriptedClassifier(decisions), ask_when_unsure=True)
    system.ingest(msg(1, "dentist Monday"))
    system.ingest(msg(2, "dentist Friday"))
    assert system.store.outdated() == {"D1:1"}
    assert system.on_idle()["asked_replaces"] == 0


def test_confident_update_between_two_named_things_is_asked():
    llm = FakeLLM({"Answer:": "NO", "Booked": "Ana booked a flight."})
    decisions = {"Booked LH514 to Prague": keep(),
                 "Booked FR211 to Vienna": keep(candidate="D1:1", replace_score=0.99, replaces="D1:1")}
    system = TrainedMemory(llm, fake_embed, ScriptedClassifier(decisions), ask_when_unsure=True)
    system.ingest(msg(1, "Booked LH514 to Prague", "Ana's calendar"))
    system.ingest(msg(2, "Booked FR211 to Vienna", "Ana's calendar"))
    stats = system.on_idle()
    assert stats["asked_replaces"] == 1 and system.store.outdated() == set()


def test_unsure_message_is_kept_only_if_the_llm_finds_a_fact():
    llm = FakeLLM({"gym": "Ana goes to the gym on Tuesdays."})
    decisions = {"gym on tuesdays": keep(score=0.45), "lol ok": keep(score=0.42), "noise": keep(score=0.1)}
    system = TrainedMemory(llm, fake_embed, ScriptedClassifier(decisions), ask_when_unsure=True)
    for i, text in enumerate(decisions, 1):
        system.ingest(msg(i, text))
    stats = system.on_idle()
    assert stats["unsure_store"] == 2 and stats["unsure_store_kept"] == 1
    assert system.store.counts()["facts_active"] == 1 and system.store.counts()["recall"] == 3
    assert [item.id for item in system.active] == ["D1:1"]


def test_trained_plus_turns_everything_on_and_rejects_unknown_options():
    classifier = ScriptedClassifier({})
    system = TrainedPlus(FakeLLM(), fake_embed, classifier)
    assert system.name == "trained_plus" and system.secret_rules and system.ask_when_unsure
    assert different_owner in classifier.guards
    with pytest.raises(ValueError):
        TrainedPlus(FakeLLM(), fake_embed, ScriptedClassifier({}), options=("magic",))
