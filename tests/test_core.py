import hashlib
from datetime import date

import numpy as np
import pytest

from inferx.assistant_score import contains, correct, out_of_date
from inferx.controller import Decision, core_facts, decide, valid_fact, worth_remembering
from inferx.context import fit
from inferx.dates import annotate, parse_date, remove_unsupported_dates
from inferx.interface import Message
from inferx.llm import Completion, LLMError, LocalLLM
from inferx.locomo import evidence_recall, f1_score, score
from inferx.store import MemoryStore
from inferx.memory_classifier import Decision as MemoryDecision
from inferx.cloud import CloudClient, CloudPolicy
from inferx.systems import DecoupledMemory, FullContext, LocalMemory, PlainRetrieval, TrainedMemory, entities


def fake_embed(text):
    """Bag-of-words hash embedding: stable, and similar text gets similar vectors."""
    vector = np.zeros(384, dtype=np.float32)
    for word in text.lower().split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1
    return vector / max(np.linalg.norm(vector), 1e-9)


class FakeLLM:
    def __init__(self, replies=None, fail=False):
        self.replies = replies or {}
        self.fail = fail
        self.prompts = []

    def count_tokens(self, text):
        return len(text.split())

    def complete(self, system, user, max_tokens=200):
        self.prompts.append(user)
        if self.fail:
            raise LLMError("boom")
        text = next((reply for key, reply in self.replies.items() if key in user), "NONE")
        return Completion(text, 0.1, 0.2, 3, self.count_tokens(user))


def msg(i, text, speaker="Ana"):
    return Message(f"D1:{i}", speaker, text, "8 May 2023", 1)


def test_store_marks_superseded_facts_without_erasing():
    store = MemoryStore()
    old = store.add_fact("Ana lives in Rome", "D1:1", "", fake_embed("Ana lives in Rome"))
    new = store.add_fact("Ana lives in Oslo", "D1:2", "", fake_embed("Ana lives in Oslo"))
    store.supersede(old, new)
    hits = store.search_facts(fake_embed("Ana lives in Rome"), 5)
    assert [hit.rowid for hit in hits] == [new]
    assert store.counts()["facts_superseded"] == 1
    assert store.db.execute("select count(*) from archival").fetchone()[0] == 2


def test_recall_search_returns_nearest_with_score():
    store = MemoryStore()
    for i, text in enumerate(["cats are great pets", "the stock market fell"]):
        message = msg(i, text)
        store.add_message(message, fake_embed(message.rendered))
    hit = store.search_recall(fake_embed("Ana: cats are great pets"), 1)[0]
    assert hit.source == "D1:0" and hit.score > 0.5


def test_controller_filters_and_extracts_core():
    assert not worth_remembering(msg(1, "Hey Mel!"))
    assert not worth_remembering(msg(2, "Thanks so much, that is great!"))
    assert worth_remembering(msg(3, "I went to the support group yesterday."))
    assert core_facts(msg(4, "By the way I am allergic to walnuts, sadly.")) == [
        ("Ana.allergy", "Ana allergy: walnuts")
    ]


@pytest.mark.parametrize(
    "fact,nearest,score_,expected",
    [
        ("a", None, None, Decision.ADD),
        ("a", "b", 0.5, Decision.ADD),
        ("same", "same", 0.99, Decision.IGNORE),
        ("Ana has 3 cats", "Ana has 2 cats", 0.97, Decision.UPDATE),
        ("close", "near", 0.9, Decision.AMBIGUOUS),
    ],
)
def test_decide(fact, nearest, score_, expected):
    assert decide(fact, nearest, score_) is expected


def test_later_similar_event_never_replaces_an_earlier_one():
    old = "Gina launched her online clothes store on 16 March 2023."
    new = "Gina launched her online clothes store on 16 June 2023."
    assert decide(new, old, 0.97) is Decision.ADD
    assert decide("Ana lives in Oslo since 2023", "Ana lives in Rome since 2021", 0.97) is Decision.UPDATE


def test_judge_cannot_replace_an_event():
    old = "Ana went to Rome."
    new = "Ana went to Oslo."
    llm = FakeLLM({"went to Oslo": "UPDATE"})
    system = DecoupledMemory(llm, fake_embed)
    assert system._judge(new, old, {"llm_calls": 0, "llm_s": 0.0, "errors": 0}) is Decision.ADD
    llm = FakeLLM({"Stored fact": "UPDATE"})
    system = DecoupledMemory(llm, fake_embed)
    stats = {"llm_calls": 0, "llm_s": 0.0, "errors": 0}
    assert system._judge("Ana lives in Oslo.", "Ana lives in Rome.", stats) is Decision.UPDATE


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I went yesterday.", "I went yesterday (7 May 2023)."),
        ("Started last week.", "Started last week (the week before 8 May 2023)."),
        ("Going next month.", "Going next month (June 2023)."),
        ("Saw it 3 days ago.", "Saw it 3 days ago (5 May 2023)."),
        ("Joined two weeks ago.", "Joined two weeks ago (2 weeks before 8 May 2023)."),
        # 8 May 2023 is a Monday, so last Monday is a week earlier.
        ("Met last Monday.", "Met last Monday (1 May 2023)."),
        ("Meet next Friday.", "Meet next Friday (12 May 2023)."),
        ("Got it a few years ago.", "Got it a few years ago."),
        ("My last chance.", "My last chance."),
    ],
)
def test_annotate_resolves_only_exact_relative_times(text, expected):
    assert annotate(text, date(2023, 5, 8)) == expected


def test_parse_date_reads_locomo_format():
    assert parse_date("7:18 pm on 27 May, 2023") == date(2023, 5, 27)
    assert parse_date("8 May 2023") == date(2023, 5, 8)
    assert parse_date("sometime") is None


def test_unsupported_dates_are_removed_from_facts():
    assert remove_unsupported_dates(
        "Gina got a tattoo representing freedom on 8 February 2023.", "Got the tattoo a few years ago."
    ) == ("Gina got a tattoo representing freedom.", 1)
    assert remove_unsupported_dates(
        "Ana went to Rome on 7 May 2023.", "I went yesterday (7 May 2023) to Rome."
    ) == ("Ana went to Rome on 7 May 2023.", 0)
    assert remove_unsupported_dates("Ana flew on 3 June 2023.", "My flight is on June 3rd")[1] == 0
    assert remove_unsupported_dates("Ana flew on 3 June 2023.", "I flew on 13 June")[1] == 1


def test_rewrite_prompt_has_resolved_date_and_not_message_date():
    llm = FakeLLM({"yesterday": "Ana went to Rome on 7 May 2023 and again on 2 January 2022."})
    system = DecoupledMemory(llm, fake_embed)
    system.ingest(msg(1, "I went to Rome yesterday."))
    stats = system.on_idle()
    assert llm.prompts == ["Ana: I went to Rome yesterday (7 May 2023)."]
    assert stats["dates_removed"] == 1
    assert system.retrieve("Rome")[0].text == "[8 May 2023] Ana went to Rome on 7 May 2023 and again."


def test_fit_stops_at_budget():
    assert fit(["one two", "three four", "five"], 4, lambda t: len(t.split())) == ["one two"]


def test_entities_skips_common_words():
    assert entities(["Hey Caroline went to Paris on 7 May 2023"]) == ["Caroline", "Paris", "7 May 2023"]


def test_decoupled_writes_only_at_idle_and_uses_fact_over_raw():
    llm = FakeLLM({"support group": "Ana went to a support group on 7 May 2023."})
    system = DecoupledMemory(llm, fake_embed)
    system.ingest(msg(1, "I went to the support group yesterday."))
    assert llm.prompts == []  # nothing runs on the LLM during ingest
    assert system.store.counts()["queued"] == 1
    assert system.store.counts()["facts_active"] == 0
    stats = system.on_idle()
    assert stats["add"] == 1 and system.store.counts()["queued"] == 0
    hits = system.retrieve("support group")
    assert [hit.tier for hit in hits] == ["archival"]
    assert hits[0].source == "D1:1"


def test_controller_rejects_facts_that_are_not_standalone():
    assert valid_fact("Ana went to Rome on 7 May 2023.", "Ana")
    assert not valid_fact("I went to Rome.", "Ana")
    assert not valid_fact("You went to Rome, Ana.", "Ana")
    assert not valid_fact("The speaker went to Rome.", "Ana")
    assert not valid_fact("Ana: I went to Rome.", "Ana")


def test_rejected_rewrite_keeps_raw_message_only():
    llm = FakeLLM({"support group": "I went to a support group."})
    system = DecoupledMemory(llm, fake_embed)
    system.ingest(msg(1, "I went to the support group yesterday."))
    stats = system.on_idle()
    assert stats["rejected"] == 1 and system.store.counts()["facts_active"] == 0
    assert system.retrieve("support group")[0].tier == "recall"


def test_question_matches_rank_above_entity_matches():
    system = DecoupledMemory(FakeLLM(), fake_embed, k=5)
    system.ingest(msg(1, "Ana loves the blue kayak"))
    system.ingest(msg(2, "Ana Ana Ana Ana"))
    assert system.retrieve("blue kayak")[0].source == "D1:1"


def test_failed_rewrite_stays_queued_and_raw_stays_searchable():
    llm = FakeLLM(fail=True)
    system = DecoupledMemory(llm, fake_embed)
    system.ingest(msg(1, "I went to the support group yesterday."))
    stats = system.on_idle()
    assert stats["errors"] == 1 and system.store.counts()["queued"] == 1
    assert system.retrieve("support group")[0].tier == "recall"


def test_answer_reports_llm_failure():
    system = PlainRetrieval(FakeLLM(fail=True), fake_embed)
    system.ingest(msg(1, "I went to the support group yesterday."))
    answer = system.answer("support group?")
    assert answer.error == "boom" and answer.text == ""


def test_sources_only_include_what_fit_in_the_prompt():
    system = PlainRetrieval(FakeLLM(), fake_embed)
    for i in range(50):
        system.ingest(msg(i, "word " * 100))
    answer = system.answer("word")
    assert 0 < len(answer.sources) < 30


def test_llm_refuses_non_local_address():
    with pytest.raises(ValueError):
        LocalLLM("https://api.example.com/v1", "m", "unused")


def test_scoring_matches_official_rules():
    assert f1_score("7 May 2023", "7 May 2023") == 1.0
    assert score({"category": 2, "answer": "7 May 2023"}, "the 7 May 2023") == 1.0
    assert score({"category": 5}, "Not mentioned.") == 1.0
    assert score({"category": 5}, "self-care") == 0.0
    assert score({"category": 1, "answer": "cats, dogs"}, "cats") == 0.5
    assert score({"category": 3, "answer": "yes; because"}, "yes") == 1.0
    assert evidence_recall({"evidence": ["D1:1", "D1:2"]}, ["D1:1"]) == 0.5
    assert evidence_recall({"evidence": []}, []) is None


class FakeClassifier:
    """Returns scripted decisions by message text, and embeds nothing itself."""

    def __init__(self, decisions):
        self.decisions = decisions

    def decide(self, speaker, text, active):
        return self.decisions.get(text, MemoryDecision(False, None, False, None)), fake_embed(text)


def trained(llm, decisions):
    return TrainedMemory(llm, fake_embed, FakeClassifier(decisions))


KEEP = MemoryDecision(True, "state", False, None)


def test_secret_is_never_saved_in_any_tier():
    system = trained(FakeLLM(), {"code 123456": MemoryDecision(False, None, True, None)})
    system.ingest(msg(1, "code 123456"))
    assert system.store.counts()["recall"] == 0 and system.store.counts()["queued"] == 0
    assert system.on_idle()["secrets_dropped"] == 1


def test_noise_stays_raw_only_and_makes_no_llm_call():
    llm = FakeLLM()
    system = trained(llm, {})
    system.ingest(msg(1, "lol nice"))
    system.on_idle()
    assert llm.prompts == []
    assert system.store.counts()["recall"] == 1 and system.store.counts()["facts_active"] == 0


def test_update_retires_the_old_fact_and_labels_the_old_message():
    llm = FakeLLM({"Earlier:": "Ana has a dentist visit on Friday.", "Monday": "Ana has a dentist visit on Monday."})
    system = trained(llm, {"dentist Monday": KEEP, "dentist Friday": MemoryDecision(True, "state", False, "D1:1")})
    system.ingest(msg(1, "dentist Monday"))
    system.ingest(msg(2, "dentist Friday"))
    assert system.store.outdated() == {"D1:1"}  # correct straight away, before idle
    stats = system.on_idle()
    assert stats["update"] == 1 and system.store.counts()["facts_active"] == 1
    texts = [hit.text for hit in system.retrieve("when was the dentist visit first set for")]
    assert "Friday" in texts[0]
    assert texts[1].startswith("Out of date:") and "Monday" in texts[1]
    current = [hit.text for hit in system.retrieve("when is the dentist visit")]
    assert not any(text.startswith("Out of date:") for text in current)


def test_update_draft_sees_the_earlier_message():
    llm = FakeLLM({"Earlier:": "Ana has a dentist visit at Elm Clinic on Friday.", "Monday": "Ana has a dentist visit on Monday."})
    system = trained(llm, {"dentist Monday Elm Clinic": KEEP, "dentist Friday": MemoryDecision(True, "state", False, "D1:1")})
    system.ingest(msg(1, "dentist Monday Elm Clinic"))
    system.ingest(msg(2, "dentist Friday"))
    system.on_idle()
    assert llm.prompts[-1] == "Earlier: Ana: dentist Monday Elm Clinic\nNow: Ana: dentist Friday"
    assert "Elm Clinic" in system.retrieve("where is the dentist")[0].text


def test_earlier_version_follows_its_replacement_through_a_chain():
    decisions = {"dentist Monday Elm Clinic": KEEP, "dentist Friday": MemoryDecision(True, "state", False, "D1:1"),
                 "dentist Sunday": MemoryDecision(True, "state", False, "D1:2"), "gym dentist Sunday": KEEP}
    system = trained(FakeLLM(), decisions)
    for number, text in enumerate(decisions, 1):
        system.ingest(msg(number, text))
    hits = system.retrieve("where was the dentist before")
    order = [hit.source for hit in hits]
    index = order.index("D1:3")
    assert order[index + 1:index + 3] == ["D1:2", "D1:1"]
    assert hits[index + 2].text.startswith("Out of date:")


def test_phone_source_facts_are_about_their_owner():
    assert valid_fact("Ravi ordered a desk lamp for $34.", "Ravi's orders")
    assert valid_fact("Ravi's boiler inspection is on 2 April.", "Ravi's calendar")
    assert not valid_fact("The desk lamp cost $34.", "Ravi's orders")


def test_assistant_scoring_accepts_formats_but_not_wrong_values():
    assert contains("12/4/2025, 9:30 am", "12 April 2025")
    assert contains("Sat 12 Apr", "12 April 2025") and contains("April 12th", "12 April 2025")
    assert not contains("13 April", "12 April 2025") and not contains("12/5/2025", "12 April 2025")
    assert contains("$7 per month", "$7") and not contains("$17", "$7") and not contains("$7.50 today x", "$8")
    assert contains("045 57 49 51 49", "04557 495149")
    assert contains("Refund issued", "refunded") and not contains("Left at door", "delivered")
    assert correct({"category": 5, "answer": "Not mentioned"}, "Not mentioned.")
    assert not correct({"category": 5, "answer": "Not mentioned"}, "3 April")
    qa = {"category": 2, "answer": "9 April 2025", "stale_answer": "3 April 2025"}
    assert out_of_date(qa, "Thursday 3 April") and not out_of_date(qa, "3 April, moved to 9 April")


def test_update_works_even_if_the_new_fact_is_rejected():
    llm = FakeLLM({"Earlier:": "I moved it.", "Monday": "Ana has a dentist visit on Monday."})
    system = trained(llm, {"dentist Monday": KEEP, "dentist Friday": MemoryDecision(True, "state", False, "D1:1")})
    system.ingest(msg(1, "dentist Monday"))
    system.ingest(msg(2, "dentist Friday"))
    stats = system.on_idle()
    assert stats["rejected"] == 1 and system.store.counts()["facts_active"] == 0


def test_core_fact_goes_in_core_memory_and_leaves_when_replaced():
    llm = FakeLLM({"Earlier:": "Ana is vegetarian.", "vegan": "Ana is vegan."})
    decisions = {"I am vegan": MemoryDecision(True, "core", False, None),
                 "I am vegetarian": MemoryDecision(True, "core", False, "D1:1")}
    system = trained(llm, decisions)
    system.ingest(msg(1, "I am vegan"))
    system.on_idle()
    assert system.store.core() == ["Ana is vegan."]
    system.ingest(msg(2, "I am vegetarian"))
    system.on_idle()
    assert system.store.core() == ["Ana is vegetarian."]


def test_failed_draft_keeps_the_decision_for_the_next_idle_period():
    system = trained(FakeLLM(fail=True), {"dentist Monday": KEEP})
    system.ingest(msg(1, "dentist Monday"))
    assert system.on_idle()["errors"] == 1
    system.llm = FakeLLM({"Monday": "Ana has a dentist visit on Monday."})
    assert system.on_idle()["add"] == 1


class FakeCloudClient:
    def __init__(self):
        self.questions = []

    def ask(self, question, memories):
        from inferx.cloud import CloudReply
        self.questions.append((question, memories))
        return CloudReply("cloud says hi", 100, 10)


def test_cloud_client_refuses_remote_addresses_unless_allowed():
    with pytest.raises(ValueError):
        CloudClient("https://api.example.com/v1")
    assert CloudClient("https://api.example.com/v1", allow_remote=True).base_url.startswith("https://")


def test_gate_calls_cloud_only_when_memory_is_not_enough():
    client = FakeCloudClient()
    llm = FakeLLM({"Do the memories answer it?": "NO"})
    system = PlainRetrieval(llm, fake_embed)
    system.cloud = CloudPolicy(client, "gate")
    system.ingest(msg(1, "I went to the support group yesterday."))
    answer = system.answer("Where does Ana work?")
    assert answer.cloud_calls == 1 and answer.sufficient is False and answer.cloud_reply == "cloud says hi"
    assert client.questions[0][0] == "Where does Ana work?"
    assert answer.text == "NONE"  # the scored answer stays the local one
    llm.replies = {"Do the memories answer it?": "YES"}
    assert system.answer("support group?").cloud_calls == 0 and len(client.questions) == 1


def test_local_memory_baseline_always_calls_cloud():
    client = FakeCloudClient()
    system = LocalMemory(FakeLLM(), fake_embed)
    with pytest.raises(RuntimeError):
        system.answer("anything?")
    system.cloud = CloudPolicy(client, "always")
    system.ingest(msg(1, "I went to the support group yesterday."))
    assert system.answer("support group?").cloud_calls == 1


def test_memory_work_never_calls_the_cloud():
    client = FakeCloudClient()
    llm = FakeLLM({"support group": "Ana went to a support group on 7 May 2023."})
    system = DecoupledMemory(llm, fake_embed)
    system.cloud = CloudPolicy(client, "always")
    system.ingest(msg(1, "I went to the support group yesterday."))
    system.on_idle()
    assert client.questions == []


def test_full_context_keeps_the_most_recent_messages_that_fit():
    system = FullContext(FakeLLM(), fake_embed, context_tokens=20)
    for i in range(5):
        system.ingest(msg(i, f"note number {i} here"))
    answer = system.answer("which notes?")
    assert answer.sources == ["D1:3", "D1:4"]
