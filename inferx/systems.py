"""The proposed design and the plain-retrieval baseline."""
from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Callable

import numpy as np

from .context import ANSWER_SYSTEM, Budget, build_prompt, fit
from .controller import Decision, Thresholds, core_facts, decide, may_replace, valid_fact, worth_remembering
from .dates import annotate, parse_date, remove_unsupported_dates
from .interface import Answer, MemorySystem, Message
from .llm import LLMError, LocalLLM
from .memory_classifier import Item, MemoryClassifier, different_item, different_owner
from .scheduler import Scheduler
from .sensitive import secret_kind
from .store import Hit, MemoryStore

REWRITE_SYSTEM = (
    "Rewrite the message as one short standalone fact about the speaker, starting with the "
    "speaker's name. Keep names, places and details. Use only what the message says. "
    "If a date is in parentheses, use that date, and never add any other date. "
    "Keep vague times such as 'a few years ago' as written. "
    "Keep plans and wishes as plans, not as things that happened. "
    "Never use I, you, my or your. Reply with only the fact. "
    "If there is no lasting fact, reply exactly: NONE\n"
    "Examples:\n"
    "Ana: I went to a pottery class yesterday (7 May 2023) in Rome!\n"
    "Ana went to a pottery class in Rome on 7 May 2023.\n"
    "Ben: My sister just adopted two cats.\n"
    "Ben's sister adopted two cats.\n"
    "Ana: Got this scarf a few years ago, it reminds me of Lisbon.\n"
    "Ana got a scarf a few years ago that reminds her of Lisbon.\n"
    "Ben: I'll come to your show next week (the week after 3 June 2023)!\n"
    "Ben plans to come to a show the week after 3 June 2023.\n"
    "Ben: Haha, that sounds so fun, you should do it!\n"
    "NONE"
)
MERGE_SYSTEM = (
    "Write one short standalone fact giving the current situation, starting with the person's name. "
    "Use the new message for anything that changed, and keep details from the earlier message that "
    "did not change, such as places and names. If a date is in parentheses, use that date, and never "
    "add any other date. Never use I, you, my or your. Reply with only the fact.\n"
    "Examples:\n"
    "Earlier: Ana's calendar: Dentist on 4 May at 10:00, Elm Clinic.\n"
    "Now: Ana's calendar: Dentist moved to 9 May.\n"
    "Ana's dentist appointment at Elm Clinic is on 9 May at 10:00.\n"
    "Earlier: Ben: We live at 3 Hill Road.\n"
    "Now: Ben: Moved again, we're at 8 Park Lane now.\n"
    "Ben lives at 8 Park Lane."
)
HISTORY = re.compile(r"\b(first|originally|original|initially|before|previous|previously|used to|earlier|at first|old)\b",
                     re.I)
JUDGE_SYSTEM = (
    "Compare the new fact with the stored fact. Reply with one word. "
    "IGNORE if they say the same thing, UPDATE if the new fact replaces the stored fact, "
    "ADD if both should be kept."
)
# Only asked when the controller is unsure whether a new message replaces a stored one.
REPLACES_SYSTEM = (
    "Does the new message change or cancel what the earlier message said, so that the earlier message is "
    "no longer true? Two different things of the same type (two flights, two subscriptions) do not "
    "cancel each other. Reply with one word: YES or NO.\n"
    "Examples:\n"
    "Earlier: Sam: I teach at Elm School.\nNew: Sam: Started a new job at Northwind Labs!\nYES\n"
    "Earlier: Sam's calendar: Dentist on 4 May at 10:00.\nNew: Sam's calendar: Dentist moved to 9 May.\nYES\n"
    "Earlier: Sam's orders: StreamBox is $9 a month.\nNew: Sam's orders: Welcome to GymPlus, $30 a month.\nNO\n"
    "Earlier: Sam's calendar: Flight BA117 to Oslo, 3 July.\nNew: Sam's calendar: Flight KL902 to Riga, 8 July.\nNO"
)
ENTITY = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d{1,2} [A-Z][a-z]+ \d{4}|\d{4})\b")
STOP = {
    "The", "I", "We", "My", "It", "He", "She", "They", "That", "This", "What", "When",
    "Hey", "Hi", "Thanks", "Wow", "Yes", "No", "Oh", "And", "But", "So",
}


def entities(texts: list[str], limit: int = 12) -> list[str]:
    seen: dict[str, None] = {}
    for text in texts:
        for match in ENTITY.findall(text):
            words = match.split()
            while words and words[0] in STOP:
                words.pop(0)
            if words:
                seen.setdefault(" ".join(words))
    return list(seen)[:limit]


def _answer_with(llm: LocalLLM, question: str, core: list[str], hits: list[Hit], recent: list[str],
                 budget: Budget, retrieval_s: float, cloud=None) -> Answer:
    retrieved = fit([hit.text for hit in hits], budget.retrieved, llm.count_tokens)
    used = hits[: len(retrieved)]
    prompt = build_prompt(question, core, retrieved, recent)
    answer = Answer(
        text="",
        retrieval_s=retrieval_s,
        top_score=max((hit.score for hit in hits), default=None),
        sources=[hit.source for hit in used],
    )
    try:
        reply = llm.complete(ANSWER_SYSTEM, prompt, max_tokens=min(budget.reply, 200))
    except LLMError as error:
        answer.error = str(error)
        return answer
    answer.text = reply.text
    answer.ttft_s = reply.ttft_s
    answer.total_s = reply.total_s
    answer.prompt_tokens = reply.prompt_tokens
    answer.output_chunks = reply.chunks
    if cloud is not None:
        _maybe_cloud(llm, cloud, question, core + retrieved, prompt, answer)
    return answer


def _maybe_cloud(llm: LocalLLM, cloud, question: str, memories: list[str], prompt: str, answer: Answer) -> None:
    """Apply the cloud rule. The local reply is kept as the scored answer, and the cloud reply is recorded
    beside it, so local accuracy and cloud use can be measured separately."""
    if cloud.mode == "gate":
        answer.sufficient = cloud.sufficient(llm, prompt)
        if answer.sufficient:
            return
    try:
        reply = cloud.client.ask(question, memories)
    except LLMError as error:
        answer.cloud_reply = f"error: {error}"
        answer.cloud_calls = 1
        return
    answer.cloud_calls = 1
    answer.cloud_bytes = reply.bytes_sent
    answer.cloud_reply = reply.text


class PlainRetrieval(MemorySystem):
    """Baseline: stores every message and does no memory management."""

    name = "plain_retrieval"

    def __init__(self, llm: LocalLLM, embed: Callable[[str], np.ndarray], budget: Budget = Budget(), k: int = 30):
        self.llm, self.embed, self.budget, self.k = llm, embed, budget, k
        self.store = MemoryStore()

    def ingest(self, message: Message) -> None:
        self.store.add_message(message, self.embed(message.rendered))

    def on_idle(self) -> dict:
        return {}

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.store.search_recall(self.embed(question), self.k)
        return _answer_with(self.llm, question, [], hits, [], self.budget, time.perf_counter() - start, self.cloud)


class LocalMemory(PlainRetrieval):
    """Baseline from the proposal: memory stays on the device, but every question goes to the cloud."""

    name = "local_memory"

    def answer(self, question: str) -> Answer:
        if self.cloud is None or self.cloud.mode != "always":
            raise RuntimeError("local_memory needs a cloud policy in 'always' mode")
        return super().answer(question)


class FullContext(MemorySystem):
    """Baseline: no memory store. The most recent messages that fit the window go in every prompt."""

    name = "full_context"

    def __init__(self, llm: LocalLLM, embed: Callable[[str], np.ndarray], budget: Budget = Budget(),
                 context_tokens: int = 3000):
        self.llm, self.embed, self.budget, self.context_tokens = llm, embed, budget, context_tokens
        self.store = MemoryStore()
        self.messages: list[Message] = []

    def ingest(self, message: Message) -> None:
        self.messages.append(message)
        self.store.add_message(message, np.zeros(384, dtype=np.float32))

    def on_idle(self) -> dict:
        return {}

    def answer(self, question: str) -> Answer:
        kept: list[Message] = []
        used = 0
        for message in reversed(self.messages):
            cost = self.llm.count_tokens(message.rendered) + 1
            if used + cost > self.context_tokens:
                break
            kept.append(message)
            used += cost
        kept.reverse()
        hits = [Hit("recall", 0, message.rendered, message.id, 0.0) for message in kept]
        budget = Budget(retrieved=self.context_tokens)
        return _answer_with(self.llm, question, [], hits, [], budget, 0.0, self.cloud)


class DecoupledMemory(MemorySystem):
    """Proposed design: the main LLM reads memory, and only the controller writes it."""

    name = "decoupled"

    def __init__(self, llm: LocalLLM, embed: Callable[[str], np.ndarray], budget: Budget = Budget(),
                 k: int = 15, thresholds: Thresholds = Thresholds(), scheduler: Scheduler | None = None):
        self.llm, self.embed, self.budget, self.k = llm, embed, budget, k
        self.thresholds = thresholds
        self.scheduler = scheduler or Scheduler()
        self.store = MemoryStore()

    def ingest(self, message: Message) -> None:
        rowid = self.store.add_message(message, self.embed(message.rendered))
        for key, fact in core_facts(message):
            self.store.set_core(key, fact, message.id)
        if worth_remembering(message):
            self.store.enqueue(rowid)

    def on_idle(self) -> dict:
        return self.scheduler.run(self._process_queue)

    def _process_queue(self) -> dict:
        stats = {"processed": 0, "add": 0, "update": 0, "ignore": 0, "none": 0, "rejected": 0,
                 "ambiguous": 0, "llm_calls": 0, "errors": 0, "llm_s": 0.0, "dates_removed": 0}
        for queue_rowid, message in self.store.queued():
            # The LLM never sees the message date: rules resolve relative times, and the
            # store shows the date each fact was said beside it.
            text = annotate(message.text, parse_date(message.date))
            try:
                reply = self.llm.complete(REWRITE_SYSTEM, f"{message.speaker}: {text}", max_tokens=100)
            except LLMError:
                stats["errors"] += 1
                continue  # stays queued, and the raw message remains searchable
            stats["llm_calls"] += 1
            stats["llm_s"] += reply.total_s
            stats["processed"] += 1
            self.store.dequeue(queue_rowid)
            fact = reply.text.splitlines()[0].strip()
            if not fact or fact.upper().startswith("NONE"):
                stats["none"] += 1
                continue
            fact, removed = remove_unsupported_dates(fact, text)
            stats["dates_removed"] += removed
            if not valid_fact(fact, message.speaker):
                stats["rejected"] += 1
                continue
            self._write_fact(fact, message, stats)
        return stats

    def _write_fact(self, fact: str, message: Message, stats: dict) -> None:
        vector = self.embed(fact)
        nearest = self.store.search_facts(vector, 1)
        hit = nearest[0] if nearest else None
        decision = decide(fact, hit.text if hit else None, hit.score if hit else None, self.thresholds)
        if decision is Decision.AMBIGUOUS:
            stats["ambiguous"] += 1
            decision = self._judge(fact, hit.text, stats)
        stats[decision.value] += 1
        if decision is Decision.IGNORE:
            return
        rowid = self.store.add_fact(fact, message.id, message.date, vector)
        if decision is Decision.UPDATE and hit:
            self.store.supersede(hit.rowid, rowid)

    def _judge(self, fact: str, stored: str, stats: dict) -> Decision:
        """Rare ambiguous cases go to the main LLM, and never the cloud."""
        try:
            reply = self.llm.complete(JUDGE_SYSTEM, f"Stored fact: {stored}\nNew fact: {fact}\nWord:", max_tokens=64)
        except LLMError:
            stats["errors"] += 1
            return Decision.ADD  # keep the information when the judge fails
        stats["llm_calls"] += 1
        stats["llm_s"] += reply.total_s
        word = reply.text.upper()
        if "IGNORE" in word:
            return Decision.IGNORE
        return Decision.UPDATE if "UPDATE" in word and may_replace(fact, stored) else Decision.ADD

    def retrieve(self, question: str) -> list[Hit]:
        """Two searches with no LLM call: the question, then entities from the top results."""
        vector = self.embed(question)
        first = self._search(vector)
        names = entities([question] + [hit.text for hit in first[:5]])
        second = self._search(self.embed(" ".join(names))) if names else []
        # A fact replaces the raw message it came from, as it is shorter and standalone.
        facts = {hit.source: hit for hit in first + second if hit.tier == "archival"}
        ranked: dict[str, Hit] = {}
        # Entity queries score high on names alone, so question matches always rank first.
        for group in (first, second):
            for hit in sorted(group, key=lambda h: -h.score):
                ranked.setdefault(hit.source, facts.get(hit.source, hit))
        return list(ranked.values())

    def _search(self, vector: np.ndarray) -> list[Hit]:
        return self.store.search_facts(vector, self.k) + self.store.search_recall(vector, self.k)

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.retrieve(question)
        core = fit(self.store.core(), self.budget.core, self.llm.count_tokens)
        return _answer_with(self.llm, question, core, hits, [], self.budget, time.perf_counter() - start, self.cloud)


class TrainedMemory(DecoupledMemory):
    """Proposed design with the trained controller: small models make every memory decision.

    The main LLM only drafts the wording of facts at idle time and answers questions.
    """

    name = "trained"

    def __init__(self, llm: LocalLLM, embed: Callable[[str], np.ndarray], classifier: MemoryClassifier,
                 budget: Budget = Budget(), k: int = 15, scheduler: Scheduler | None = None, *,
                 secret_rules: bool = False, owner_guard: bool = False, ask_when_unsure: bool = False,
                 replace_band: float = 0.2, store_band: float = 0.1):
        """The keyword options are off by default, so earlier runs stay reproducible.

        secret_rules: also drop messages whose format marks them as secret (inferx.sensitive).
        owner_guard: a first-person message can't replace a fact someone else stated.
        ask_when_unsure: phi decides at idle time when an update score is within replace_band of the
            threshold, or when the controller would replace a memory that names something different.
            Messages scored within store_band below the store threshold are queued too, and are kept only
            if phi writes a fact for them.
        """
        super().__init__(llm, embed, budget, k=k, scheduler=scheduler)
        self.classifier = classifier
        if owner_guard and different_owner not in classifier.guards:
            classifier.guards = classifier.guards + (different_owner,)
        self.secret_rules, self.ask_when_unsure = secret_rules, ask_when_unsure
        self.replace_band, self.store_band = replace_band, store_band
        self.active: list[Item] = []
        self.pending: dict = {}
        self.asks: dict[str, str] = {}  # new message id -> stored message phi is asked about
        self.unsure_store: set[str] = set()
        self.dropped = self.dropped_by_rule = 0

    def ingest(self, message: Message) -> None:
        if self.ask_when_unsure:
            threshold = max(self.classifier.store_threshold - self.store_band, 0.0)
            decision, vector = self.classifier.decide(message.speaker, message.text, self.active, store_threshold=threshold)
        else:
            decision, vector = self.classifier.decide(message.speaker, message.text, self.active)
        if decision.sensitive or (self.secret_rules and secret_kind(message.text)):
            self.dropped += 1  # a secret is never saved, not even as a raw message
            self.dropped_by_rule += not decision.sensitive
            return
        rowid = self.store.add_message(message, self.embed(message.rendered))
        if not decision.store:
            return
        unsure_store = (self.ask_when_unsure and decision.store_score is not None
                        and decision.store_score < self.classifier.store_threshold)
        if unsure_store:
            self.unsure_store.add(message.id)
            decision = replace(decision, replaces=None)  # an unsure message never replaces anything
        elif self.ask_when_unsure and decision.candidate:
            near = abs(decision.replace_score - self.classifier.replace_threshold) < self.replace_band
            old = next((item for item in self.active if item.id == decision.candidate), None)
            conflict = bool(decision.replaces and old and different_item(message.speaker, message.text, old))
            if near or conflict:
                self.asks[message.id] = decision.candidate
                decision = replace(decision, replaces=None)  # decided at idle time instead
        if decision.replaces:
            self.store.mark_outdated(decision.replaces, message.id)
            self.active = [item for item in self.active if item.id != decision.replaces]
        self.active.append(Item(message.id, message.speaker, message.text, vector, decision.kind))
        self.pending[message.id] = decision
        self.store.enqueue(rowid)

    def _llm_says_replaces(self, old_id: str, message: Message, stats: dict) -> bool:
        old = self.store.message(old_id)
        if old is None:
            return False
        prompt = f"Earlier: {old.speaker}: {old.text}\nNew: {message.speaker}: {message.text}\nAnswer:"
        try:
            reply = self.llm.complete(REPLACES_SYSTEM, prompt, max_tokens=8)
        except LLMError:
            stats["errors"] += 1
            return False  # keep both: the raw messages stay searchable
        stats["llm_calls"] += 1
        stats["llm_s"] += reply.total_s
        words = reply.text.strip().upper().split()
        return bool(words) and words[0].strip(".,!:") == "YES"

    def _process_queue(self) -> dict:
        stats = {"processed": 0, "add": 0, "update": 0, "none": 0, "rejected": 0, "core": 0, "llm_calls": 0, "errors": 0,
                 "llm_s": 0.0, "dates_removed": 0, "secrets_dropped": self.dropped, "secrets_by_rule": self.dropped_by_rule,
                 "asked_replaces": 0, "llm_said_replaces": 0, "unsure_store": 0, "unsure_store_kept": 0}
        self.dropped = self.dropped_by_rule = 0
        for queue_rowid, message in self.store.queued():
            decision = self.pending[message.id]
            candidate = self.asks.pop(message.id, None)
            if candidate and any(item.id == candidate for item in self.active):
                stats["asked_replaces"] += 1
                if self._llm_says_replaces(candidate, message, stats):
                    stats["llm_said_replaces"] += 1
                    self.store.mark_outdated(candidate, message.id)
                    self.active = [item for item in self.active if item.id != candidate]
                    decision = self.pending[message.id] = replace(decision, replaces=candidate)
            unsure = message.id in self.unsure_store
            text = annotate(message.text, parse_date(message.date))
            system, prompt = REWRITE_SYSTEM, f"{message.speaker}: {text}"
            old = self.store.message(decision.replaces) if decision.replaces else None
            if old:
                # The new fact carries forward what the update didn't repeat, such as the place.
                old_text = annotate(old.text, parse_date(old.date))
                system, prompt = MERGE_SYSTEM, f"Earlier: {old.speaker}: {old_text}\nNow: {prompt}"
                text = f"{old_text} {text}"
            try:
                reply = self.llm.complete(system, prompt, max_tokens=100)
            except LLMError:
                stats["errors"] += 1
                continue  # stays queued, and the raw message remains searchable
            stats["llm_calls"] += 1
            stats["llm_s"] += reply.total_s
            stats["processed"] += 1
            self.store.dequeue(queue_rowid)
            self.pending.pop(message.id)
            fact = reply.text.splitlines()[0].strip() if reply.text.strip() else ""
            fact, removed = remove_unsupported_dates(fact, text)
            stats["dates_removed"] += removed
            new_rowid = None
            if fact and not fact.upper().startswith("NONE") and valid_fact(fact, message.speaker):
                new_rowid = self.store.add_fact(fact, message.id, message.date, self.embed(fact))
            else:
                stats["rejected"] += 1
            if unsure:
                # The controller wasn't sure this was worth keeping, so it stays only if phi found a fact in it.
                self.unsure_store.discard(message.id)
                stats["unsure_store"] += 1
                if not new_rowid:
                    self.active = [item for item in self.active if item.id != message.id]
                    continue
                stats["unsure_store_kept"] += 1
            if decision.kind == "core":
                self.store.set_core(message.id, fact if new_rowid else f"{message.speaker}: {message.text}", message.id)
                stats["core"] += 1
            stats["add"] += 1
            if decision.replaces:
                self.store.retire_facts(decision.replaces, new_rowid)
                self.store.remove_core(decision.replaces)
                stats["update"] += 1
        return stats

    def retrieve(self, question: str) -> list[Hit]:
        """Current memories first. Replaced ones appear, labelled, only when they can help.

        They follow their replacement for history questions, and when the replacement has no merged
        fact holding the details it didn't repeat. The label keeps them from being taken as current.
        """
        outdated = self.store.outdated()
        hits = super().retrieve(question)
        history = bool(HISTORY.search(question))
        ordered, seen = [], set()

        def add(hit: Hit, old: bool) -> None:
            if hit.source not in seen:
                seen.add(hit.source)
                ordered.append(replace(hit, text=f"Out of date: {hit.text}") if old else hit)

        for hit in hits:
            if hit.source not in outdated:
                add(hit, False)
                # A merged fact already carries the unchanged details, so older versions only help
                # history questions, or a current message whose fact could not be written.
                if history or not self.store.has_fact(hit.source):
                    for earlier in self.store.earlier_versions(hit.source):
                        add(earlier, True)
        if history:
            for hit in hits:
                if hit.source in outdated:
                    add(hit, True)
        return ordered


class TrainedPlus(TrainedMemory):
    """The trained controller with the additions meant to generalize beyond the training data:
    secret format rules, the owner guard, and phi deciding the cases the controller is unsure about."""

    name = "trained_plus"
    OPTIONS = ("secret_rules", "owner_guard", "ask_when_unsure")

    def __init__(self, llm: LocalLLM, embed: Callable[[str], np.ndarray], classifier: MemoryClassifier,
                 budget: Budget = Budget(), options: tuple[str, ...] = OPTIONS, **kwargs):
        unknown = set(options) - set(self.OPTIONS)
        if unknown:
            raise ValueError(f"unknown trained_plus options: {sorted(unknown)}")
        super().__init__(llm, embed, classifier, budget, **{option: True for option in options}, **kwargs)
