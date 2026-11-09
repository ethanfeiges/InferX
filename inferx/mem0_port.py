"""On-device port of mem0 (mem0ai 2.2.1, its 2026 additive algorithm).

mem0's own code makes every memory decision, with its own prompts. Only its plumbing is swapped:
  - LLM: phi on the laptop through Foundry Local's OpenAI-style API, on a loopback address only
  - embeddings: the same MiniLM model the other systems use
  - vector store: a small in-memory store, because mem0's default (Qdrant) needs grpcio, which
    has no Windows-on-ARM build
Telemetry is turned off before mem0 is imported, so it sends nothing anywhere.

Runs only in .venv-mem0. Unlike the proposed design, mem0 extracts memories with an LLM call
on every message, while the user is waiting.
"""
from __future__ import annotations

import os
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse

os.environ["MEM0_TELEMETRY"] = "False"


def _stub_qdrant() -> None:
    """mem0 imports Qdrant's client while building its default config, even when it isn't used."""
    try:
        import qdrant_client  # noqa: F401
        return
    except Exception:
        for name in [name for name in sys.modules if name.startswith("qdrant_client")]:
            del sys.modules[name]
    module = types.ModuleType("qdrant_client")

    class QdrantClient:  # never constructed, because the vector store factory is replaced
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Qdrant is not available in this port")

    module.QdrantClient = QdrantClient
    sys.modules["qdrant_client"] = module


_stub_qdrant()

import numpy as np  # noqa: E402
from mem0 import Memory  # noqa: E402
from mem0.utils import factory  # noqa: E402
from mem0.vector_stores.base import VectorStoreBase  # noqa: E402

from .cloud import user_env  # noqa: E402
from .context import Budget, fit  # noqa: E402
from .interface import Answer, MemorySystem, Message  # noqa: E402
from .llm import LOOPBACK, LocalLLM  # noqa: E402
from .store import Hit, MemoryStore  # noqa: E402


@dataclass
class _Point:
    id: str
    payload: dict
    score: float = 0.0
    vector: np.ndarray | None = field(default=None, repr=False)


def _matches(payload: dict, filters: dict | None) -> bool:
    for key, value in (filters or {}).items():
        if isinstance(value, dict):
            if "eq" in value and payload.get(key) != value["eq"]:
                return False
            if "in" in value and payload.get(key) not in value["in"]:
                return False
            if "ne" in value and payload.get(key) == value["ne"]:
                return False
        elif payload.get(key) != value:
            return False
    return True


class InMemoryStore(VectorStoreBase):
    """Cosine search over a dictionary. Enough for one user's memories."""

    def __init__(self, config=None):
        self.points: dict[str, _Point] = {}

    def create_col(self, name, vector_size, distance):
        pass

    def insert(self, vectors, payloads=None, ids=None):
        for index, vector in enumerate(vectors):
            point_id = str(ids[index]) if ids else str(uuid.uuid4())
            array = np.asarray(vector, dtype=np.float32)
            self.points[point_id] = _Point(point_id, dict((payloads or [{}] * len(vectors))[index]),
                                           vector=array / max(float(np.linalg.norm(array)), 1e-9))

    def search(self, query, vectors, top_k=5, filters=None):
        array = np.asarray(vectors, dtype=np.float32)
        array = array / max(float(np.linalg.norm(array)), 1e-9)
        scored = [_Point(p.id, p.payload, float(array @ p.vector)) for p in self.points.values() if _matches(p.payload, filters)]
        return sorted(scored, key=lambda point: -point.score)[:top_k]

    def search_batch(self, queries, vectors_list, top_k=1, filters=None):
        return [self.search(query, vectors, top_k, filters) for query, vectors in zip(queries, vectors_list)]

    def delete(self, vector_id):
        self.points.pop(str(vector_id), None)

    def update(self, vector_id, vector=None, payload=None):
        point = self.points[str(vector_id)]
        if vector is not None:
            array = np.asarray(vector, dtype=np.float32)
            point.vector = array / max(float(np.linalg.norm(array)), 1e-9)
        if payload is not None:
            point.payload = dict(payload)

    def get(self, vector_id):
        point = self.points.get(str(vector_id))
        return _Point(point.id, point.payload) if point else None

    def list_cols(self):
        return ["memories"]

    def delete_col(self):
        self.points.clear()

    def col_info(self):
        return {"points": len(self.points)}

    def list(self, filters=None, top_k=None):
        points = [_Point(p.id, p.payload) for p in self.points.values() if _matches(p.payload, filters)]
        return [points[:top_k] if top_k else points]

    def reset(self):
        self.points.clear()


class _Embedder:
    def __init__(self, embed):
        self.embed_fn = embed

    def embed(self, text, memory_action=None):
        return self.embed_fn(text).tolist()

    def embed_batch(self, texts, memory_action=None):
        return [self.embed(text) for text in texts]


class Mem0Port(MemorySystem):
    """mem0's memory, phi's answers: the answer step is the same as for every other system."""

    name = "mem0_local"

    def __init__(self, llm: LocalLLM, embed, budget: Budget = Budget(), k: int = 30):
        if urlparse(llm.base_url).hostname not in LOOPBACK:
            raise ValueError("mem0 port must use a local LLM")
        self.llm, self.embed, self.budget, self.k = llm, embed, budget, k
        self.store = MemoryStore()  # raw messages, kept only for the run's saved database
        factory.VectorStoreFactory.create = classmethod(lambda cls, provider, config: InMemoryStore())
        factory.EmbedderFactory.create = classmethod(lambda cls, provider, config, vector_config=None: _Embedder(embed))
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": llm.model, "openai_base_url": llm.base_url,
                                                     "api_key": "local", "temperature": 0, "max_tokens": 600}},
            "embedder": {"provider": "openai", "config": {"api_key": "local", "embedding_dims": 384}},
            "vector_store": {"provider": "qdrant", "config": {"embedding_model_dims": 384}},
            "history_db_path": ":memory:",
        })
        self.user = "owner"
        self.stats = {"add_calls": 0, "errors": 0, "memories_added": 0, "add_s": 0.0}

    def ingest(self, message: Message) -> None:
        self.store.add_message(message, np.zeros(384, dtype=np.float32))
        start = time.perf_counter()
        try:
            result = self.memory.add([{"role": "user", "content": message.rendered}], user_id=self.user)
            self.stats["memories_added"] += len(result.get("results", []))
        except Exception:
            self.stats["errors"] += 1
        self.stats["add_calls"] += 1
        self.stats["add_s"] += time.perf_counter() - start

    def on_idle(self) -> dict:
        stats, self.stats = self.stats, {"add_calls": 0, "errors": 0, "memories_added": 0, "add_s": 0.0}
        return stats

    def answer(self, question: str) -> Answer:
        from .systems import _answer_with
        start = time.perf_counter()
        results = self.memory.search(question, top_k=self.k, filters={"user_id": self.user}).get("results", [])
        hits = [Hit("archival", 0, item["memory"], item.get("id", ""), float(item.get("score") or 0.0)) for item in results]
        return _answer_with(self.llm, question, [], hits, [], self.budget, time.perf_counter() - start, self.cloud)


def _json_object(text: str) -> dict:
    """Read the first JSON object in a reply, tolerating code fences and extra words.

    phi often keeps writing after its answer, inventing further "Input: ... Output: {...}" examples,
    so only the first complete object is read.
    """
    import json
    import re
    cleaned = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder(strict=False)
    start = cleaned.find("{")
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(cleaned, start)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        start = cleaned.find("{", start + 1)
    return {}


class Mem0V1Port(MemorySystem):
    """Port of mem0's 2025 algorithm, using the prompts that ship with mem0.

    For each message: one LLM call extracts facts (FACT_RETRIEVAL_PROMPT), then one LLM call compares
    them with the 5 most similar stored memories per fact and chooses ADD, UPDATE, DELETE or NONE
    (get_update_memory_messages). mem0's 2026 algorithm can't run on this model: its extraction
    prompt is about 9,500 tokens, beyond what the NPU build of phi accepts.
    """

    name = "mem0_2025_local"

    def __init__(self, llm: LocalLLM, embed, budget: Budget = Budget(), k: int = 30):
        from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT, get_update_memory_messages
        if urlparse(llm.base_url).hostname not in LOOPBACK:
            raise ValueError("mem0 port must use a local LLM")
        self.llm, self.embed, self.budget, self.k = llm, embed, budget, k
        self.fact_prompt, self.update_messages = FACT_RETRIEVAL_PROMPT, get_update_memory_messages
        self.store = MemoryStore()  # raw messages, kept only for the run's saved database
        self.vectors = InMemoryStore()
        self.stats = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return {"messages": 0, "llm_calls": 0, "errors": 0, "add": 0, "update": 0, "delete": 0, "none": 0, "llm_s": 0.0}

    def _ask(self, system: str, user: str, max_tokens: int) -> dict:
        from .llm import LLMError
        try:
            reply = self.llm.complete(system, user, max_tokens=max_tokens)
        except LLMError:
            self.stats["errors"] += 1
            return {}
        self.stats["llm_calls"] += 1
        self.stats["llm_s"] += reply.total_s
        return _json_object(reply.text)

    def ingest(self, message: Message) -> None:
        self.store.add_message(message, np.zeros(384, dtype=np.float32))
        self.stats["messages"] += 1
        facts = [fact for fact in self._ask(self.fact_prompt, f"Input:\nuser: {message.rendered}", 300).get("facts", [])
                 if isinstance(fact, str) and fact.strip()]
        if not facts:
            return
        old: dict[str, str] = {}
        for fact in facts:
            for point in self.vectors.search(fact, self.embed(fact), top_k=5):
                old[point.id] = point.payload["data"]
        temp_ids = {str(index): point_id for index, point_id in enumerate(old)}
        retrieved = [{"id": str(index), "text": text} for index, text in enumerate(old.values())]
        plan = self._ask("You manage a memory store. Reply in JSON only.", self.update_messages(retrieved, facts), 600)
        for action in plan.get("memory", []) if isinstance(plan.get("memory"), list) else []:
            if not isinstance(action, dict):
                continue
            event, text = str(action.get("event", "")).upper(), str(action.get("text") or "").strip()
            target = temp_ids.get(str(action.get("id")))
            if event == "ADD" and text:
                self.vectors.insert([self.embed(text)], [{"data": text, "source": message.id}])
                self.stats["add"] += 1
            elif event == "UPDATE" and target and text:
                self.vectors.update(target, self.embed(text), {"data": text, "source": message.id})
                self.stats["update"] += 1
            elif event == "DELETE" and target:
                self.vectors.delete(target)
                self.stats["delete"] += 1
            else:
                self.stats["none"] += 1

    def on_idle(self) -> dict:
        stats, self.stats = self.stats, self._fresh()
        return stats

    def answer(self, question: str) -> Answer:
        from .systems import _answer_with
        start = time.perf_counter()
        points = self.vectors.search(question, self.embed(question), top_k=self.k)
        hits = [Hit("archival", 0, point.payload["data"], point.payload.get("source", ""), point.score) for point in points]
        return _answer_with(self.llm, question, [], hits, [], self.budget, time.perf_counter() - start, self.cloud)


def _message_timestamp(date: str) -> int | None:
    from datetime import datetime
    for layout in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try:
            return int(datetime.strptime(date, layout).timestamp())
        except ValueError:
            pass
    return None


class Mem0Cloud(MemorySystem):
    """mem0's hosted platform: every memory decision is made in mem0's cloud, as mem0 ships it.

    Each message goes to mem0 when it arrives; questions search mem0, and phi answers from the
    results with the same answer step as every other system. Every request to mem0 is counted,
    with its size, as memory-work cloud traffic. Refuses to start unless remote calls are allowed.
    """

    name = "mem0_cloud"

    def __init__(self, llm: LocalLLM, embed, budget: Budget = Budget(), k: int = 30, *,
                 allow_remote: bool = False, run_id: str | None = None, http=None,
                 settle_s: float = 15.0, settle_max_s: float = 900.0):
        if not allow_remote:
            raise ValueError("mem0_cloud sends every message to mem0's servers; pass --allow-remote")
        import httpx
        from mem0.client.main import MemoryClient
        self.llm, self.embed, self.budget, self.k = llm, embed, budget, k
        self.store = MemoryStore()  # raw messages, kept only for the run's saved database
        self.traffic = {"requests": 0, "bytes_sent": 0, "bytes_received": 0}

        def sent(request):
            self.traffic["requests"] += 1
            self.traffic["bytes_sent"] += len(request.content or b"")

        def received(response):
            response.read()
            self.traffic["bytes_received"] += len(response.content or b"")

        http = http or httpx.Client(timeout=300)
        http.event_hooks = {"request": [sent], "response": [received]}
        self.client = MemoryClient(api_key=user_env("MEM0_API_KEY"), client=http)
        self.filters = {"user_id": run_id or f"inferx-{uuid.uuid4().hex[:12]}"}
        self.settle_s, self.settle_max_s, self.settled = settle_s, settle_max_s, False
        self.failures_in_a_row = 0
        self.stats = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return {"messages": 0, "errors": 0, "add_s": 0.0}

    def ingest(self, message: Message) -> None:
        self.store.add_message(message, np.zeros(384, dtype=np.float32))
        self.stats["messages"] += 1
        self.settled = False
        start = time.perf_counter()
        try:
            # mem0's add needs the user ID as a top-level field (its SDK docs say filters, which mem0 rejects);
            # search and get_all take it in filters.
            self.client.add([{"role": "user", "content": message.rendered}], user_id=self.filters["user_id"],
                            timestamp=_message_timestamp(message.date), metadata={"source": message.id})
            self.failures_in_a_row = 0
        except Exception as error:
            self.stats["errors"] += 1
            self.failures_in_a_row += 1
            if self.failures_in_a_row >= 3:
                raise RuntimeError(f"mem0 rejected 3 messages in a row: {error}") from error
        self.stats["add_s"] += time.perf_counter() - start

    def on_idle(self) -> dict:
        # the run script adds these up across sessions, so report traffic since the last call
        stats, self.stats = {**self.stats, **self.traffic}, self._fresh()
        self.traffic = {key: 0 for key in self.traffic}
        return stats

    def _count(self) -> int:
        result = self.client.get_all(filters=self.filters)
        if isinstance(result, dict):
            return int(result.get("count") or len(result.get("results") or []))
        return len(result or [])

    def settle(self) -> None:
        """mem0 processes added messages in the background; wait until its memory count stops changing."""
        if self.settled:
            return
        deadline, last, steady = time.monotonic() + self.settle_max_s, -1, 0
        while time.monotonic() < deadline and steady < 3:
            count = self._count()
            steady, last = (steady + 1, last) if count == last else (0, count)
            if steady < 3:
                time.sleep(self.settle_s)
        self.settled = True

    def answer(self, question: str) -> Answer:
        from .systems import _answer_with
        before = dict(self.traffic)
        self.settle()
        start = time.perf_counter()
        results = self.client.search(question, filters=self.filters, top_k=self.k).get("results", [])
        retrieval_s = time.perf_counter() - start
        hits = [Hit("archival", 0, item["memory"], str((item.get("metadata") or {}).get("source", "")),
                    float(item.get("score") or 0.0)) for item in results if item.get("memory")]
        answer = _answer_with(self.llm, question, [], hits, [], self.budget, retrieval_s, self.cloud)
        answer.memory_cloud_requests = self.traffic["requests"] - before["requests"]
        answer.memory_cloud_bytes = (self.traffic["bytes_sent"] - before["bytes_sent"]
                                     + self.traffic["bytes_received"] - before["bytes_received"])
        self.traffic = before  # recorded on the answer, so on_idle doesn't count it again
        return answer
