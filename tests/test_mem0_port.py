import pytest

mem0_port = pytest.importorskip("inferx.mem0_port", exc_type=ImportError)


def test_reads_only_first_object_when_model_keeps_writing():
    reply = ('{"facts" : ["Boiler inspection on 29 March"]}  Input: I like tea. '
             'Output: {"facts" : ["Likes tea"]}')
    assert mem0_port._json_object(reply) == {"facts": ["Boiler inspection on 29 March"]}


def test_code_fences_and_leading_words():
    assert mem0_port._json_object('Sure:\n```json\n{"memory": []}\n```') == {"memory": []}


def test_skips_broken_braces_before_the_object():
    assert mem0_port._json_object('{oops} then {"facts": ["a"]}') == {"facts": ["a"]}


def test_no_object():
    assert mem0_port._json_object("no json here") == {}


class FakeMem0:
    """Answers mem0's platform API in memory, so the cloud baseline can be tested without a network."""

    def __init__(self):
        import httpx
        self.memories, self.hosts = [], set()
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request):
        import json
        import httpx
        self.hosts.add(request.url.host)
        path = request.url.path
        body = json.loads(request.content or b"{}")
        if path == "/v1/ping/":
            return httpx.Response(200, json={"org_id": "o", "project_id": "p", "user_email": None})
        if path == "/v3/memories/add/":
            if not body.get("user_id"):  # as mem0 does: add rejects an ID given only in filters
                return httpx.Response(400, json={"error": "At least one entity ID is required"})
            for message in body["messages"]:
                self.memories.append({"id": str(len(self.memories)), "memory": message["content"],
                                      "metadata": body.get("metadata"), "user_id": body["user_id"]})
            return httpx.Response(200, json={"status": "PENDING"})
        if path == "/v3/memories/":
            mine = [m for m in self.memories if m["user_id"] == body["filters"]["user_id"]]
            return httpx.Response(200, json={"count": len(mine), "results": mine})
        if path == "/v3/memories/search/":
            words = set(body["query"].lower().split())
            mine = [m for m in self.memories if m["user_id"] == body["filters"]["user_id"]]
            hits = [dict(m, score=1.0) for m in mine if words & set(m["memory"].lower().split())]
            return httpx.Response(200, json={"results": hits[: body.get("top_k", 10)]})
        return httpx.Response(404, json={"detail": path})


class FakeLLM:
    base_url, model = "http://127.0.0.1:1/v1", "fake"

    def count_tokens(self, text):
        return len(text.split())

    def complete(self, system, prompt, max_tokens=200):
        from types import SimpleNamespace
        return SimpleNamespace(text="Riverside" if "Riverside" in prompt else "Not mentioned",
                               ttft_s=0.0, total_s=0.0, prompt_tokens=1, chunks=1)


def test_mem0_cloud_refuses_without_allow_remote():
    with pytest.raises(ValueError, match="allow-remote"):
        mem0_port.Mem0Cloud(FakeLLM(), None)


def test_mem0_cloud_round_trip_counts_traffic(monkeypatch):
    import httpx
    from inferx.interface import Message
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    fake = FakeMem0()
    system = mem0_port.Mem0Cloud(FakeLLM(), None, allow_remote=True, run_id="u1",
                                 http=httpx.Client(transport=fake.transport), settle_s=0)
    system.ingest(Message("D1:1", "calendar", "Skin check at Riverside Dermatology", "8:00 pm on 17 March, 2025"))
    stats = system.on_idle()
    # the key check when the client starts, then the add
    assert stats["messages"] == 1 and stats["errors"] == 0 and stats["requests"] == 2 and stats["bytes_sent"] > 0
    assert fake.memories[0]["metadata"] == {"source": "D1:1"}
    answer = system.answer("Where is the skin check?")
    assert answer.text == "Riverside" and answer.sources == ["D1:1"]
    assert answer.memory_cloud_requests >= 2 and answer.memory_cloud_bytes > 0
    assert system.on_idle()["requests"] == 0  # question-time traffic is on the answer, not counted again
    assert fake.hosts == {"api.mem0.ai"}


def test_message_timestamp():
    assert mem0_port._message_timestamp("8:00 pm on 17 March, 2025") is not None
    assert mem0_port._message_timestamp("") is None


def test_mem0_cloud_stops_after_repeated_rejections(monkeypatch):
    import httpx
    from inferx.interface import Message
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    fake = FakeMem0()
    system = mem0_port.Mem0Cloud(FakeLLM(), None, allow_remote=True, run_id="u1",
                                 http=httpx.Client(transport=fake.transport), settle_s=0)
    system.filters = {"user_id": ""}  # every add is rejected
    with pytest.raises(RuntimeError, match="3 messages in a row"):
        for index in range(3):
            system.ingest(Message(f"D1:{index}", "calendar", "anything"))
