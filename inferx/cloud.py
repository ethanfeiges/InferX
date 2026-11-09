"""The cloud rule and the cloud client.

A question goes to the cloud only when the local memories are not enough to answer it. The same
rule applies to every system, so the comparison is fair. Memory work never calls the cloud.

Until a real provider is configured, the client only talks to a local stand-in
(scripts/cloud_stub.py) that counts calls and bytes. A remote address must be allowed
explicitly, so no request can leave the device by accident.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from .llm import LOOPBACK, LLMError

SUFFICIENT_SYSTEM = (
    "Decide whether the memories contain the answer to the question. "
    "Reply with one word: YES if they do, NO if they do not."
)
CLOUD_SYSTEM = "Answer the question using the memories provided. Reply with a short phrase."


@dataclass
class CloudReply:
    text: str
    bytes_sent: int
    bytes_received: int


class CloudClient:
    """OpenAI-style chat client. Counts every call and byte it sends."""

    def __init__(self, base_url: str, model: str = "cloud", api_key: str | None = None, allow_remote: bool = False,
                 timeout_s: float = 60):
        if urlparse(base_url).hostname not in LOOPBACK and not allow_remote:
            raise ValueError(f"refusing remote cloud address without allow_remote: {base_url}")
        self.base_url, self.model, self.api_key, self.timeout_s = base_url.rstrip("/"), model, api_key, timeout_s
        self.calls = 0
        self.bytes_sent = 0

    def ask(self, question: str, memories: list[str]) -> CloudReply:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": CLOUD_SYSTEM},
                {"role": "user", "content": "Memories:\n" + "\n".join(memories) + f"\n\nQuestion: {question} Answer:"},
            ],
            "temperature": 0,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        self.calls += 1
        self.bytes_sent += len(body)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except Exception as error:
            raise LLMError(f"cloud call failed: {type(error).__name__}: {error}") from error
        text = json.loads(raw)["choices"][0]["message"]["content"]
        return CloudReply(text, len(body), len(raw))


class CloudPolicy:
    """When a question goes to the cloud.

    gate:   only when the main LLM says the memories don't answer the question
    always: every question, as in the local-memory baseline
    """

    def __init__(self, client: CloudClient, mode: str = "gate"):
        if mode not in ("gate", "always"):
            raise ValueError(f"unknown cloud mode: {mode}")
        self.client, self.mode = client, mode

    def sufficient(self, llm, prompt: str) -> bool | None:
        """The main LLM's yes or no. None when the check itself fails, which counts as not enough."""
        try:
            reply = llm.complete(SUFFICIENT_SYSTEM, prompt + "\nDo the memories answer it?", max_tokens=8)
        except LLMError:
            return None
        return reply.text.strip().upper().startswith("YES")


def user_env(name: str) -> str | None:
    """An environment variable, also read from the user's saved Windows settings.

    A variable saved with setx or the Windows settings window only reaches programs started
    afterwards, so this reads it from the registry too. The value is never printed or logged.
    """
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, name)[0]) or None
    except OSError:
        return None

