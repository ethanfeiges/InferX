"""Run InferX's models on a Galaxy phone over adb: phi-3.5 through Qualcomm Genie, and the memory
controller (MiniLM and its 4 decision heads) through qnn-net-run, both on the phone's NPU (HTP).

The laptop sends inputs over USB and reads results back. The phone does the model compute; the
laptop does only tokenization, mean pooling of MiniLM's output, and the controller's rules.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

ADB = os.environ.get("INFERX_ADB") or shutil.which("adb") or str(
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe")
PHONE = "/data/local/tmp"
RUNTIME = f"{PHONE}/qairt252"
ENV = (f"export LD_LIBRARY_PATH={RUNTIME}/lib; "
       f"export ADSP_LIBRARY_PATH='{RUNTIME}/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp'; ")


def adb(*args: str, check: bool = True, timeout: float = 600) -> str:
    result = subprocess.run([ADB, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args[:2])} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def shell(command: str, timeout: float = 600) -> str:
    return adb("shell", ENV + command, timeout=timeout)


class PhoneNet:
    """One DLC on the phone's NPU. run() takes a list of input dicts and returns one output array each."""

    def __init__(self, dlc: str, inputs: dict[str, type], output: str, output_shape: tuple):
        """inputs maps each input name to the numpy dtype the DLC expects (np.int32 or np.float32)."""
        self.dlc, self.inputs, self.output, self.output_shape = dlc, inputs, output, output_shape

    def run(self, batch: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
        job = f"{PHONE}/jobs/{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lines = []
            for i, feed in enumerate(batch):
                parts = []
                for name, dtype in self.inputs.items():
                    (tmp / f"{i}_{name}.raw").write_bytes(np.asarray(feed[name], dtype=dtype).tobytes())
                    parts.append(f"{name}:={job}/{i}_{name}.raw")
                lines.append(" ".join(parts))
            (tmp / "inputs.txt").write_text("\n".join(lines) + "\n", newline="\n")
            shell(f"mkdir -p {job}")
            adb("push", f"{tmp}{os.sep}.", job + "/")
            shell(f"cd {job} && {RUNTIME}/bin/qnn-net-run --backend {RUNTIME}/lib/libQnnHtp.so "
                  f"--model {RUNTIME}/lib/libQnnModelDlc.so --dlc_path {PHONE}/controller/{self.dlc} "
                  f"--input_list inputs.txt --use_native_input_files --output_dir out > log.txt 2>&1 || (cat log.txt; exit 1)")
            adb("pull", f"{job}/out", str(tmp))
            shell(f"rm -rf {job}")
            results = []
            for i in range(len(batch)):
                found = list((tmp / "out" / f"Result_{i}").rglob("*.raw"))
                if len(found) != 1:
                    raise RuntimeError(f"expected one output for input {i}, found {[p.name for p in found]}")
                results.append(np.fromfile(found[0], dtype=np.float32).reshape(self.output_shape))
            return results


# --- phi-3.5 through Qualcomm Genie -----------------------------------------------------------------

BUNDLE = f"{PHONE}/genie_bundle"
LOCAL_BUNDLE = Path(__file__).resolve().parent.parent / "models/galaxy/phi35_8elite"


def phi_prompt(system: str, user: str) -> str:
    """phi-3.5's chat template, as Foundry Local applies it on the laptop."""
    head = f"<|system|>\n{system}<|end|>\n" if system else ""
    return f"{head}<|user|>\n{user}<|end|>\n<|assistant|>\n"


class PhoneLLM:
    """Drop-in for LocalLLM that runs phi-3.5-mini on the phone's NPU.

    Each call uses Qualcomm's Genie config with greedy sampling (top-k 1, as LocalLLM asks Foundry for
    temperature 0) and max-num-tokens set to the same limit LocalLLM sends as max_tokens. The model
    reloads for every call (about 1.5 s); speed is not a goal here.
    """

    def __init__(self, tokenizer_path: str | Path, deadline_s: float = 300):
        from tokenizers import Tokenizer
        from inferx.llm import MIN_MAX_TOKENS
        self.model = "phi-3.5-mini-instruct (Genie, Galaxy NPU)"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.deadline_s, self.min_tokens = deadline_s, MIN_MAX_TOKENS
        self.calls = 0
        self.configs: set[int] = set()

    def config(self, limit: int) -> str:
        """The name of a greedy Genie config that stops after `limit` new tokens, pushed on first use."""
        import json
        name = f"genie_greedy_{limit}.json"
        if limit not in self.configs:
            config = json.loads((LOCAL_BUNDLE / "genie_config.json").read_text())
            config["dialog"]["sampler"].update({"temp": 1.0, "top-k": 1, "top-p": 1.0})
            config["dialog"]["max-num-tokens"] = limit
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_text(json.dumps(config, indent=2))
                adb("push", str(path), f"{BUNDLE}/{name}")
            self.configs.add(limit)
        return name

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return text if len(encoded.ids) <= max_tokens else self.tokenizer.decode(encoded.ids[:max_tokens])

    def complete(self, system: str, user: str, max_tokens: int = 200):
        import time
        from inferx.llm import Completion, LLMError
        self.calls += 1
        name = f"prompt_{uuid.uuid4().hex[:8]}.txt"
        limit = max(max_tokens, self.min_tokens)
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(phi_prompt(system, user), encoding="utf-8", newline="\n")
            try:
                config = self.config(limit)
                adb("push", str(path), f"{BUNDLE}/{name}")
                output = shell(f"cd {BUNDLE} && {RUNTIME}/bin/genie-t2t-run -c {config} --prompt_file {name}; "
                               f"rm -f {name}", timeout=self.deadline_s)
            except Exception as error:
                raise LLMError(f"{type(error).__name__}: {error}") from error
        if "[BEGIN]:" not in output or "[END]" not in output:
            raise LLMError("no reply from Genie: " + output.strip()[-300:])
        text = output.split("[BEGIN]:", 1)[1].split("[END]", 1)[0]
        text = self.truncate(text, limit).strip()
        if not text:
            raise LLMError("empty reply")
        total = time.perf_counter() - start
        return Completion(text=text, ttft_s=total, total_s=total, chunks=self.count_tokens(text),
                          prompt_tokens=self.count_tokens(system) + self.count_tokens(user))


# --- the memory controller -------------------------------------------------------------------------

class PhoneEmbedder:
    """MiniLM on the phone's NPU, with the same tokenization and pooling as NpuEmbedder.

    Results are cached by text, and prefetch() embeds many texts in one phone call.
    """

    LENGTH = 128

    def __init__(self, tokenizer_path: str | Path, chunk: int = 64):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.LENGTH)
        self.tokenizer.enable_padding(length=self.LENGTH)
        self.net = PhoneNet("minilm.dlc", {"input_ids": np.int32, "attention_mask": np.int32, "token_type_ids": np.int32},
                            "last_hidden_state", (1, self.LENGTH, 384))
        self.cache: dict[str, np.ndarray] = {}
        self.chunk, self.phone_calls = chunk, 0

    def prefetch(self, texts) -> None:
        todo = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(todo), self.chunk):
            group = todo[start:start + self.chunk]
            encoded = [self.tokenizer.encode(text) for text in group]
            feeds = [{"input_ids": np.array([e.ids]), "attention_mask": np.array([e.attention_mask]),
                      "token_type_ids": np.zeros((1, self.LENGTH))} for e in encoded]
            self.phone_calls += 1
            for text, e, hidden in zip(group, encoded, self.net.run(feeds)):
                mask = np.array([e.attention_mask], dtype=np.float32)[..., None]
                vector = ((hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9))[0]
                self.cache[text] = (vector / np.maximum(np.linalg.norm(vector), 1e-9)).astype(np.float32)

    def __call__(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.prefetch([text])
        return self.cache[text]


class PhoneHead:
    """Drop-in for LinearHead's proba and predict, running one decision head on the phone's NPU."""

    def __init__(self, classes: list, name: str, width: int):
        self.classes, self.name, self.width = list(classes), name, width
        self.net = PhoneNet(f"{name}.dlc", {"x": np.float32}, "probabilities", (1, len(self.classes)))
        self.cache: dict[bytes, np.ndarray] = {}
        self.phone_calls = 0

    def prefetch(self, rows) -> None:
        rows = [np.asarray(row, dtype=np.float32).reshape(1, self.width) for row in rows]
        todo = list({row.tobytes(): row for row in rows if row.tobytes() not in self.cache}.values())
        if todo:
            self.phone_calls += 1
            for row, result in zip(todo, self.net.run([{"x": row} for row in todo])):
                self.cache[row.tobytes()] = result[0]

    def proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        self.prefetch(list(x))
        return np.vstack([self.cache[row.reshape(1, -1).tobytes()] for row in x])

    def predict(self, x: np.ndarray) -> list:
        return [self.classes[i] for i in self.proba(x).argmax(1)]


def load_phone_controller(heads_path: str | Path, tokenizer_path: str | Path):
    """The trained controller with MiniLM and all 4 heads on the phone. Thresholds and rules are unchanged."""
    from inferx.memory_classifier import MemoryClassifier, pair_features, render

    class PhoneClassifier(MemoryClassifier):
        def prefetch(self, messages) -> None:
            """Embed a user's messages and score the 3 single-message heads in a few phone calls."""
            texts = [render(m.speaker, m.text) for m in messages]
            self.embed.prefetch(texts)
            vectors = [self.embed(text) for text in texts]
            for head in (self.sensitive, self.store, self.kind):
                head.prefetch(vectors)

        def decide(self, speaker, text, active, store_threshold=None):
            self._current = (speaker, text)
            return super().decide(speaker, text, active, store_threshold)

        def nearest(self, vector, active):
            # Score every replace candidate in one phone call; decide() then reads them from the cache.
            chosen = super().nearest(vector, active)
            speaker, text = self._current
            self.replaces.prefetch([pair_features(vector, item.vector, text, item.text, speaker, item.speaker)
                                    for item in chosen])
            return chosen

    data = np.load(heads_path)
    classifier = PhoneClassifier.load(heads_path, PhoneEmbedder(tokenizer_path))
    for name in ("store", "kind", "sensitive", "replaces"):
        head = getattr(classifier, name)
        setattr(classifier, name, PhoneHead(head.classes, name, data[f"{name}_weights"].shape[0]))
    return classifier
