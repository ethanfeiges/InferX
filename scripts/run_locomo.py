"""Run one LoCoMo conversation through the decoupled design and the plain-retrieval baseline.

Usage:
  python scripts/run_locomo.py --url http://127.0.0.1:<port>/v1 [--conversation 1] [--limit 30]

Writes per-question results and a summary to runs/<timestamp>/.
Simulated idle: deferred work runs at the end of each session.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.context import Budget  # noqa: E402
from inferx.embed import Embedder  # noqa: E402
from inferx.llm import LocalLLM  # noqa: E402
from inferx.locomo import evidence_recall, load, score, sessions  # noqa: E402
from inferx.memory_classifier import MemoryClassifier  # noqa: E402
from inferx.cloud import CloudClient, CloudPolicy, user_env  # noqa: E402
from inferx.systems import DecoupledMemory, FullContext, LocalMemory, PlainRetrieval, TrainedMemory  # noqa: E402

DEFAULT_MODEL = "qwen2.5-1.5b-instruct-qnn-npu:3"
MAX_CONSECUTIVE_ERRORS = 3


def on_ac_power() -> bool:
    """Power state may change NPU speed, so every run records it."""
    status = ctypes.create_string_buffer(12)
    ctypes.windll.kernel32.GetSystemPowerStatus(status)
    return status.raw[0] == 1


def sample_questions(qa: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Take an even share from each category, keeping dataset order within the result."""
    if not limit or limit >= len(qa):
        return qa
    by_category = defaultdict(list)
    for index, item in enumerate(qa):
        by_category[item["category"]].append(index)
    rng = random.Random(seed)
    for indices in by_category.values():
        rng.shuffle(indices)
    chosen: list[int] = []
    while len(chosen) < limit:
        for category in sorted(by_category):
            if by_category[category] and len(chosen) < limit:
                chosen.append(by_category[category].pop())
    return [qa[index] for index in sorted(chosen)]


def run_system(system, data, questions, out, log):
    timings = {"ingest_s": 0.0, "idle_s": 0.0}
    idle_totals: dict[str, float] = defaultdict(float)
    for number, _date, messages in data:
        start = time.perf_counter()
        for message in messages:
            system.ingest(message)
        timings["ingest_s"] += time.perf_counter() - start
        start = time.perf_counter()
        for key, value in system.on_idle().items():
            idle_totals[key] += value
        timings["idle_s"] += time.perf_counter() - start
        log(f"{system.name}: session {number} ready ({timings['idle_s']:.0f}s idle work so far)")

    results = []
    consecutive_errors = 0
    for index, qa in enumerate(questions, 1):
        answer = system.answer(qa["question"])
        record = {
            "question": qa["question"],
            "category": qa["category"],
            "answer": qa.get("answer", qa.get("adversarial_answer")),
            "evidence": qa.get("evidence", []),
            "qtype": qa.get("qtype"),
            "stale_answer": qa.get("stale_answer"),
            "score": None if answer.error else score(qa, answer.text),
            "evidence_recall": evidence_recall(qa, answer.sources),
            **asdict(answer),
        }
        results.append(record)
        out.write(json.dumps(record) + "\n")
        out.flush()
        log(f"{system.name}: q{index}/{len(questions)} score={record['score']} "
            f"recall={record['evidence_recall']} ttft={answer.ttft_s and round(answer.ttft_s, 1)} "
            f"reply={answer.text[:50]!r} err={answer.error}")
        consecutive_errors = consecutive_errors + 1 if answer.error else 0
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log(f"{system.name}: stopping after {MAX_CONSECUTIVE_ERRORS} consecutive LLM errors")
            break
    return results, timings, dict(idle_totals)


def mean(values):
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(results, timings, idle, system, planned):
    answered = [r for r in results if not r["error"]]
    categories = {}
    for category in sorted({r["category"] for r in results}):
        group = [r for r in answered if r["category"] == category]
        categories[str(category)] = {
            "n": len(group),
            "score": mean([r["score"] for r in group]),
            "evidence_recall": mean([r["evidence_recall"] for r in group]),
        }
    ttfts = [r["ttft_s"] for r in answered if r["ttft_s"] is not None]
    return {
        "system": system.name,
        "questions_planned": planned,
        "questions_answered": len(answered),
        "errors": len(results) - len(answered),
        "score": mean([r["score"] for r in answered]),
        "score_excluding_adversarial": mean([r["score"] for r in answered if r["category"] != 5]),
        "evidence_recall": mean([r["evidence_recall"] for r in answered]),
        "not_mentioned_rate": mean([float("not mentioned" in r["text"].lower()) for r in answered]),
        "by_category": categories,
        "ttft_s_median": round(statistics.median(ttfts), 2) if ttfts else None,
        "total_s_mean": mean([r["total_s"] for r in answered]),
        "retrieval_ms_mean": round(1000 * statistics.mean(r["retrieval_s"] for r in answered), 1) if answered else None,
        "prompt_tokens_mean": mean([r["prompt_tokens"] for r in answered]),
        "cloud_calls": sum(r["cloud_calls"] for r in results),
        "cloud_bytes": sum(r.get("cloud_bytes", 0) for r in results),
        "memory_cloud_requests": sum(r.get("memory_cloud_requests", 0) for r in results) + int(idle.get("requests", 0)),
        "memory_cloud_bytes": sum(r.get("memory_cloud_bytes", 0) for r in results)
                              + int(idle.get("bytes_sent", 0) + idle.get("bytes_received", 0)),
        "ingest_s": round(timings["ingest_s"], 1),
        "idle_s": round(timings["idle_s"], 1),
        "idle": {key: round(value, 1) for key, value in idle.items()},
        "store": system.store.counts(),
    }


def load_controller(on_npu: bool, embed):
    """The trained controller, on the CPU or with MiniLM and its decision heads on the NPU.

    Retrieval embeddings stay on the shared CPU embedder either way, so only memory decisions move.
    """
    heads = ROOT / "models/assistant_classifier/heads.npz"
    if not on_npu:
        return MemoryClassifier.load(heads, embed)
    from inferx.npu_classifier import NpuEmbedder, move_to_npu
    return move_to_npu(MemoryClassifier.load(heads, NpuEmbedder(ROOT / "models/minilm_fp32")),
                       ROOT / "models/assistant_classifier/npu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=str(
        Path.home() / ".foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-qnn-npu-3/v3/tokenizer.json"))
    parser.add_argument("--data", default=str(ROOT / "data/locomo/locomo10.json"),
                        help="any file in LoCoMo's layout, such as data/assistant/val.json")
    parser.add_argument("--conversation", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sessions", type=int, help="use only the first N sessions (for smoke tests)")
    parser.add_argument("--retrieved-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", default="plain_retrieval,decoupled")
    parser.add_argument("--cloud", choices=["off", "gate"], default="off",
                        help="gate applies the cloud rule to every system; local_memory always uses the cloud")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8765/v1",
                        help="the local stand-in by default; a remote provider needs --allow-remote")
    parser.add_argument("--cloud-model", default="cloud", help="model name at the cloud provider")
    parser.add_argument("--allow-remote", action="store_true",
                        help="allow requests off this laptop: a remote --cloud-url and the mem0_cloud system. "
                             "The cloud key is read from INFERX_CLOUD_API_KEY, mem0's from MEM0_API_KEY")
    parser.add_argument("--npu-controller", action="store_true",
                        help="run the trained controller (MiniLM and its decision heads) on the NPU")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log_file = (run_dir / "log.txt").open("w", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%H:%M:%S')} {text}"
        print(line.encode("ascii", "replace").decode(), flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    sample = load(args.data)[args.conversation]
    data = sessions(sample)
    if args.sessions:
        data = data[: args.sessions]
        kept = {str(number) for number, _, _ in data}
        qa = [q for q in sample["qa"] if q.get("evidence") and all(
            e.split(":")[0][1:] in kept for e in q["evidence"])]
    else:
        qa = sample["qa"]
    questions = sample_questions(qa, args.limit, args.seed)

    log(f"on_ac_power={on_ac_power()}")
    embed = Embedder(ROOT / "models/minilm")
    budget = Budget(retrieved=args.retrieved_tokens, recent=0)
    summaries = {}
    (run_dir / "config.json").write_text(json.dumps({
        **vars(args), "budget": asdict(budget), "questions": len(questions),
        "messages": sum(len(m) for _, _, m in data), "sessions_used": len(data),
    }, indent=2))
    factories = {
        "plain_retrieval": PlainRetrieval,
        "local_memory": LocalMemory,
        "full_context": FullContext,
        "decoupled": DecoupledMemory,
        # Only in .venv-mem0, which has mem0 installed.
        "mem0_2025_local": lambda llm, embed, budget: __import__("inferx.mem0_port", fromlist=["Mem0V1Port"]).Mem0V1Port(
            llm, embed, budget),
        "mem0_cloud": lambda llm, embed, budget: __import__("inferx.mem0_port", fromlist=["Mem0Cloud"]).Mem0Cloud(
            llm, embed, budget, allow_remote=args.allow_remote, run_id=f"inferx-{run_dir.name}-{args.conversation}"),
        "trained": lambda llm, embed, budget: TrainedMemory(llm, embed, load_controller(args.npu_controller, embed), budget),
    }
    def cloud_client():
        return CloudClient(args.cloud_url, args.cloud_model, user_env("INFERX_CLOUD_API_KEY"),
                           allow_remote=args.allow_remote)

    for name in args.systems.split(","):
        llm = LocalLLM(args.url, args.model, args.tokenizer)
        system = factories[name](llm, embed, budget)
        if name == "local_memory":
            system.cloud = CloudPolicy(cloud_client(), "always")
        elif args.cloud == "gate":
            system.cloud = CloudPolicy(cloud_client(), "gate")
        with (run_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            results, timings, idle = run_system(system, data, questions, out, log)
        summaries[name] = summarize(results, timings, idle, system, len(questions))
        summaries[name]["llm_calls_total"] = llm.calls
        with sqlite3.connect(run_dir / f"{name}.sqlite") as saved:
            system.store.db.backup(saved)
        (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print("RUN_DIR", run_dir)


if __name__ == "__main__":
    main()
