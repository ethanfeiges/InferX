"""Export the memory classifier's heads to ONNX, run them on the NPU, and check they match the CPU.

Usage: python scripts/check_classifier_npu.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.embed import Embedder  # noqa: E402
from inferx.memory_classifier import MemoryClassifier, evaluate, pair_features, render  # noqa: E402
from inferx.npu_classifier import export_all, move_to_npu  # noqa: E402

embed = Embedder(ROOT / "models/minilm")
heads = ROOT / "models/assistant_classifier/heads.npz"
out = ROOT / "models/assistant_classifier/npu"
cpu = MemoryClassifier.load(heads, embed)
export_all(cpu, out)
npu = move_to_npu(MemoryClassifier.load(heads, embed), out)
print("sessions:", npu.store.session.get_providers())

rows = []
for name in ("handwritten.json", "handwritten_check.json", "handwritten_check2.json"):
    rows += json.loads((ROOT / "data/assistant" / name).read_text(encoding="utf-8"))
vectors = np.stack([embed(render(row["speaker"], row["text"])) for row in rows])
report = {}
for name in ("store", "kind", "sensitive"):
    a, b = getattr(cpu, name).proba(vectors), getattr(npu, name).proba(vectors)
    report[name] = {"max_probability_difference": round(float(np.abs(a - b).max()), 4),
                    "same_decision": f"{int((a.argmax(1) == b.argmax(1)).sum())} of {len(a)}"}
pairs = np.stack([pair_features(vectors[i], vectors[j], rows[i]["text"], rows[j]["text"], rows[i]["speaker"], rows[j]["speaker"])
                  for i in range(len(rows)) for j in range(max(0, i - 5), i)])
a, b = cpu.replaces.proba(pairs), npu.replaces.proba(pairs)
threshold = cpu.replace_threshold
report["replaces"] = {"max_probability_difference": round(float(np.abs(a - b).max()), 4),
                      "same_decision_at_threshold": f"{int(((a[:, 1] >= threshold) == (b[:, 1] >= threshold)).sum())} of {len(a)}"}
for name in ("handwritten_check2.json",):
    timeline = json.loads((ROOT / "data/assistant" / name).read_text(encoding="utf-8"))
    on_cpu, on_npu = evaluate(cpu, timeline), evaluate(npu, timeline)
    report[f"{name} replay"] = {"cpu": {k: on_cpu[k] for k in ("store", "update", "sensitive_kept")},
                                "npu": {k: on_npu[k] for k in ("store", "update", "sensitive_kept")}}
print(json.dumps(report, indent=1))
(out / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# The whole controller on the NPU: embeddings and decisions.
from inferx.npu_classifier import NpuEmbedder  # noqa: E402

npu_embed = NpuEmbedder(ROOT / "models/minilm_fp32")
full = move_to_npu(MemoryClassifier.load(heads, npu_embed), out)
full_report = {}
for name in ("handwritten.json", "handwritten_check.json", "handwritten_check2.json"):
    timeline = json.loads((ROOT / "data/assistant" / name).read_text(encoding="utf-8"))
    on_cpu, on_npu = evaluate(cpu, timeline), evaluate(full, timeline)
    keys = ("store", "update", "kind_accuracy", "sensitive_kept")
    full_report[name] = {"cpu": {k: on_cpu[k] for k in keys}, "all_npu": {k: on_npu[k] for k in keys}}
print(json.dumps(full_report, indent=1))
(out / "check_all_npu.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
