"""Run the memory classifier's decision heads on the Snapdragon NPU.

Each linear head becomes a tiny ONNX graph (MatMul, Add, Softmax), with its input scaling folded into
the weights. The QNN execution provider runs it on the NPU's HTP in fp16, with CPU fallback disabled,
so a session only opens if the NPU runs the whole graph.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EP = "QNNExecutionProvider"


def head_to_onnx(head, path: Path, batch: int = 1) -> None:
    """Write one LinearHead as an ONNX graph with a fixed input shape, as the NPU needs."""
    weights = (head.weights / head.std[:, None]).astype(np.float32)
    bias = (head.bias - (head.mean / head.std) @ head.weights).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["logits_raw"]),
        helper.make_node("Add", ["logits_raw", "b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "memory_head",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, weights.shape[0]])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [batch, weights.shape[1]])],
        initializer=[numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def npu_session(path: Path):
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
    try:
        ort.register_execution_provider_library(EP, qnn.get_library_path())
    except Exception:
        pass  # already registered in this process
    devices = [d for d in ort.get_ep_devices() if d.ep_name == EP and "NPU" in str(d.device.type)]
    if not devices:
        raise RuntimeError("no Snapdragon NPU device found")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {"htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"})
    return ort.InferenceSession(str(path), sess_options=options)


class NpuHead:
    """Drop-in for LinearHead's predict and proba, running on the NPU."""

    def __init__(self, classes: list, path: Path, batch: int = 1):
        self.classes, self.batch = list(classes), batch
        self.session = npu_session(path)

    def proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        rows = []
        for start in range(0, len(x), self.batch):
            chunk = x[start:start + self.batch]
            padded = np.vstack([chunk, np.zeros((self.batch - len(chunk), x.shape[1]), dtype=np.float32)])
            rows.append(self.session.run(None, {"x": padded})[0][: len(chunk)])
        return np.vstack(rows)

    def predict(self, x: np.ndarray) -> list:
        return [self.classes[i] for i in self.proba(x).argmax(1)]


def export_all(classifier, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("store", "kind", "sensitive", "replaces"):
        head_to_onnx(getattr(classifier, name), out_dir / f"{name}.onnx")


def move_to_npu(classifier, out_dir: Path):
    """Swap each head of a loaded MemoryClassifier for its NPU version."""
    for name in ("store", "kind", "sensitive", "replaces"):
        head = getattr(classifier, name)
        setattr(classifier, name, NpuHead(head.classes, out_dir / f"{name}.onnx"))
    return classifier


class NpuEmbedder:
    """MiniLM on the NPU in fp16. Same pooling as the CPU Embedder, with inputs padded to 128 tokens.

    Needs models/minilm_fp32/model_fixed.onnx: the full-precision model from
    sentence-transformers/all-MiniLM-L6-v2 with its shape fixed to 1 x 128, because the CPU copy is
    dynamically quantized and the NPU can't run that.
    """

    LENGTH = 128

    def __init__(self, model_dir: str | Path):
        from tokenizers import Tokenizer
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.LENGTH)
        self.tokenizer.enable_padding(length=self.LENGTH)
        self.session = npu_session(model_dir / "model_fixed.onnx")

    def __call__(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)
        hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
        weights = mask[..., None].astype(np.float32)
        vector = ((hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9))[0]
        return (vector / np.maximum(np.linalg.norm(vector), 1e-9)).astype(np.float32)
