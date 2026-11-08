"""On-device sentence embeddings using MiniLM on the CPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DIM = 384


class Embedder:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.session = ort.InferenceSession(
            str(model_dir / "model_qint8_arm64.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def __call__(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        vector = pooled[0]
        return (vector / np.maximum(np.linalg.norm(vector), 1e-9)).astype(np.float32)
