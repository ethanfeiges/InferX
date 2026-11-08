"""Run a small convolution model on the NPU repeatedly, without Foundry.

Each round creates a fresh session and runs it many times, with CPU fallback off.
A clean run means the NPU and its driver can handle repeated work outside Foundry.
It does not rule out problems that only appear with large models.

Usage: python npu_stress_test.py [rounds] [runs_per_round]
"""
import os
import sys
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import onnxruntime_qnn as qnn

EP = "QNNExecutionProvider"
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 500

ort.register_execution_provider_library(EP, qnn.get_library_path())
npu = [
    d
    for d in ort.get_ep_devices()
    if d.ep_name == EP and "NPU" in str(d.device.type)
]
if not npu:
    sys.exit("FAIL: no NPU device")


def build(path, layers=8, channels=32, size=64):
    """Save a stack of quantized (QDQ) convolutions."""
    rng = np.random.default_rng(0)
    inits = [
        helper.make_tensor("s", TensorProto.FLOAT, [], [0.02]),
        helper.make_tensor("z", TensorProto.UINT8, [], [128]),
        helper.make_tensor("ws", TensorProto.FLOAT, [], [0.01]),
        helper.make_tensor("wz", TensorProto.UINT8, [], [128]),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "z"], ["t0_q"]),
        helper.make_node("DequantizeLinear", ["t0_q", "s", "z"], ["t0"]),
    ]
    in_channels = 3
    for i in range(layers):
        weights = rng.integers(100, 156, (channels, in_channels, 3, 3), dtype=np.uint8)
        inits.append(numpy_helper.from_array(weights, f"w{i}"))
        nodes += [
            helper.make_node("DequantizeLinear", [f"w{i}", "ws", "wz"], [f"w{i}_d"]),
            helper.make_node(
                "Conv", [f"t{i}", f"w{i}_d"], [f"c{i}"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
            ),
            helper.make_node("QuantizeLinear", [f"c{i}", "s", "z"], [f"t{i + 1}_q"]),
            helper.make_node("DequantizeLinear", [f"t{i + 1}_q", "s", "z"], [f"t{i + 1}"]),
        ]
        in_channels = channels
    nodes.append(helper.make_node("Identity", [f"t{layers}"], ["y"]))
    graph = helper.make_graph(
        nodes,
        "npu_stress",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, channels, size, size])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.save(model, path)


x = np.random.default_rng(1).random((1, 3, 64, 64), dtype=np.float32)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "stress.onnx")
    build(path)
    for r in range(1, rounds + 1):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(npu, {"htp_performance_mode": "burst"})
        session = ort.InferenceSession(path, sess_options=options)
        start = time.perf_counter()
        for i in range(runs):
            try:
                session.run(None, {"x": x})
            except Exception as error:
                sys.exit(f"FAIL: round {r}, run {i}: {error}")
        ms = (time.perf_counter() - start) / runs * 1000
        print(f"round {r}: {runs} runs OK, {ms:.2f} ms/run", flush=True)
        del session
print("PASS: no NPU errors")
