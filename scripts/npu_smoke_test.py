"""Check that ONNX Runtime can see the Snapdragon NPU and run a tiny model on it.

CPU fallback is disabled, so the inference step only passes if the NPU runs the
whole graph. Needs the `onnx` package.
"""
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import onnxruntime_qnn as qnn

EP = "QNNExecutionProvider"

ort.register_execution_provider_library(EP, qnn.get_library_path())
devices = ort.get_ep_devices()
print("EP devices:")
for d in devices:
    print(f"  {d.ep_name:<28} {d.device.type}  vendor={d.device.vendor}")

npu = [d for d in devices if d.ep_name == EP and "NPU" in str(d.device.type)]
if not npu:
    sys.exit("FAIL: no QNN NPU device found")
print("PASS: NPU device visible")

# A quantized (QDQ) Add graph, because the NPU runs quantized models natively.
scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.01])
zero = helper.make_tensor("zero", TensorProto.UINT8, [], [0])
const = helper.make_tensor("b", TensorProto.FLOAT, [1, 4], [1.0, 1.0, 1.0, 1.0])
nodes = [
    helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["x_q"]),
    helper.make_node("DequantizeLinear", ["x_q", "scale", "zero"], ["x_d"]),
    helper.make_node("QuantizeLinear", ["b", "scale", "zero"], ["b_q"]),
    helper.make_node("DequantizeLinear", ["b_q", "scale", "zero"], ["b_d"]),
    helper.make_node("Add", ["x_d", "b_d"], ["sum"]),
    helper.make_node("QuantizeLinear", ["sum", "scale", "zero"], ["sum_q"]),
    helper.make_node("DequantizeLinear", ["sum_q", "scale", "zero"], ["y"]),
]
graph = helper.make_graph(
    nodes,
    "npu_smoke",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    initializer=[scale, zero, const],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
model.ir_version = 10

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "smoke.onnx")
    onnx.save(model, path)

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(npu, {"htp_performance_mode": "burst"})
    session = ort.InferenceSession(path, sess_options=options)
    print("Session providers:", session.get_providers())

    x = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    y = session.run(None, {"x": x})[0]

print("Output:", y)
if not np.allclose(y, x + 1.0, atol=0.02):
    sys.exit("FAIL: wrong output")
print("PASS: model ran on the NPU with CPU fallback disabled")
