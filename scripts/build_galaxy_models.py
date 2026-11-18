"""Build the controller's DLC files for the Galaxy phone and copy everything the phone needs.

Steps:
  1. Copy MiniLM (models/minilm_fp32/model_fixed.onnx) with its attention mask rebuilt as plain arithmetic,
     (1 - mask) * -10000, the standard BERT form. The original casts the mask to bool, which the phone's
     NPU rejects. On the laptop the copy's output is identical to the original's.
  2. Convert MiniLM and the 4 heads (models/assistant_classifier/npu/*.onnx) to DLC with QAIRT's
     qairt-converter, at full precision. The NPU runs them in fp16 either way; converting at fp16
     (--float_bitwidth 16) gave wrong MiniLM outputs on the phone.
  3. Push the DLC files, QAIRT's Android runtime and the phi-3.5 Genie bundle to the phone.

Needs: QAIRT 2.52 unpacked under models/galaxy/qairt_2.52, the converter's Python in .venv-qairt
(x64 Python 3.12, numpy 1.26.4, onnx 1.19.1, setuptools<75), and the phi-3.5 bundle for Snapdragon 8 Elite
in models/galaxy/phi35_8elite.

Usage: .venv\\Scripts\\python scripts\\build_galaxy_models.py [--no-push]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QAIRT = ROOT / "models/galaxy/qairt_2.52/qairt/2.52.1.261105"
OUT = ROOT / "models/galaxy/controller"
HEADS = ("store", "kind", "sensitive", "replaces")


def phone_minilm(source: Path, target: Path) -> None:
    model = onnx.load(str(source))
    graph = model.graph
    users = {}
    for node in graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    # The bool mask feeds a single Where node, whose output is the additive attention bias.
    where = next(node for node in graph.node if node.op_type == "Where"
                 and any(p.op_type == "Cast" for p in graph.node if p.output and p.output[0] == node.input[0]))
    bias = where.output[0]
    length = graph.input[0].type.tensor_type.shape.dim[1].dim_value
    new = [
        helper.make_node("Cast", ["attention_mask"], ["mb_f"], to=TensorProto.FLOAT, name="mb_cast"),
        helper.make_node("Sub", ["mb_one", "mb_f"], ["mb_inv"], name="mb_sub"),
        helper.make_node("Mul", ["mb_inv", "mb_neg"], ["mb_bias2d"], name="mb_mul"),
        helper.make_node("Reshape", ["mb_bias2d", "mb_shape"], [bias], name="mb_reshape"),
    ]
    graph.initializer.extend([numpy_helper.from_array(np.array(1.0, np.float32), "mb_one"),
                              numpy_helper.from_array(np.array(-1e4, np.float32), "mb_neg"),
                              numpy_helper.from_array(np.array([1, 1, 1, length], np.int64), "mb_shape")])
    nodes = [node for node in graph.node if node is not where]
    del graph.node[:]
    graph.node.extend(new + nodes)
    outputs = {output.name for output in graph.output}
    while True:  # drop the old mask nodes, now unused
        used = {name for node in graph.node for name in node.input} | outputs
        dead = [node for node in graph.node if not any(name in used for name in node.output)]
        if not dead:
            break
        for node in dead:
            graph.node.remove(node)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))


def check_same(source: Path, target: Path) -> float:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / "models/minilm_fp32/tokenizer.json"))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)
    a = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    worst = 0.0
    for text in ("user: Sam moved to Leeds last month.", "assistant: ok thanks"):
        encoded = tokenizer.encode(text)
        feed = {"input_ids": np.array([encoded.ids]), "attention_mask": np.array([encoded.attention_mask]),
                "token_type_ids": np.zeros((1, 128), np.int64)}
        mask = feed["attention_mask"][0].astype(bool)
        worst = max(worst, float(np.abs(a.run(None, feed)[0][0][mask] - b.run(None, feed)[0][0][mask]).max()))
    return worst


def convert(source: Path, target: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(QAIRT / "lib/python"),
           "PATH": str(QAIRT / "lib/x86_64-windows-msvc") + os.pathsep + os.environ["PATH"]}
    subprocess.run([str(ROOT / ".venv-qairt/Scripts/python.exe"), str(QAIRT / "bin/x86_64-windows-msvc/qairt-converter"),
                    "--input_network", str(source), "--output_path", str(target)],
                   env=env, check=True, capture_output=True)


def push() -> None:
    from inferx.galaxy import BUNDLE, PHONE, RUNTIME, adb, shell
    shell(f"mkdir -p {RUNTIME}/bin {RUNTIME}/lib {RUNTIME}/dsp {PHONE}/controller {BUNDLE}")
    for local, remote in ((QAIRT / "bin/aarch64-android", f"{RUNTIME}/bin"), (QAIRT / "lib/aarch64-android", f"{RUNTIME}/lib"),
                          (QAIRT / "lib/hexagon-v79/unsigned", f"{RUNTIME}/dsp")):
        adb("push", f"{local}{os.sep}.", remote + "/")
    shell(f"chmod +x {RUNTIME}/bin/*")
    for name in ("minilm",) + HEADS:
        adb("push", str(OUT / f"{name}.dlc"), f"{PHONE}/controller/")
    bundle = ROOT / "models/galaxy/phi35_8elite"
    for path in bundle.iterdir():
        if path.suffix in (".bin", ".json") and not path.name.startswith("genie_greedy"):
            adb("push", str(path), f"{BUNDLE}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "models/minilm_fp32/model_fixed.onnx"
    phone_onnx = OUT / "minilm_phone.onnx"
    phone_minilm(source, phone_onnx)
    print("MiniLM copy, largest output difference from the original:", check_same(source, phone_onnx))
    convert(phone_onnx, OUT / "minilm.dlc")
    for name in HEADS:
        convert(ROOT / f"models/assistant_classifier/npu/{name}.onnx", OUT / f"{name}.dlc")
    print("DLC files:", sorted(p.name for p in OUT.glob("*.dlc")))
    if not args.no_push:
        push()
        print("pushed to the phone")


if __name__ == "__main__":
    main()
