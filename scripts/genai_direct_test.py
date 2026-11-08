"""Run a Foundry-cached NPU model through ONNX Runtime GenAI, without Foundry.

Comparing this with foundry_repro.py shows whether a failure follows Foundry's
runtime or the NPU stack underneath it.

Usage: python genai_direct_test.py <model_dir> [requests] [max_length]
"""
import sys
import time

import onnxruntime_genai as og
import onnxruntime_qnn as qnn

model_dir = sys.argv[1]
requests = int(sys.argv[2]) if len(sys.argv) > 2 else 6
max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 200
PROMPTS = [
    "Name a color.",
    "What is 2 plus 2?",
    "Name a fruit.",
    "What is the capital of France?",
    "Name an animal.",
    "What day comes after Monday?",
]

og.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
start = time.perf_counter()
model = og.Model(model_dir)
tokenizer = og.Tokenizer(model)
stream = tokenizer.create_stream()
print(f"loaded in {time.perf_counter() - start:.1f}s", flush=True)

for i in range(requests):
    prompt = f"<|im_start|>user\n{PROMPTS[i % len(PROMPTS)]}<|im_end|>\n<|im_start|>assistant\n"
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)
    generator = og.Generator(model, params)
    start = time.perf_counter()
    text = ""
    tokens = 0
    try:
        generator.append_tokens(tokenizer.encode(prompt))
        while not generator.is_done():
            generator.generate_next_token()
            text += stream.decode(generator.get_next_tokens()[0])
            tokens += 1
    except Exception as error:
        print(f"req {i + 1}: ERROR after {tokens} tokens: {str(error)[:300]}", flush=True)
        sys.exit(1)
    elapsed = time.perf_counter() - start
    print(f"req {i + 1}: {tokens} tokens, {elapsed:.1f}s, {tokens / elapsed:.1f} tok/s, reply={text.strip()[:60]!r}", flush=True)
    del generator
print("PASS: all requests completed")
