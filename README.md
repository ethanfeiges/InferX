# InferX: decoupled memory control for on-device LLM agents

Laptop build of the thesis experiment. The main LLM answers, reads memory and drafts facts, and a
controller makes every memory decision. Heavy work waits for idle time. The first target task is
a personal assistant on a Galaxy phone.

## What runs today

| Part | Status |
|---|---|
| `inferx/store.py`: core, recall and archival tiers in SQLite with sqlite-vec | Built |
| `inferx/controller.py`: filter, core-fact rules, add/update/ignore, fact checks | Built, kept as the rules baseline |
| `inferx/memory_classifier.py`: trained memory decisions | Built and connected, runs on the CPU |
| `inferx/systems.py`: plain retrieval, rule-based design and trained design | Built |
| `inferx/assistant_sim.py`, `inferx/assistant_score.py`: simulated assistant data and scoring | Built |
| `inferx/scheduler.py`: when deferred work runs | Stub, no device checks |
| `inferx/locomo.py`: LoCoMo loading and official-style scoring | Built |
| NPU compile of the classifiers, consolidation, summaries, cloud fallback | Not built |
| mem0 and Letta baselines, LongMemEval, Galaxy S25 | Not built |

The classifiers and embeddings run on the CPU, so this build does not yet test memory control on
the NPU. Speed and energy are not goals
of the thesis. Timings stay in the run files only as diagnostics.

## Setup (Windows ARM64)

Needs native ARM64 Python 3.12, the MSVC ARM64 build tools, the Windows SDK and Foundry Local.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
scripts\build_sqlite_vec.cmd
.venv\Scripts\python scripts\test_sqlite_vec.py
.venv\Scripts\python -m pytest tests
```

`third_party/`, `data/locomo/` and the MiniLM `.onnx` files are downloaded, not stored in this repository:
- sqlite-vec v0.1.9 amalgamation and SQLite 3.49.1 amalgamation, for building `vec0.dll`
- `data/locomo/locomo10.json` from snap-research/locomo (CC BY-NC 4.0)
- `models/minilm`: `model_qint8_arm64.onnx` and `tokenizer.json` from
  sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0)

## Run

```
foundry server start
foundry model load qwen2.5-1.5b
foundry server status
.venv\Scripts\python scripts\run_locomo.py --url <Public API URL> --conversation 1
```

Results go to `runs/<timestamp>/`: `summary.json`, one `.jsonl` per system, the saved memory
database and a log.

To use phi instead, load `phi-3.5-mini`, then add
`--model phi-3.5-mini-instruct-qnn-npu:3 --tokenizer <its tokenizer.json in the Foundry cache>`.

## Results so far

Conversation 1 only (369 messages, 105 questions), on mains power, same prompts for both models.
Prompts were written and tuned on qwen. These are early numbers, not thesis results.

| | qwen plain | qwen decoupled | phi plain | phi decoupled |
|---|---|---|---|---|
| Score | 0.179 | 0.195 | 0.354 | 0.385 |
| Score without adversarial questions | 0.183 | 0.216 | 0.372 | 0.463 |
| Evidence recall | 0.56 | 0.628 | 0.541 | 0.664 |
| Adversarial questions declined (of 24) | 4 | 3 | 7 | 3 |

Runs: `runs/20261105-164655` (qwen) and `runs/20261105-195628` (phi).

Known problems in the phi run, from reading a sample of its memory:
- About 9 of 25 sampled facts had a clear error, mostly wrong dates or events that did not happen.
- None of the 9 updates looked like a real change of fact. Some replaced an earlier event date.

Comparing the two models on conversation 1 means it is no longer clean for choosing a model.
Use other conversations for the final comparison. The fact-writing fixes then scored 0.377 on
conversation 1 (`runs/20261105-214852`), with no clear change from 0.385, and made no updates.

### Personal assistant (simulated)

`inferx/assistant_sim.py` generates calendar, message, order and noise data with exact labels.
The test users are untouched until the final comparison.

```
.venv\Scripts\python scripts\generate_assistant_data.py
.venv\Scripts\python scripts\train_memory_classifier.py
.venv\Scripts\python scripts\tune_memory_thresholds.py
.venv\Scripts\python scripts\run_locomo.py --url <URL> --model <phi> --tokenizer <phi tokenizer> --data data/assistant/val.json --conversation 0 --systems plain_retrieval,decoupled,trained
.venv\Scripts\python scripts\compare_assistant_runs.py runs\<oldest> ... runs\<newest>
```

5 validation users, 268 questions, phi. These users share wording with the training data, which
favors the trained system. Runs: `20261105-220607` to `225313` (plain) and `230839` to `234602`.

| | Plain retrieval | Current rules | Trained classifiers |
|---|---|---|---|
| Correct overall | 76.9% | 73.9% | 78.0% |
| About something that changed (72 questions) | 44 | 43 | 50 |
| Replies giving the old value | 18 | 23 | 10 |
| Details that never changed (126) | 88.9% | 85.7% | 86.5% |
| History questions (12) | 58.3% | 50.0% | 66.7% |
| No-answer questions declined (25) | 23 | 23 | 24 |

Trained minus plain is +1.1 points overall (95% interval -3.7 to +6.0), and +8.3 on questions
about something that changed (-4.2 to +20.8), so neither difference is clear yet. All 52 one-time
codes were dropped before saving. Each user has only about 100 saved messages, so plain retrieval
almost always finds the evidence, and longer histories are the next test.

Classifier decisions on `data/assistant/handwritten_check2.json` (40 messages I wrote and scored
once): worth remembering F1 0.95, type accuracy 0.97, 4 of 7 updates found, 1 wrong overwrite,
0 of 2 codes kept. The classifiers run on the CPU, and unclear cases don't go to the main LLM yet.

## Method notes

- Conversation 0 is for tuning and conversation 1 is the test. The answer prompt was chosen on
  synthetic planted-fact tests (`scripts/answer_prompt_sweep.py`).
- Scoring follows LoCoMo's `task_eval/evaluation.py`. Evidence recall measures whether the
  evidence messages reached the prompt, regardless of the model's answer.
- Simulated idle: deferred work runs at the end of each session.
- Recent messages are set to 0 tokens in benchmark runs, because benchmark questions aren't
  part of the chat.
- Foundry's `completion_tokens` over-reports, so `inferx/llm.py` counts output chunks itself.
- `scripts/` also holds the NPU and Foundry diagnostics used to debug the driver failure.
