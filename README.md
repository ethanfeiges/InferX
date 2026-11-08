# InferX: decoupled memory control for on-device LLM agents

Laptop build of the thesis experiment, on a Snapdragon X Elite (Windows on ARM). The main LLM
answers questions. A small, separate controller, trained for one task, decides what goes into
memory, what replaces what, and what is never saved. The controller runs on the NPU, and heavy
memory work waits for idle time. The first task is a personal assistant that syncs a phone's
calendar, messages and purchases. A Galaxy S25 build comes after the laptop experiments, and a
home robot (written facts plus images) is future work.

Speed and energy are not goals of the thesis. Timings stay in the run files only as diagnostics.

## Status

| Part | Status |
|---|---|
| `inferx/store.py`: core, recall and archival memory in SQLite with sqlite-vec, plus outdated facts and their replacements | Built |
| `inferx/memory_classifier.py`: trained controller (MiniLM embeddings and 4 small decision heads) | Built |
| `inferx/npu_classifier.py`: controller on the NPU (QNN, fp16, CPU fallback disabled) | Built, checked against the CPU |
| `inferx/cloud.py`: cloud rule (the main LLM asks the cloud only when its memories don't answer) | Built, tested against a local stand-in |
| `inferx/systems.py`: plain retrieval, full context, local memory, rules (`decoupled`) and trained systems | Built |
| `inferx/mem0_port.py`: mem0 2025 algorithm on the local model | Built, validation running |
| `inferx/assistant_sim.py`, `inferx/assistant_score.py`: simulated assistant data and scoring | Built |
| `inferx/locomo.py`: LoCoMo loading and official-style scoring | Built |
| `inferx/scheduler.py`: when deferred work runs | Stub; idle is simulated at the end of each session |
| Long-history comparison (3 users, 112 days each) | Running |
| Final comparison on held-out data with a packet capture (`scripts/run_final.sh`) | Written, not run |
| Cloud mem0 and Letta baselines | Need a cloud API key and permission to send data out |
| Test messages written by someone other than the developer | Needed |
| Galaxy S25 | Not started |

## How it works

1. A new message arrives (calendar entry, text, order update).
2. The controller embeds it with MiniLM and decides:
   - **store:** worth remembering or not
   - **kind:** state (can change), event (happened once), core (about the owner) or none
   - **sensitive:** one-time codes and passwords are dropped before anything is saved
   - **replaces:** whether it changes an earlier saved message
3. Guards on those decisions:
   - Events are never replaced.
   - Only items of the same kind can replace each other.
   - A message about someone else ("my sister moved") can't replace the owner's own fact.
4. In idle time, the main LLM writes each saved message as a fact. For an update, it merges the
   old and new messages so unchanged details carry over. The old version is kept, marked
   outdated, and only shown for history questions.
5. At question time, the main LLM answers from retrieved memories. With `--cloud gate`, it first
   says whether those memories answer the question; only a "no" sends the question to the cloud.
   Memory work never calls the cloud.

## Setup (Windows ARM64)

Needs native ARM64 Python 3.12, the MSVC ARM64 build tools, the Windows SDK and Foundry Local.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
scripts\build_sqlite_vec.cmd
.venv\Scripts\python scripts\test_sqlite_vec.py
.venv\Scripts\python -m pytest tests
```

The mem0 port needs its own environment, because mem0's `grpcio` dependency has no Windows ARM64
wheel. `inferx/mem0_port.py` stubs the qdrant parts mem0 imports.

```
python -m venv .venv-mem0
.venv-mem0\Scripts\python -m pip install -r requirements.txt -r requirements-mem0.txt
.venv-mem0\Scripts\python -m pip install --no-deps mem0ai==2.2.1 qdrant-client==1.20.0
```

mem0's telemetry is switched off in `inferx/mem0_port.py` before mem0 is imported.

These files are downloaded, not stored in this repository:
- `third_party/sqlite-vec`: sqlite-vec v0.1.9 amalgamation and SQLite 3.49.1 amalgamation, for
  building `vec0.dll`
- `data/locomo/locomo10.json` from snap-research/locomo (CC BY-NC 4.0)
- `models/minilm/model_qint8_arm64.onnx` from sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0)
- `models/minilm_fp32/model.onnx`: the full-precision MiniLM from the same repository.
  `model_fixed.onnx` is the same model with its input shape fixed to 1 x 128, which the NPU needs.

## Run

Start Foundry Local and load phi:

```
foundry server start
foundry model load phi-3.5-mini
foundry server status
```

Use the Public API URL it prints as `<URL>`. `<TOKENIZER>` is phi's `tokenizer.json` in the
Foundry cache (`~/.foundry/cache/models/Microsoft/phi-3.5-mini-instruct-qnn-npu-3/v3/`).

Simulated assistant data, classifier training and one validation user:

```
.venv\Scripts\python scripts\generate_assistant_data.py
.venv\Scripts\python scripts\generate_assistant_data.py --long
.venv\Scripts\python scripts\train_memory_classifier.py
.venv\Scripts\python scripts\tune_memory_thresholds.py
.venv\Scripts\python scripts\check_classifier_npu.py
.venv\Scripts\python scripts\cloud_stub.py --port 8765 --log runs\cloud_stub.jsonl
.venv\Scripts\python scripts\run_locomo.py --url <URL> --model phi-3.5-mini-instruct-qnn-npu:3 --tokenizer <TOKENIZER> --data data\assistant\val.json --conversation 0 --cloud gate --systems plain_retrieval,full_context,local_memory,decoupled,trained
.venv\Scripts\python scripts\compare_assistant_runs.py runs\<run> [runs\<run> ...]
```

For mem0, run the same command with `.venv-mem0\Scripts\python` and `--systems mem0_2025_local`.
Leave out `--data` to run LoCoMo instead.

Each run writes `runs/<timestamp>/`: `config.json`, `summary.json`, one `.jsonl` per system with
every question, answer and score, and a log. The memory databases are left out of the repository.
`compare_assistant_runs.py` lets later runs replace earlier ones for the same system and user.

The cloud stand-in (`scripts/cloud_stub.py`) answers on 127.0.0.1 and logs each call and its size,
so cloud use can be counted without sending anything out. The cloud client refuses non-local
addresses unless `allow_remote` is set.

## Results so far

All on phi-3.5-mini (Foundry Local, NPU build). These are validation numbers, not thesis results:
the held-out test users and LoCoMo conversations 2 to 9 stay untouched until the final comparison.

### Personal assistant, 5 short validation users (268 questions)

| | Plain retrieval | Rules | Trained |
|---|---|---|---|
| Correct overall | 76.9% | 73.9% | **82.5%** |
| About something that changed (72 questions) | 44 | 43 | **58** |
| Replies giving the old value | 18 | 23 | **2** |
| Details that never changed (126) | 88.9% | 85.7% | 89.7% |
| History questions (12) | 58.3% | 50.0% | 41.7% |
| No-answer questions declined (25) | 23 | 23 | 23 |
| Questions sent to the cloud (cloud rule on) | n/a | n/a | 54 |

Trained minus plain retrieval: +5.6 points overall (95% interval 0.0 to +11.2), and +19.4 on
questions about something that changed (+5.6 to +33.3). All 52 one-time codes were dropped before
saving. Runs: plain and rules `20261105-220607` to `234602`, trained `20261108-141538` to `145144`.

Caveats:
- The changes behind these numbers (merged updates, outdated versions hidden, the third-party
  guard) were designed by reading errors on these same users, so these numbers favour the trained
  system. The long users and the test users are the fair checks.
- These users share wording with the training data.
- History questions got worse (8 of 12 before the changes, 5 of 12 after).
- Plain retrieval and rules were run before the cloud rule existed, so they have no cloud counts here.

### Personal assistant, long history (user 0 of 3, 112 days, 84 questions, cloud rule on)

| | Plain retrieval | Full context | Local memory | Rules | Trained |
|---|---|---|---|---|---|
| Correct overall | 66.7% | 9.5% | 66.7% | 69.0% | **73.8%** |
| About something that changed (29) | 15 | 4 | 15 | 17 | **19** |
| Replies giving the old value | 5 | 1 | 5 | 7 | **0** |
| Questions sent to the cloud | 29 | 39 | 84 | 50 | **15** |
| Bytes sent to the cloud | 77 KB | 274 KB | 228 KB | 149 KB | **50 KB** |

Run `20261108-145846`. One user is too few for a clear difference (trained minus plain retrieval
+7.1 points, interval -4.8 to +19.0). Users 1 and 2 are running.

- **Full context** fits only the last few days of a 112-day history into phi's window, which is
  why it scores so low.
- **Local memory** answers like plain retrieval but sends every question to the cloud, so it is the
  "memory on the device, answers from the cloud" reference point.

### Controller on the NPU

`scripts/check_classifier_npu.py` runs the decision heads, and then MiniLM as well, on the NPU
with CPU fallback disabled.
- Heads only on the NPU: every decision matched the CPU (146 of 146 messages, 715 of 715 pairs);
  probabilities differed by at most 0.002.
- MiniLM on the NPU too (fp16): decisions shift slightly. On `handwritten_check2.json`, worth
  remembering F1 went from 0.954 to 0.937, and updates found went from 4 of 7 to 5 of 7 with one
  wrong overwrite. One-time codes kept: 0 on both.

Results: `models/assistant_classifier/npu/check.json` and `check_all_npu.json`.

### mem0

- **mem0 2026** (mem0ai 2.2.1) can't run on this model: its extraction prompt alone is about 9,500
  tokens, and phi's NPU build fails above about 3,500 to 4,000. A memory system designed for cloud
  models doesn't fit an on-device one.
- **mem0 2025** (`mem0_2025_local`), the earlier algorithm with its own prompts, runs on phi. Its
  validation runs are queued.
- Cloud mem0 and Letta need a cloud API key and permission to send data out.

### Network

`scripts/watch_connections.ps1` logs connections from Python and Foundry Local without admin.
During the validation runs, Foundry Local kept an encrypted connection to a Microsoft address
(20.52.64.206, port 443). Its contents can't be seen, so the "no cloud" claim needs the full packet
capture in `scripts/capture_network.ps1`, which needs an administrator PowerShell.

### LoCoMo, conversation 1 (early, before the assistant work)

| | qwen plain | qwen rules | phi plain | phi rules |
|---|---|---|---|---|
| Score | 0.179 | 0.195 | 0.354 | 0.385 |
| Score without adversarial questions | 0.183 | 0.216 | 0.372 | 0.463 |
| Evidence recall | 0.56 | 0.628 | 0.541 | 0.664 |

Runs `20261105-164655` (qwen) and `20261105-195628` (phi). Comparing models on conversation 1
means it is no longer clean, so the final comparison uses conversations 2 to 9.

## Method notes

- Simulated data splits: 40 training, 5 validation and 10 test users. Test users hold out every
  third name, item and template.
- Assistant scoring counts a reply as correct if it contains the answer in any format. LoCoMo
  scoring follows LoCoMo's `task_eval/evaluation.py`. Evidence recall measures whether the
  evidence messages reached the prompt, whatever the model answered.
- Intervals are 95% bootstrap intervals over questions.
- `data/assistant/handwritten*.json` were written by the developer while building the system, so
  they are not independent. Messages written by someone else are still needed.
- Runs `20261108-135534`, `20261108-135621` and `20261108-141228` were stopped or smoke tests and
  are left out of comparisons.
- Foundry's `completion_tokens` over-reports, so `inferx/llm.py` counts output chunks itself.
- `scripts/` also holds the NPU and Foundry diagnostics used to debug the earlier driver failure.
