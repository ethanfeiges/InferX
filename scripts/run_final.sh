#!/usr/bin/env bash
# The final comparison. Run it once, after all tuning is finished, because it uses the held-out data:
# the 10 simulated test users and LoCoMo conversations 2 to 9.
#
# For the network evidence, start it from an administrator PowerShell through capture_network.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\capture_network.ps1 -Out runs\network_final -Command "bash scripts/run_final.sh <url>"
#
# Usage: bash scripts/run_final.sh <Foundry Public API URL> [cloud stand-in URL]
# Start the cloud stand-in first: .venv/Scripts/python scripts/cloud_stub.py
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:?give the Foundry Public API URL}"
CLOUD="${2:-http://127.0.0.1:8765/v1}"
MODEL="phi-3.5-mini-instruct-qnn-npu:3"
TOKENIZER="$HOME/.foundry/cache/models/Microsoft/phi-3.5-mini-instruct-qnn-npu-3/v3/tokenizer.json"
SYSTEMS="plain_retrieval,full_context,local_memory,decoupled,trained"
run() {  # python, data, conversation, systems
  "$1" scripts/run_locomo.py --url "$URL" --model "$MODEL" --tokenizer "$TOKENIZER" --data "$2" \
    --conversation "$3" --cloud gate --cloud-url "$CLOUD" --systems "$4" --npu-controller < /dev/null
}
for user in $(seq 0 9); do
  run .venv/Scripts/python data/assistant/test.json "$user" "$SYSTEMS"
  run .venv-mem0/Scripts/python data/assistant/test.json "$user" mem0_2025_local
done
for conversation in $(seq 2 9); do
  run .venv/Scripts/python data/locomo/locomo10.json "$conversation" "$SYSTEMS"
  run .venv-mem0/Scripts/python data/locomo/locomo10.json "$conversation" mem0_2025_local
done
echo FINAL_DONE
