"""A local stand-in for a cloud model. It answers nothing and records every call.

Usage: python scripts/cloud_stub.py [--port 8765] [--log runs/cloud_stub.jsonl]

It listens on 127.0.0.1 only. Each request is logged with its size, so cloud calls and bytes can
be counted without sending data anywhere or paying for a provider.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPLY = "No answer: this is the local cloud stand-in."

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--log", default=str(Path(__file__).resolve().parent.parent / "runs/cloud_stub.jsonl"))
args = parser.parse_args()
log_path = Path(args.log)
log_path.parent.mkdir(parents=True, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(size)
        try:
            question = json.loads(body)["messages"][-1]["content"].rsplit("Question:", 1)[-1].strip()[:200]
        except (ValueError, KeyError, IndexError):
            question = None
        reply = json.dumps({"choices": [{"message": {"role": "assistant", "content": REPLY}}]}).encode()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": time.time(), "bytes_received": size, "question": question}) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *_):
        pass


print(f"cloud stand-in on http://127.0.0.1:{args.port}/v1, logging to {log_path}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
