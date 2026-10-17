"""
Sonnet OpenAI-compatible bridge — lets the Codex harness (which has no
native Anthropic protocol) drive the exact same Claude Sonnet-4-6 backbone
used by the Claude Code baseline and ProClaim's own production runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import litellm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(message)s")
logger = logging.getLogger(__name__)

# Vertex AI by default; SONNET_BRIDGE_MODEL=anthropic/claude-sonnet-4-6 uses the direct
# Anthropic API instead. Vertex project comes from VERTEXAI_PROJECT; region us-east5.
TARGET_MODEL = os.environ.get("SONNET_BRIDGE_MODEL", "vertex_ai/claude-sonnet-4-6")
VERTEX_LOCATION = os.environ.get("SONNET_BRIDGE_VERTEX_LOCATION", "us-east5")
# Sent as extra_body on every call, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}'
# to switch GLM 5's reasoning off (codex_baseline.py --bridge-extra-body).
EXTRA_BODY = json.loads(os.environ["SONNET_BRIDGE_EXTRA_BODY"]) if os.environ.get("SONNET_BRIDGE_EXTRA_BODY") else None
_FORWARDED_FIELDS = (
    "input", "instructions", "tools", "tool_choice", "max_output_tokens",
    "temperature", "previous_response_id", "parallel_tool_calls",
)
_DROPPED_TOOL_TYPES = {"namespace"}


def _translate_tools(tools: list[dict]) -> list[dict]:
    translated = []
    for tool in tools:
        t = tool.get("type")
        if t in _DROPPED_TOOL_TYPES:
            continue
        if t == "web_search":
            translated.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 8})
            continue
        translated.append(tool)
    return translated


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": "claude-sonnet-4-6",
                    "object": "model",
                    "owned_by": "anthropic-via-litellm-bridge",
                }],
            })
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/responses":
            self._send_json(404, {"error": {"message": "not found"}})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": {"message": f"invalid JSON: {exc}"}})
            return

        kwargs = {k: body[k] for k in _FORWARDED_FIELDS if k in body}
        kwargs.setdefault("max_output_tokens", 8000)
        stream = bool(body.get("stream", False))

        if "tools" in kwargs:
            kwargs["tools"] = _translate_tools(kwargs["tools"])

        try:
            if TARGET_MODEL.startswith("vertex_ai/"):
                kwargs["vertex_location"] = VERTEX_LOCATION
            if EXTRA_BODY:
                kwargs["extra_body"] = EXTRA_BODY
            response = litellm.responses(model=TARGET_MODEL, **kwargs)
        except Exception as exc:
            logger.error("litellm.responses failed: %s", exc)
            self._send_json(500, {"error": {"message": str(exc), "type": "bridge_error"}})
            return

        data = response.model_dump()

        debug_dir = os.environ.get("SONNET_BRIDGE_DEBUG_DIR")
        if debug_dir:
            import uuid
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            dump_path = Path(debug_dir) / f"{time.time():.6f}_{uuid.uuid4().hex[:8]}.json"
            with open(dump_path, "w") as f:
                json.dump({"request": body, "response_pre_strip": data}, f, indent=2, default=str)
        data["output"] = [
            o for o in data.get("output", [])
            if not (o.get("type") == "function_call" and str(o.get("call_id", "")).startswith("srvtoolu_"))
        ]
        data["object"] = "response"

        if not stream:
            self._send_json(200, data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event: dict) -> None:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))

        emit({"type": "response.created", "response": {**data, "output": [], "status": "in_progress"}})

        for idx, item in enumerate(data.get("output", [])):
            item_id = item.get("id", f"item_{idx}")
            in_progress_item = {**item, "status": "in_progress"}
            emit({"type": "response.output_item.added", "output_index": idx, "item": in_progress_item})

            if item.get("type") == "message":
                for cidx, part in enumerate(item.get("content", [])):
                    empty_part = {**part, "text": ""}
                    emit({"type": "response.content_part.added", "item_id": item_id,
                          "output_index": idx, "content_index": cidx, "part": empty_part})
                    text = part.get("text") or ""
                    emit({"type": "response.output_text.delta", "item_id": item_id,
                          "output_index": idx, "content_index": cidx, "delta": text})
                    emit({"type": "response.output_text.done", "item_id": item_id,
                          "output_index": idx, "content_index": cidx, "text": text})
                    emit({"type": "response.content_part.done", "item_id": item_id,
                          "output_index": idx, "content_index": cidx, "part": part})
            elif item.get("type") == "function_call":
                args = item.get("arguments") or ""
                emit({"type": "response.function_call_arguments.delta", "item_id": item_id,
                      "output_index": idx, "delta": args})
                emit({"type": "response.function_call_arguments.done", "item_id": item_id,
                      "output_index": idx, "arguments": args})

            emit({"type": "response.output_item.done", "output_index": idx, "item": item})

        emit({"type": "response.completed", "response": data})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4180)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if TARGET_MODEL.startswith("vertex_ai/"):
        if not os.environ.get("VERTEXAI_PROJECT"):
            logger.warning("VERTEXAI_PROJECT not set — Vertex AI calls to Sonnet will fail.")
    elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GLM_API_KEY")):
        logger.warning("ANTHROPIC_API_KEY not set — litellm calls to Sonnet will fail.")

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    logger.info("Sonnet bridge listening on http://%s:%d (target model: %s, region: %s, extra_body: %s)",
                args.host, args.port, TARGET_MODEL, VERTEX_LOCATION, EXTRA_BODY)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
