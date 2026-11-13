"""Local UI and streaming bridge for the Ollama API."""

from __future__ import annotations

import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
PORT = int(os.environ.get("VISUALIZER_PORT", "8765"))


def ollama_json(path: str, payload: dict | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[visualizer] {self.address_string()} - {fmt % args}")

    def send_json(self, status: int, value: dict):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            try:
                result = ollama_json("/api/tags")
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(503, {"error": f"Cannot reach Ollama at {OLLAMA_URL}: {exc}"})
            return

        if parsed.path == "/api/model":
            name = parse_qs(parsed.query).get("name", ["gemma3:4b"])[0]
            try:
                result = ollama_json("/api/show", {"model": name})
                info = result.get("model_info", {})

                def info_value(suffix: str):
                    return next((v for k, v in info.items() if k.endswith(suffix)), None)

                self.send_json(200, {
                    "name": name,
                    "family": (result.get("details") or {}).get("family", "unknown"),
                    "parameter_size": (result.get("details") or {}).get("parameter_size", "unknown"),
                    "quantization": (result.get("details") or {}).get("quantization_level", "unknown"),
                    "layers": info_value(".block_count") or 34,
                    "embedding": info_value(".embedding_length") or 2560,
                    "mlp_width": info_value(".feed_forward_length") or 10240,
                    "attention_heads": info_value(".head_count") or 8,
                    "kv_heads": info_value(".head_count_kv") or 4,
                    "head_width": info_value(".key_length") or 256,
                    "parameter_count": info_value(".parameter_count") or 3880101888,
                    "context": info_value(".context_length") or 131072,
                    "vision": "clip" in ((result.get("details") or {}).get("families") or []),
                    # Gemma 3 4B's released vision configuration. Ollama exposes
                    # the GGUF text configuration, but not these vision dimensions.
                    "vision_layers": 27,
                    "vision_embedding": 1152,
                    "vision_mlp_width": 4304,
                    "vision_attention_heads": 16,
                    "vision_image_size": 896,
                    "vision_patch_size": 14,
                    "architecture_source": "https://huggingface.co/unsloth/gemma-3-4b-it/blob/main/config.json",
                })
            except Exception as exc:
                self.send_json(502, {"error": f"Could not inspect {name}: {exc}"})
            return

        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20_000_000:
                self.send_json(413, {"error": "Request is too large"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not payload.get("model") or not isinstance(payload.get("prompt"), str):
                self.send_json(400, {"error": "A model and prompt are required"})
                return
            request = Request(
                f"{OLLAMA_URL}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            upstream = urlopen(request, timeout=3600)
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                detail = str(exc)
            self.send_json(exc.code, {"error": detail or str(exc)})
            return
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self.send_json(502, {"error": f"Ollama request failed: {exc}"})
            return
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            with upstream:
                while True:
                    chunk = upstream.readline()
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Gemma 3 visualizer: http://127.0.0.1:{PORT}")
    print(f"Ollama API: {OLLAMA_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down visualizer")
    finally:
        server.server_close()
