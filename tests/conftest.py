"""Shared pytest fixtures — notably a minimal mock Ollama server for the remote backend."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _MockOllamaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps(
                {"models": [{"name": name} for name in self.server.state["models"]]}
            ).encode()
            self._send(200, body)
        else:
            self._send(404, b"{}")

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.server.state["generate_calls"].append(payload)
        response_fn = self.server.state["generate_response_fn"]
        if response_fn:
            result = response_fn(payload)
            # response_fn may return just a body dict, or (status, body dict)
            # for per-request status codes (e.g. simulating one failing image).
            status, body = result if isinstance(result, tuple) else (200, result)
        else:
            status, body = self.server.state["generate_status"], self.server.state["generate_response"]
        self._send(status, json.dumps(body).encode())

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - matches BaseHTTPRequestHandler signature
        pass


class MockOllama:
    """A background HTTP server standing in for a real Ollama instance."""

    def __init__(self):
        self.server = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
        self.server.state = {
            "models": ["qwen2.5vl:7b"],
            "generate_calls": [],
            "generate_status": 200,
            "generate_response": {
                "response": "test description",
                "eval_count": 4,
                "done_reason": "stop",
            },
            "generate_response_fn": None,
        }
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    @property
    def generate_call_count(self) -> int:
        return len(self.server.state["generate_calls"])

    def set_models(self, models: list[str]) -> None:
        self.server.state["models"] = models

    def set_generate_response(self, response: dict, status: int = 200) -> None:
        self.server.state["generate_response"] = response
        self.server.state["generate_status"] = status

    def set_generate_response_fn(self, fn) -> None:
        """Supply a callable(payload) -> response dict, for responses that vary per request."""
        self.server.state["generate_response_fn"] = fn

    def shutdown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)


@pytest.fixture
def mock_ollama():
    server = MockOllama()
    yield server
    server.shutdown()
