"""Dependency-free HTTP serving for speculative generation."""

from __future__ import annotations

import json
import random
import threading
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends import SamplingBackend
from .config import DecodeConfig
from .decoder import SpeculativeDecoder
from .models import ProbabilityModel
from .tokenizers import encode_prompt


class RequestValidationError(ValueError):
    """Raised when an HTTP generation request is malformed."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Network and request limits for the HTTP server."""

    host: str = "127.0.0.1"
    port: int = 8000
    max_body_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("port must be an integer")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if (
            isinstance(self.max_body_bytes, bool)
            or not isinstance(self.max_body_bytes, int)
            or self.max_body_bytes < 1
        ):
            raise ValueError("max_body_bytes must be a positive integer")


class GenerationService:
    """Validate requests and serialize access to request-local model caches."""

    def __init__(
        self,
        draft_model: ProbabilityModel,
        target_model: ProbabilityModel,
        tokenizer: Any,
        config: DecodeConfig,
        sampling_backend: SamplingBackend,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.config = config
        self.sampling_backend = sampling_backend
        self._model_lock = threading.Lock()

    @staticmethod
    def _request_value(payload: dict[str, Any], name: str, expected: type, default: Any) -> Any:
        value = payload.get(name, default)
        if expected is int and isinstance(value, bool):
            raise RequestValidationError(f"{name} must be an integer")
        if not isinstance(value, expected):
            raise RequestValidationError(f"{name} must be a {expected.__name__}")
        return value

    def generate(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RequestValidationError("request body must be a JSON object")
        prompt = self._request_value(payload, "prompt", str, None)
        seed = self._request_value(payload, "seed", int, 7)
        chat = self._request_value(payload, "chat", bool, False)
        max_new_tokens = self._request_value(
            payload,
            "max_new_tokens",
            int,
            self.config.max_new_tokens,
        )
        if not prompt:
            raise RequestValidationError("prompt must not be empty")
        if not 0 <= max_new_tokens <= self.config.max_new_tokens:
            raise RequestValidationError(
                f"max_new_tokens must be between 0 and {self.config.max_new_tokens}"
            )

        try:
            prompt_tokens = encode_prompt(self.tokenizer, prompt, chat=chat)
        except (KeyError, TypeError, ValueError) as error:
            raise RequestValidationError(str(error)) from error
        decode_config = replace(self.config, max_new_tokens=max_new_tokens)
        decoder = SpeculativeDecoder(
            self.draft_model,
            self.target_model,
            decode_config,
            rng=random.Random(seed),
            sampling_backend=self.sampling_backend,
        )
        with self._model_lock:
            result = decoder.generate(prompt_tokens)
        text = self.tokenizer.decode(
            result.generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "text": text,
            "token_ids": list(result.generated_tokens),
            "stats": asdict(result.stats),
            "sampling_backend": decoder.sampling_backend.name,
        }


class GenerationHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying one shared generation service."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: GenerationService,
        *,
        max_body_bytes: int,
    ) -> None:
        self.service = service
        self.max_body_bytes = max_body_bytes
        super().__init__(address, GenerationRequestHandler)


class GenerationRequestHandler(BaseHTTPRequestHandler):
    """JSON API exposing health and generation endpoints."""

    server: GenerationHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/generate":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            content_length = -1
        if content_length < 0:
            self._write_json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length required"})
            return
        if content_length > self.server.max_body_bytes:
            self.close_connection = True
            self._write_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            response = self.server.service.generate(payload)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RequestValidationError,
        ) as error:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        # This is the transport boundary: model failures become a generic 500,
        # never invented output or internal implementation details.
        except Exception:  # noqa: BLE001
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "generation failed"})
            return
        self._write_json(HTTPStatus.OK, response)


def create_server(
    service: GenerationService,
    config: ServerConfig | None = None,
) -> GenerationHTTPServer:
    """Create a configured server without starting its blocking loop."""

    server_config = config or ServerConfig()
    return GenerationHTTPServer(
        (server_config.host, server_config.port),
        service,
        max_body_bytes=server_config.max_body_bytes,
    )


def serve(service: GenerationService, config: ServerConfig | None = None) -> None:
    """Serve requests until interrupted."""

    with create_server(service, config) as server:
        server.serve_forever()
