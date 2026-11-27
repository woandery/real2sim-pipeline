"""OpenAI Responses adapter with strict outputs and content-bound call records."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel

from .artifacts import (
    canonical,
    digest,
    exclusive_lock,
    fingerprint,
    read_json,
    utc_now,
    write_json,
)
from .config import APISettings
from .errors import IntegrityError, NeedsAttention, WorkflowError

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider:
    """One request per attempt; ambiguous transport failures are never blindly retried.

    The SDK client can be injected for contract tests. Production uses only the
    configured key environment variable, never a ChatGPT/Codex login directory.
    """

    def __init__(self, settings: APISettings, client: OpenAI | None = None):
        self.settings = settings
        self._client = client

    def _connect(self) -> OpenAI:
        if self._client is None:
            key = os.environ.get(self.settings.key_env)
            if not key:
                raise WorkflowError(f"Set {self.settings.key_env} in the process environment")
            self._client = OpenAI(
                api_key=key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def call(
        self,
        stage: str,
        prompt: str,
        images: list[Path],
        schema: type[T],
        directory: Path,
        *,
        review: bool = False,
        retry_uncertain: bool = False,
    ) -> T:
        settings = self.settings
        if len(images) > settings.max_images:
            raise NeedsAttention(
                "Image count exceeds the configured request budget; no images dropped"
            )
        if sum(p.stat().st_size for p in images) > settings.max_image_bytes:
            raise NeedsAttention("Image bytes exceed the request budget; no images dropped")
        model = settings.review_model if review and settings.review_model else settings.model
        request = {
            "stage": stage,
            "model": model,
            "base_url": settings.base_url,
            "prompt": prompt,
            "schema": schema.model_json_schema(),
            "images": [{"name": p.name, "sha256": digest(p)} for p in images],
            "max_output_tokens": settings.max_output_tokens,
            "reasoning_effort": settings.reasoning_effort,
        }
        request_hash = fingerprint(request)
        # Base64 expands binary images by ~4/3. Leave margin for SDK JSON framing.
        encoded_bytes = sum(4 * ((p.stat().st_size + 2) // 3) for p in images)
        if encoded_bytes + len(canonical(request).encode()) > 48_000_000:
            raise NeedsAttention("Encoded vision request exceeds 48 MB; no images dropped")
        directory.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(directory / ".lock"):
            identity = directory / "request.json"
            if identity.exists() and read_json(identity) != request:
                raise IntegrityError("Call inputs changed; create a new stage/revision directory")
            result_file = directory / "result.json"
            receipt_file = directory / "receipt.json"
            if receipt_file.exists():
                receipt = read_json(receipt_file)
                if receipt.get("status") == "completed":
                    if (
                        receipt["request_sha256"] != request_hash
                        or digest(result_file) != receipt["result_sha256"]
                    ):
                        raise IntegrityError("Cached API response no longer matches its receipt")
                    return schema.model_validate(read_json(result_file))
                if receipt.get("status") in {"in_flight", "uncertain"} and not retry_uncertain:
                    raise NeedsAttention(
                        "Previous API request may have reached the server. Inspect its receipt; "
                        "--retry-uncertain explicitly permits another billable attempt."
                    )
            client = self._connect()
            content: list[dict] = [{"type": "input_text", "text": prompt}]
            for path in images:
                suffix = path.suffix.lower()
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
                    suffix
                )
                if mime is None:
                    raise WorkflowError(f"Unsupported image format: {suffix}")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,"
                        + base64.b64encode(path.read_bytes()).decode(),
                        "detail": "high",
                    }
                )
            write_json(identity, request)
            attempt = len(list(directory.glob("attempt-*.json"))) + 1
            receipt = {
                "status": "in_flight",
                "request_sha256": request_hash,
                "started_at": utc_now(),
                "attempt": attempt,
            }
            write_json(receipt_file, receipt)
            kwargs = {
                "model": model,
                "input": [{"role": "user", "content": content}],
                "text_format": schema,
                "max_output_tokens": settings.max_output_tokens,
                "store": False,
            }
            if settings.reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": settings.reasoning_effort}
            try:
                response = client.responses.parse(**kwargs)
                receipt["response_id"] = response.id
                receipt["request_id"] = getattr(response, "_request_id", None)
                receipt["usage"] = response.usage.model_dump() if response.usage else None
                receipt["response_status"] = response.status
                if response.status != "completed" or response.output_parsed is None:
                    raise WorkflowError(
                        "API response incomplete, refused or missing structured output"
                    )
                result = schema.model_validate(response.output_parsed.model_dump())
                write_json(result_file, result.model_dump())
                receipt.update(status="completed", result_sha256=digest(result_file))
            except APIConnectionError as exc:
                receipt.update(status="uncertain", error_type=type(exc).__name__)
                raise NeedsAttention(
                    "API transport interrupted; completion is unknown (see receipt)"
                ) from None
            except APIStatusError as exc:
                receipt.update(
                    status="failed", error_type=type(exc).__name__, http_status=exc.status_code
                )
                raise WorkflowError(
                    f"API rejected request with HTTP {exc.status_code}; see receipt"
                ) from None
            except Exception as exc:
                # The server may have billed a completed generation even if local
                # schema parsing failed. Explicit retry is required for this case.
                receipt.update(status="uncertain", error_type=type(exc).__name__)
                # Do not persist raw SDK exceptions: they may contain headers or request bodies.
                raise WorkflowError(f"Structured API stage failed: {type(exc).__name__}") from None
            finally:
                receipt["finished_at"] = utc_now()
                write_json(receipt_file, receipt)
                write_json(directory / f"attempt-{attempt:03d}.json", receipt)
            return result
