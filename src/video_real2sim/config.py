"""Explicit configuration; no host names, account IDs or secrets in source."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class APISettings(Settings):
    model: str = Field(min_length=1)
    review_model: str | None = None
    base_url: str = "https://api.openai.com/v1"
    key_env: str = Field(default="OPENAI_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    timeout_seconds: float = Field(default=600, gt=0, le=3600)
    max_output_tokens: int = Field(default=24000, ge=1024, le=128000)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    max_images: int = Field(default=100, ge=1, le=200)
    max_image_bytes: int = Field(default=40_000_000, ge=1000, le=45_000_000)

    @model_validator(mode="after")
    def validate_url(self) -> APISettings:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("API base_url must be an HTTPS endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API URL must not contain credentials, queries or fragments")
        return self


class SamplingSettings(Settings):
    uniform_frames: int = Field(default=48, ge=2, le=240)
    event_frames: int = Field(default=24, ge=0, le=120)
    event_separation_seconds: float = Field(default=0.18, gt=0)
    refinement_rounds: int = Field(default=2, ge=0, le=5)
    frames_per_interval: int = Field(default=6, ge=2, le=24)
    max_refinement_frames: int = Field(default=48, ge=2, le=120)


class ReviewSettings(Settings):
    max_revisions: int = Field(default=4, ge=1, le=20)
    passes_required: int = Field(default=2, ge=2, le=5)
    require_human_approval: bool = True


class RuntimeSettings(Settings):
    python: str = "python"
    timeout_seconds: int = Field(default=3600, ge=30, le=86400)
    device_lock: str = "~/.cache/video-real2sim/gpu.lock"
    accept_eula: bool = False
    max_penetration_m: float = Field(default=0.001, gt=0, le=0.01)
    max_rest_translation_m: float = Field(default=0.01, gt=0, le=0.05)
    max_rest_rotation_deg: float = Field(default=10, gt=0, le=30)


class ProjectConfig(Settings):
    api: APISettings
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @classmethod
    def load(cls, path: Path) -> ProjectConfig:
        with path.open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))
