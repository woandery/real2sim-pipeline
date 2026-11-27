"""Small command surface for local preparation and GPU-host execution."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError

from .artifacts import read_json
from .config import ProjectConfig
from .errors import WorkflowError
from .models import Proposal
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video → evidence → reviewed Isaac Sim scenes")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Check configuration and available dependencies (no API charges)"
    )
    for command in ("prepare", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("video", type=Path)
        sub.add_argument("--scene", required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--retry-uncertain", action="store_true")
    sub = commands.add_parser(
        "approve", help="Attest to actual inspection of a qualified candidate"
    )
    sub.add_argument("run", type=Path)
    sub.add_argument("--reviewer", required=True)
    sub.add_argument("--observations", required=True)
    sub = commands.add_parser("record")
    sub.add_argument("run", type=Path)
    sub.add_argument("--retry-uncertain", action="store_true")
    sub = commands.add_parser(
        "native-smoke", help="Exercise native compilation/physics without API approval"
    )
    sub.add_argument("proposal", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--record", action="store_true")
    sub = commands.add_parser("status")
    sub.add_argument("run", type=Path)
    sub = commands.add_parser("validate", help="Validate a scene proposal without GPU or API")
    sub.add_argument("proposal", type=Path)
    sub = commands.add_parser(
        "api-smoke", help="One billable vision + structured-output API request"
    )
    sub.add_argument("image", type=Path)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--retry-uncertain", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = read_json(args.run / "status.json")
        elif args.command == "validate":
            proposal = Proposal.model_validate(read_json(args.proposal))
            result = {
                "valid": True,
                "scene_id": proposal.scene.scene_id,
                "assets": len(proposal.scene.assets),
            }
        else:
            config = ProjectConfig.load(args.config)
            pipeline = Pipeline(config)
            if args.command == "doctor":
                result = {
                    "model": config.api.model,
                    "review_model": config.api.review_model or config.api.model,
                    "api_base_url": config.api.base_url,
                    "api_key_present": bool(os.getenv(config.api.key_env)),
                    "runtime_python_resolves": bool(shutil.which(config.runtime.python)),
                    "eula_accepted": config.runtime.accept_eula,
                    "note": "Configuration checks only; does not prove API access, GPU availability or Isaac compatibility.",
                }
            elif args.command == "native-smoke":
                from .smoke import native_smoke

                result = native_smoke(
                    pipeline.backend,
                    Proposal.model_validate(read_json(args.proposal)),
                    args.output,
                    record=args.record,
                )
            elif args.command == "api-smoke":
                from .smoke import api_smoke

                result = api_smoke(
                    pipeline.provider, args.image, args.output, retry_uncertain=args.retry_uncertain
                )
            elif args.command == "prepare":
                from .artifacts import exclusive_lock

                with exclusive_lock(args.output / ".pipeline.lock"):
                    result = pipeline.prepare(args.video, args.output.resolve(), args.scene)
                result = {
                    "decoded_frames": result["decoded_frames"],
                    "sampled_frames": len(result["selected_frames"]),
                }
            elif args.command == "run":
                result = pipeline.run(
                    args.video, args.output, args.scene, retry_uncertain=args.retry_uncertain
                )
            elif args.command == "approve":
                result = pipeline.approve(args.run, args.reviewer, args.observations)
            else:
                result = pipeline.record(args.run, retry_uncertain=args.retry_uncertain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("status") == "native_checks_failed" else 0
    except (WorkflowError, ValidationError, ValueError, OSError, KeyError) as exc:
        print(f"real2sim: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
