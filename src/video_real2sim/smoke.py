"""Small billable vision/schema check, deliberately separate from scene acceptance."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .artifacts import exclusive_lock, read_json, write_json
from .backend import IsaacBackend
from .errors import IntegrityError
from .models import Proposal
from .provider import OpenAIProvider
from .video import inspect_video


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible_objects: list[str]
    spatial_description: str
    uncertainties: list[str]


def api_smoke(provider: OpenAIProvider, image: Path, output: Path, *, retry_uncertain=False):
    result = provider.call(
        "api-smoke",
        "Describe only the visible objects and spatial arrangement in this image. Record uncertainties. This is a vision/structured-output connectivity check, not reconstruction or acceptance.",
        [image],
        VisualObservation,
        output,
        retry_uncertain=retry_uncertain,
    )
    return {
        "status": "api_smoke_completed",
        "observation": result.model_dump(),
        "scope": "One vision/structured response; no physics or reconstruction qualification.",
    }


def native_smoke(backend: IsaacBackend, proposal: Proposal, output: Path, *, record=False):
    """Exercise fixed native code; never fabricate visual-review approval."""
    output = output.resolve()
    with exclusive_lock(output / ".native-smoke.lock"):
        proposal_path = output / "proposal.json"
        if proposal_path.exists() and read_json(proposal_path) != proposal.model_dump():
            raise IntegrityError("Smoke proposal changed; use a new output directory")
        write_json(proposal_path, proposal.model_dump())
        spec = output / "scene.json"
        write_json(spec, proposal.scene.compiler_dict())
        backend.render(spec, output / "render")
        physics = backend.validate(proposal, spec, output / "render", output / "physics")
        result = {
            "status": "native_checks_passed" if physics["passed"] else "native_checks_failed",
            "failures": physics["failures"],
            "scope": "Native execution and numerical gates only; no model/human visual approval or reconstruction qualification.",
        }
        if record and physics["passed"]:
            backend.render(spec, output / "recording", record=True)
            result["video_integrity"] = inspect_video(
                output / "recording/isaac-scene-orbit.mp4",
                output / "recording/inspection",
                len(proposal.scene.diagnostic_states),
            )
        write_json(output / "smoke.json", result)
        return result
