"""Run bounded port checks from retained geometry, never historical approvals.

Usage (control Python, with this project's src on PYTHONPATH):
  python scripts/gpu_smoke_suite.py box --isaac-python /path/to/python --output /data/check
This script preserves all source geometry and selects a small diagnostic-state subset.
It does not call OpenAI, qualify reconstruction, or replay a robot policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_real2sim.artifacts import digest, read_json, write_json
from video_real2sim.backend import IsaacBackend
from video_real2sim.config import RuntimeSettings
from video_real2sim.models import Proposal
from video_real2sim.smoke import native_smoke

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def make_proposal(case: str) -> tuple[Proposal, dict]:
    if case == "box":
        path = EXAMPLES / "box-proposal.json"
        return Proposal.model_validate(read_json(path)), {
            "fixture": path.name,
            "sha256": digest(path),
            "state_indices": [0, 1],
        }
    selection = {
        "surface": ("ego_lid", [0, 4]),
        "freefall": ("ego_ball_free", [0, 11, 19]),
        "assembly": ("assembly_0037", [0, 1, 2, 6]),
        "interface": ("ego_allen_assembly", [0, 5, 10, 12]),
    }
    name, indices = selection[case]
    path = EXAMPLES / "historical" / f"{name}.json"
    scene = read_json(path)
    scene["diagnostic_states"] = [scene["diagnostic_states"][i] for i in indices]
    plan = dict(
        rest_checks=[],
        excluded_states=[
            dict(
                state_index=i,
                reason="Focused port diagnostic; this subset does not qualify every source state or its rest stability.",
            )
            for i in range(len(indices))
        ],
        sweeps=[],
        additional_requirements=[],
    )
    if case == "surface":
        plan["surface_checks"] = [
            dict(
                assets=[dict(asset_id="lid_left_small", slope_radii_m=[0.0295, 0.032])],
                rationale="One retained lid: two flat and six slope contacts, plus eight native queries.",
            )
        ]
    elif case == "freefall":
        plan["freefall_checks"] = [
            dict(
                asset_id="tennis_ball",
                support_asset_id="horizontal_surface",
                support_part_id="surface_slab",
                state_indices=[0, 1, 2],
                rationale="Original initial, high airborne and final held snapshots released at zero velocity.",
            )
        ]
    elif case == "assembly":
        plan["assembly_checks"] = [
            dict(
                state_index=1 if mode == "hold" else 2,
                mode=mode,
                rationale="Held parent only; measure every child connection and release.",
            )
            for mode in ("hold", "level_then_release")
        ]
    else:
        plan["interface_checks"] = [
            dict(
                tool_id="allen_key",
                fixture_id="upright_fixture",
                support_id="perforated_base_plate",
                fastener_ids=["fastener_a", "fastener_b"],
                cases=[
                    dict(state_index=1, target_id="fastener_a"),
                    dict(state_index=2, target_id="fastener_b"),
                ],
                rationale="Both sockets, both turn directions, native entry/withdrawal and target response.",
            )
        ]
    return Proposal.model_validate(dict(scene=scene, validation=plan)), {
        "fixture": path.name,
        "sha256": digest(path),
        "state_indices": indices,
        "scope": "All fixture geometry retained; bounded state subset. No historical acceptance transferred.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["box", "surface", "freefall", "assembly", "interface"])
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-lock", default="~/.cache/video-real2sim/gpu.lock")
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    proposal, provenance = make_proposal(args.case)
    write_json(args.output / "fixture-selection.json", provenance)
    backend = IsaacBackend(
        RuntimeSettings(
            python=args.isaac_python, device_lock=args.device_lock, accept_eula=args.accept_eula
        )
    )
    result = native_smoke(backend, proposal, args.output, record=args.record)
    print(result["status"], result["failures"], flush=True)
    return 0 if result["status"] == "native_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
