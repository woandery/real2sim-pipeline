"""Read-only audit of five completed native port checks with current acceptance code."""

from __future__ import annotations

import argparse
from pathlib import Path

from video_real2sim.artifacts import digest, read_json, verify_files, write_json
from video_real2sim.backend import check_probe, native_sources
from video_real2sim.config import RuntimeSettings
from video_real2sim.models import Proposal
from video_real2sim.specialized import (
    check_assembly,
    check_freefall,
    check_interface,
    check_surface,
)


def audit(root: Path) -> dict:
    result = {
        "scope": "Bounded native port checks; no live API, model/human reconstruction approval or policy execution.",
        "native_source_hashes": native_sources(),
        "cases": [],
    }
    for name in ("box", "freefall", "surface", "assembly", "interface"):
        folder = root / name
        proposal = Proposal.model_validate(read_json(folder / "proposal.json"))
        spec, usd = folder / "scene.json", folder / "render/scene.usda"
        smoke = read_json(folder / "smoke.json")
        assert smoke["status"] == "native_checks_passed", name
        jobs = list(folder.rglob("job.json"))
        assert jobs, name
        for job in jobs:
            receipt = read_json(job)
            assert receipt["status"] == "completed" and receipt["returncode"] == 0
            assert receipt["request"]["sources"] == native_sources(), (
                "Native implementation changed"
            )
            verify_files(job.parent, receipt["files"])
        failures, measurements = [], {}
        for probe in (folder / "physics").iterdir():
            if not probe.is_dir():
                continue
            if probe.name.startswith("surface-"):
                mode = probe.name.rsplit("-", 1)[1]
                raw = read_json(
                    probe / ("surface-query.json" if mode == "query" else "surface-probe.json")
                )
                inputs = dict(
                    source_spec_sha256=digest(spec),
                    source_usd_sha256=digest(usd),
                    probe_plan_sha256=digest(probe / "probe-plan.json"),
                )
                failures += check_surface(raw, proposal.validation.surface_checks[0], mode, inputs)
                measurements[probe.name] = dict(cases=len(raw["cases"]))
                continue
            raw = read_json(probe / "physics-probe.json")
            _, errors = check_probe(
                raw,
                dict(spec_sha256=digest(spec), usd_sha256=digest(usd)),
                RuntimeSettings(),
                enforce_rest=probe.name == "rest",
            )
            failures += errors
            measurements[probe.name] = dict(
                states=len(raw["states"]),
                native_steps=[s["integration_clock"]["native_steps"] for s in raw["states"]],
            )
            if probe.name.startswith("assembly-"):
                failures += check_assembly(
                    raw,
                    proposal.validation.assembly_checks[int(probe.name.split("-")[1])],
                    proposal.scene,
                )
            if probe.name.startswith("freefall-"):
                failures += check_freefall(
                    raw, proposal.validation.freefall_checks[0], proposal.scene
                )
            if probe.name.startswith("interface-"):
                failures += check_interface(
                    raw,
                    proposal.validation.interface_checks[0],
                    dict(probe_plan_sha256=digest(probe / "probe-plan.json")),
                )
        result["cases"].append(
            dict(
                case=name,
                passed=not failures,
                failures=failures,
                verified_jobs=len(jobs),
                measurements=measurements,
                spec_sha256=digest(spec),
                usd_sha256=digest(usd),
                smoke_sha256=digest(folder / "smoke.json"),
            )
        )
    result["passed"] = all(c["passed"] for c in result["cases"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results)
    write_json(args.output, result)
    print("AUDIT", result["passed"], [(r["case"], r["failures"]) for r in result["cases"]])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
