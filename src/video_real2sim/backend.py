"""Subprocess isolation for Isaac Sim and fail-closed native evidence checks."""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
from pathlib import Path

from .artifacts import (
    digest,
    file_record,
    fingerprint,
    read_json,
    utc_now,
    verify_files,
    write_json,
)
from .config import RuntimeSettings
from .errors import IntegrityError, NeedsAttention, WorkflowError
from .models import Proposal
from .specialized import check_assembly, check_freefall, run_interface_checks, run_surface_checks

NATIVE = Path(__file__).parent / "native"


def native_sources() -> dict[str, str]:
    return {path.name: digest(path) for path in sorted(NATIVE.glob("*.py"))}


class IsaacBackend:
    """Run on the GPU host itself; cloud-specific lifecycle is outside this package."""

    def __init__(self, settings: RuntimeSettings, key_env: str = "OPENAI_API_KEY"):
        self.settings = settings
        self.key_env = key_env

    def _job(self, script: str, arguments: list[str], output: Path, inputs: dict) -> None:
        if not self.settings.accept_eula:
            raise NeedsAttention("Set runtime.accept_eula=true after accepting the Isaac Sim EULA")
        output.mkdir(parents=True, exist_ok=True)
        request = {
            "script": script,
            "arguments": arguments,
            "inputs": inputs,
            "sources": native_sources(),
            "runtime": self.settings.model_dump(),
        }
        identity = fingerprint(request)
        receipt_path = output / "job.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["request_sha256"] != identity:
                raise IntegrityError("Native job inputs changed; use a new revision")
            if receipt["status"] == "completed":
                verify_files(output, receipt["files"])
                return
            raise NeedsAttention(
                f"Native job is {receipt['status']}; inspect {receipt_path}. "
                "No automatic restart of a potentially live or failed simulator."
            )
        for path in NATIVE.glob("*.py"):
            shutil.copy2(path, output / path.name)
        command = [self.settings.python, str(output / script), *arguments]
        receipt = {
            "request_sha256": identity,
            "request": request,
            "started_at": utc_now(),
            "status": "starting",
            "command": command,
        }
        write_json(receipt_path, receipt)
        env = dict(os.environ)
        # Native workers deliberately assert physical invariants. Never inherit -O.
        env.pop("PYTHONOPTIMIZE", None)
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["REAL2SIM_DEVICE_LOCK"] = str(Path(self.settings.device_lock).expanduser().resolve())
        # The API credential is never needed by the GPU child or written to its files.
        for name in list(env):
            if name == self.key_env or name.endswith("API_KEY") or name.startswith("OPENAI_"):
                env.pop(name)
        try:
            with (output / "stdout.log").open("w") as stdout:
                proc = subprocess.Popen(
                    command,
                    env=env,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                receipt.update(status="running", pid=proc.pid)
                write_json(receipt_path, receipt)
                try:
                    code = proc.wait(timeout=self.settings.timeout_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    receipt.update(status="timed_out", returncode=proc.returncode)
                    raise WorkflowError(
                        "Native job exceeded its runtime budget; logs retained"
                    ) from None
                receipt["returncode"] = code
                if code != 0 or (output / "failure.json").exists():
                    raise WorkflowError(f"Native job failed ({code}); inspect stdout.log")
            if any(digest(output / name) != sha for name, sha in request["sources"].items()):
                raise IntegrityError("Frozen native source changed during execution")
            receipt["status"] = "completed"
            receipt["files"] = [
                file_record(p, output)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p != receipt_path and "__pycache__" not in p.parts
            ]
        except Exception:
            if receipt["status"] != "timed_out":
                receipt["status"] = "failed"
            raise
        finally:
            receipt["finished_at"] = utc_now()
            write_json(receipt_path, receipt)

    def render(self, spec: Path, output: Path, *, record: bool = False) -> dict:
        source_sha256 = digest(spec)
        arguments = [str(spec.resolve()), str(output.resolve())]
        if record:
            arguments.append("--record")
        self._job("isaac_worker.py", arguments, output, {"spec_sha256": source_sha256})
        report = read_json(output / "runtime.json")
        if (
            report.get("source_spec_sha256") != source_sha256
            or digest(spec) != source_sha256
            or report.get("native_runtime") is not True
        ):
            raise IntegrityError("Native render does not match its scene specification")
        if report.get("frozen_worker_sha256") != digest(output / "isaac_worker.py"):
            raise IntegrityError("Native compiler receipt mismatch")
        if not list(output.glob("review-*.jpg")) or not (output / "scene.usda").is_file():
            raise IntegrityError("Native render missing source-state views or USD")
        if any(not r["finite"] for r in report["rest_diagnostics"]):
            raise WorkflowError("Nonfinite native body state")
        if any(c["anchor_error_m"] > 0.005 for c in report["initial_joint_anchor_checks"]):
            raise WorkflowError("Initial joint anchors do not coincide")
        return report

    def validate(self, proposal: Proposal, spec: Path, rendered: Path, output: Path) -> dict:
        plan = proposal.validation
        if plan.additional_requirements:
            raise NeedsAttention(
                "Task needs additional physics checks: " + "; ".join(plan.additional_requirements)
            )
        probes = (
            [("rest", ["--indices", *[str(r.state_index) for r in plan.rest_checks]])]
            if plan.rest_checks
            else []
        )
        if proposal.scene.joints and plan.rest_checks:
            probes.append(
                (
                    "passive",
                    ["--indices", *[str(r.state_index) for r in plan.rest_checks], "--passive"],
                )
            )
        for i, sweep in enumerate(plan.sweeps):
            probes.append(
                (
                    f"sweep-{i:02d}",
                    [
                        "--indices",
                        str(sweep.state_index),
                        "--steps",
                        "600",
                        "--sweep-joint",
                        sweep.joint_id,
                        "--sweep-end",
                        str(sweep.end_position),
                        "--enable-joint-collisions",
                    ],
                )
            )
        for i, check in enumerate(plan.assembly_checks):
            steps = max(900, 360 + 120 * len(proposal.scene.attachments) + 180)
            probes.append(
                (
                    f"assembly-{i:02d}",
                    [
                        "--indices",
                        str(check.state_index),
                        "--steps",
                        str(steps),
                        "--attachment-retention",
                        "--attachment-lift",
                        str(check.lift_m),
                        "--attachment-hold-only"
                        if check.mode == "hold"
                        else "--attachment-level-before-release",
                        "--enable-ccd",
                        "--substeps",
                        "10",
                        "--kinematic-targets",
                        "--sync-physics-render",
                        "--mirror-kinematic-usd",
                    ],
                )
            )
        for i, check in enumerate(plan.freefall_checks):
            support = next(a for a in proposal.scene.assets if a.id == check.support_asset_id)
            part_index = next(
                k for k, part in enumerate(support.parts) if part.id == check.support_part_id
            )
            support_path = f"/World/Assets/{support.id}/p{part_index}_{check.support_part_id}"
            probes.append(
                (
                    f"freefall-{i:02d}",
                    [
                        "--indices",
                        *map(str, check.state_indices),
                        "--steps",
                        "180",
                        "--enable-ccd",
                        "--substeps",
                        "10",
                        "--focus-asset",
                        check.asset_id,
                        "--support-collider",
                        support_path,
                        "--sync-physics-render",
                    ],
                )
            )
        reports, checks, images = {}, [], []
        for label, options in probes:
            directory = output / label
            arguments = [
                str(spec.resolve()),
                str((rendered / "scene.usda").resolve()),
                str(directory.resolve()),
                *options,
            ]
            inputs = {"spec_sha256": digest(spec), "usd_sha256": digest(rendered / "scene.usda")}
            self._job("physics_probe.py", arguments, directory, inputs)
            raw = read_json(directory / "physics-probe.json")
            if label.startswith("sweep-"):
                expected_indices = [plan.sweeps[int(label.split("-")[1])].state_index]
            elif label.startswith("assembly-"):
                expected_indices = [plan.assembly_checks[int(label.split("-")[1])].state_index]
            elif label.startswith("freefall-"):
                expected_indices = plan.freefall_checks[int(label.split("-")[1])].state_indices
            else:
                expected_indices = [r.state_index for r in plan.rest_checks]
            expected_names = [proposal.scene.diagnostic_states[i].name_zh for i in expected_indices]
            if [s["name"] for s in raw.get("states", [])] != expected_names:
                raise IntegrityError("Physics probe did not cover the requested states")
            for asset in proposal.scene.assets:
                if asset.body_type == "dynamic":
                    body = raw.get("body_physics", {}).get(asset.id, {})
                    external_hold = (
                        label.startswith("assembly-")
                        and asset.id == proposal.scene.attachments[0].parent
                    )
                    if (
                        body.get("kinematic") is not external_hold
                        or body.get("gravity_disabled") is not False
                        or body.get("rigid_body_enabled") is not True
                        or body.get("collider_count", 0) < 1
                    ):
                        raise IntegrityError(
                            f"Dynamic asset lost its physical properties: {asset.id}"
                        )
            summary, failures = check_probe(
                raw, inputs, self.settings, enforce_rest=label == "rest"
            )
            reports[label] = summary
            if label.startswith("sweep-"):
                target = plan.sweeps[int(label.split("-")[1])]
                joint = next(j for j in proposal.scene.joints if j.id == target.joint_id)
                if raw.get("joint_types", {}).get(joint.id) != joint.type:
                    raise IntegrityError("Measured sweep units do not match the requested joint")
            if label.startswith("freefall-"):
                failures.extend(
                    check_freefall(
                        raw, plan.freefall_checks[int(label.split("-")[1])], proposal.scene
                    )
                )
            if label.startswith("assembly-"):
                failures.extend(
                    check_assembly(
                        raw, plan.assembly_checks[int(label.split("-")[1])], proposal.scene
                    )
                )
            checks.extend(f"{label}: {failure}" for failure in failures)
            for item in raw["native_rest_images"]:
                image = directory / item["file"]
                if (
                    not image.resolve().is_relative_to(directory.resolve())
                    or digest(image) != item["sha256"]
                ):
                    raise IntegrityError("Native physics image changed")
            panels = list(directory.glob("rest-review-*.jpg"))
            if not panels:
                raise IntegrityError("Missing physics review contact sheets")
            images.extend(panels)
        surfaces = run_surface_checks(self._job, plan.surface_checks, spec, rendered, output)
        interfaces = run_interface_checks(
            self._job,
            lambda raw, inputs: check_probe(raw, inputs, self.settings, enforce_rest=False),
            plan.interface_checks,
            spec,
            rendered,
            output,
        )
        reports.update(surfaces["reports"])
        reports.update(interfaces["reports"])
        checks.extend(surfaces["failures"])
        checks.extend(interfaces["failures"])
        result = {
            "reports": reports,
            "failures": checks,
            "passed": not checks,
            "images": [file_record(p, output) for p in sorted(images)]
            + surfaces["images"]
            + interfaces["images"],
            "source_spec_sha256": digest(spec),
            "source_usd_sha256": digest(rendered / "scene.usda"),
            "scope": "Native rest and driven-joint diagnostics, not robot policy execution or calibrated material identification.",
        }
        write_json(output / "summary.json", result)
        return result


def check_probe(
    raw: dict, inputs: dict, settings: RuntimeSettings, *, enforce_rest: bool
) -> tuple[dict, list[str]]:
    """Conservative numerical checks supplement (never replace) visual model review."""
    if raw.get("native_runtime") is not True:
        raise IntegrityError("Missing native physics execution evidence")
    if (
        raw.get("source_spec_sha256") != inputs["spec_sha256"]
        or raw.get("source_usd_sha256") != inputs["usd_sha256"]
    ):
        raise IntegrityError("Physics evidence belongs to a different scene")
    if not raw.get("states") or not raw.get("native_rest_images"):
        raise IntegrityError("Empty physics states or images")
    failures = []
    if raw.get("contact_capture_errors") or raw.get("image_capture_errors"):
        failures.append("Native contact/image capture failed")
    states = []
    for state in raw["states"]:
        clock = state.get("integration_clock", {})
        if (
            clock.get("all_explicit_steps_checked") is not True
            or not all(
                math.isfinite(clock.get(k, float("nan"))) and clock.get(k, 0) > 0
                for k in (
                    "native_steps",
                    "expected_native_steps",
                    "native_duration_seconds",
                    "expected_duration_seconds",
                )
            )
            or clock.get("native_steps", 0) != clock.get("expected_native_steps", -1)
            or abs(
                clock.get("native_duration_seconds", -1) - clock.get("expected_duration_seconds", 1)
            )
            > 1e-5
        ):
            failures.append("Native integration clock does not match requested steps")
        if not state.get("body_drift") or not state.get("trace"):
            raise IntegrityError("Missing per-body trajectory or drift evidence")
        for name, body in state["body_drift"].items():
            if not body["finite"] or any(
                not math.isfinite(body[k]) for k in ("translation_m", "rotation_deg")
            ):
                failures.append(f"Nonfinite state: {name}")
            elif enforce_rest and (
                body["translation_m"] > settings.max_rest_translation_m
                or body["rotation_deg"] > settings.max_rest_rotation_deg
            ):
                failures.append(f"Rest state moved beyond configured tolerance: {name}")
        separation = state["contact_summary"].get("min_separation_m")
        if separation is not None and (
            not math.isfinite(separation) or separation < -settings.max_penetration_m
        ):
            failures.append("Contact penetration exceeds configured tolerance")
        compact = {k: v for k, v in state.items() if k != "trace"}
        trace = state["trace"]
        compact["final_actual_bodies"] = trace[-1]["bodies"]
        compact["trace_samples"] = len(trace)
        sweep = state.get("continuous_joint_sweep")
        if sweep:
            joint = sweep["joint_id"]
            measured = [row.get("measured_joint_positions", {}).get(joint) for row in trace]
            if any(value is None or not math.isfinite(value) for value in measured):
                failures.append("Continuous sweep lacks measured joint positions")
            else:
                # Revolute tolerance is in degrees; linear tolerance in metres.
                units = raw.get("joint_types", {}).get(joint)
                tolerance = 2.0 if units == "revolute" else 0.002
                compact["measured_sweep_start"] = measured[0]
                compact["measured_sweep_final"] = measured[-1]
                if abs(measured[-1] - sweep["end"]) > tolerance:
                    failures.append("Continuous sweep did not reach its requested target")
        states.append(compact)
    summary = {k: v for k, v in raw.items() if k not in {"states", "native_rest_images"}}
    summary["states"] = states
    return summary, failures
