"""Bounded, task-specific evidence checks beyond rest and joint motion."""

from __future__ import annotations

import math
from pathlib import Path

from .artifacts import digest, file_record, read_json, resolve_member, write_json
from .errors import IntegrityError
from .models import AssemblyCheck, FreefallCheck, InterfaceCheck, SceneSpec, SurfaceCheck


def check_assembly(report: dict, plan: AssemblyCheck, scene: SceneSpec) -> list[str]:
    """Recompute the acceptance gate from every attachment's measurements."""
    retained = report.get("attachment_retention", {})
    rows = retained.get("checks", [])
    if {r["id"] for r in rows} != {a.id for a in scene.attachments} or len(rows) != len(
        scene.attachments
    ):
        raise IntegrityError("Assembly evidence omitted or duplicated a connection")
    hold = plan.mode == "hold"
    parent = scene.attachments[0].parent
    failures = []
    if (
        retained.get("all_checks_passed") is not True
        or retained.get("leg_pose_commands_after_reset") != 0
        or retained.get("prescribed_body_ids") != [parent]
        or retained.get("all_leg_bodies_dynamic_with_gravity") is not True
        or retained.get("hold_only") is not hold
        or retained.get("level_before_release") is not (not hold)
        or retained.get("lift_m") != plan.lift_m
    ):
        failures.append("Assembly physical roles or diagnostic mode differ from plan")
    for row in rows:
        if (
            row.get("retained") is not True
            or row.get("enabled_samples", 0) < 1
            or not 0 <= row.get("maximum_enabled_anchor_error_m", math.inf) < 0.002
            or not 0 <= row.get("maximum_enabled_rotation_error_deg", math.inf) < 2
        ):
            failures.append("Attachment did not remain connected within tolerance")
        final = row.get("final_measurement", {})
        if not hold and (
            row.get("independent_after_release") is not True
            or final.get("enabled") is not False
            or not 0.02 < final.get("anchor_error_m", -1) < math.inf
        ):
            failures.append("Attachment lacks measured independent release")
    clearance = report.get("release_clearance", {})
    minima = clearance.get("minimum_per_leg_m", {})
    if (
        clearance.get("all_checks_passed") is not True
        or set(minima) != {a.child for a in scene.attachments}
        or any(not math.isfinite(value) or value < -0.001 for value in minima.values())
    ):
        failures.append("Assembly child/ground clearance is missing or exceeds 1 mm")
    if not report.get("render_transform_checks") or not all(
        c["passed"] for c in report["render_transform_checks"]
    ):
        failures.append("Assembly render/physics transform consistency failed")
    return failures


def check_freefall(report: dict, plan: FreefallCheck, scene: SceneSpec) -> list[str]:
    clearance = report.get("release_clearance", {})
    support_asset = next(a for a in scene.assets if a.id == plan.support_asset_id)
    index = next(i for i, p in enumerate(support_asset.parts) if p.id == plan.support_part_id)
    support = f"/World/Assets/{support_asset.id}/p{index}_{plan.support_part_id}"
    if clearance.get("ground", {}).get("native_collider") != support:
        raise IntegrityError("Freefall report used a different support collider")
    failures = []
    minima = clearance.get("minimum_per_leg_m", {})
    if (
        clearance.get("all_checks_passed") is not True
        or set(minima) != {plan.asset_id}
        or any(not math.isfinite(x) or x < -0.001 for x in minima.values())
    ):
        failures.append("Freefall lacks finite, in-tolerance measured clearance")
    target = f"/World/Assets/{plan.asset_id}"
    for state in report["states"]:
        pairs = state["contact_summary"]["pairs"]
        if not any(
            support in pair.split(" <-> ")
            and any(
                actor == target or actor.startswith(target + "/") for actor in pair.split(" <-> ")
            )
            and value.get("max_impulse_Ns", 0) > 0
            for pair, value in pairs.items()
        ):
            failures.append("Freefall lacks positive contact with its specified support")
    return failures


def surface_keys(plan: SurfaceCheck, mode: str) -> set[tuple[str, str]]:
    names = (
        ["central", "shoulder"]
        if mode == "flat"
        else [f"slope{slope}-az{angle}" for slope in (1, 2) for angle in (15, 135, 255)]
    )
    if mode == "query":
        names = ["central", "shoulder"] + [
            f"slope{slope}-az{angle}" for slope in (1, 2) for angle in (15, 135, 255)
        ]
    return {(asset.asset_id, name) for asset in plan.assets for name in names}


def check_surface(report: dict, plan: SurfaceCheck, mode: str, inputs: dict) -> list[str]:
    for key, expected in inputs.items():
        if report.get(key) != expected:
            raise IntegrityError(f"Surface report input mismatch: {key}")
    if (
        report.get("native_runtime") is not True
        or report.get("task_body_physics_changed") is not False
    ):
        raise IntegrityError("Surface evidence must use unchanged native task bodies")
    cases = report.get("cases", [])
    actual = [(c["site"]["lid"], c["site"]["name"]) for c in cases]
    if len(actual) != len(set(actual)) or set(actual) != surface_keys(plan, mode):
        raise IntegrityError("Surface probe did not cover every requested asset/site")
    failures = []
    if mode == "query":
        if report.get("all_native_surfaces_within_0_25mm") is not True:
            failures.append("Native surface query did not match visible geometry")
        for case in cases:
            if (
                case.get("correct_body") is not True
                or case.get("within_0_25mm") is not True
                or not abs(case.get("geometric_offset_m", float("inf"))) < 0.00025
            ):
                failures.append(
                    "Surface query hit is missing, on another body or outside tolerance"
                )
    else:
        if report.get("callback_errors") or report.get("collision_cooking_overrides"):
            failures.append("Surface contact capture failed or collision geometry was overridden")
        if report.get("all_surface_sites_passed") is not True:
            failures.append("Native surface contact tests failed")
        if mode == "slopes" and (
            report.get("first_impulse_measurement") is not True
            or report.get("slope_site_coverage_enforced") is not True
        ):
            failures.append("Slope validation must inspect the first positive contact")
        for case in cases:
            measurement = case.get("measurement") or {}
            tolerance = 0.00025 if mode == "slopes" else 0.0005
            if (
                case.get("contact_found") is not True
                or measurement.get("within_tolerance") is not True
                or not abs(measurement.get("contact_vs_visible_surface_error_m", float("inf")))
                < tolerance
            ):
                failures.append("Contact evidence missing or outside visible surface tolerance")
    return failures


def run_surface_checks(
    job, checks: list[SurfaceCheck], spec: Path, rendered: Path, output: Path
) -> dict:
    reports, failures, images = {}, [], []
    for index, check in enumerate(checks):
        for mode in ("flat", "slopes", "query"):
            label = f"surface-{index:02d}-{mode}"
            folder = output / label
            plan_path = folder / "probe-plan.json"
            write_json(plan_path, check.model_dump())
            inputs = {
                "source_spec_sha256": digest(spec),
                "source_usd_sha256": digest(rendered / "scene.usda"),
                "probe_plan_sha256": digest(plan_path),
            }
            options = ["--plan", str(plan_path.resolve())]
            script = "surface_query_worker.py" if mode == "query" else "surface_worker.py"
            if mode != "query":
                options += ["--sites", mode, "--physics-dt", "0.001", "--first-impulse"]
            job(
                script,
                [
                    str(spec.resolve()),
                    str((rendered / "scene.usda").resolve()),
                    str(folder.resolve()),
                    *options,
                ],
                folder,
                inputs,
            )
            report = read_json(
                folder / ("surface-query.json" if mode == "query" else "surface-probe.json")
            )
            failures.extend(
                f"{label}: {item}" for item in check_surface(report, check, mode, inputs)
            )
            compact = {k: v for k, v in report.items() if k not in {"cases", "visible_sections"}}
            compact["cases"] = [
                {k: v for k, v in case.items() if k != "trace"} for case in report["cases"]
            ]
            reports[label] = compact
            if mode != "query":
                for case in report["cases"]:
                    if digest(resolve_member(folder, case["image"])) != case["image_sha256"]:
                        raise IntegrityError("Surface inspection image changed")
                panels = sorted(folder.glob("rest-review-*.jpg"))
                if not panels:
                    raise IntegrityError("Surface contact images missing")
                images.extend(file_record(p, output) for p in panels)
    return {"reports": reports, "failures": failures, "images": images}


def check_interface(report: dict, plan: InterfaceCheck, inputs: dict) -> list[str]:
    for key, expected in inputs.items():
        if report.get(key) != expected:
            raise IntegrityError(f"Interface input mismatch: {key}")
    validation = report.get("interface_validation", {})
    rows = validation.get("checks", [])
    actual = [(row["source_state_index"], row["target"], row["direction"]) for row in rows]
    expected = {
        (case.state_index, case.target_id, direction)
        for case in plan.cases
        for direction in (1, -1)
    }
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise IntegrityError("Interface test omitted a requested state/target/direction")
    failures = []
    if (
        validation.get("all_checks_passed") is not True
        or validation.get("source_geometry_or_filtering_overridden") is not False
        or validation.get("other_body_pose_commands_after_reset") != 0
        or validation.get("prescribed_body_ids") != [plan.tool_id]
    ):
        failures.append("Interface validation failed or moved bodies other than the declared tool")
    if validation.get("render_bridge_enabled") is not True or not report.get(
        "render_transform_checks"
    ):
        failures.append("Interface render/physics transform evidence missing")
    elif not all(row["passed"] for row in report["render_transform_checks"]):
        failures.append("Interface rendered transforms differ from measured physics")
    for row in rows:
        rotation = row["actual_screw_rotation_vector_deg"]
        pairs = row.get("wall_contact_pairs", {})
        entry = row.get("entry_clearances", {})
        groups = {"walls", "floor", "fixture", "plate", "other_fastener"}
        if set(entry) != groups or any(m.get("intersecting_pairs") != 0 for m in entry.values()):
            failures.append("Interface entry clearance is incomplete or intersects other geometry")
        if (
            row.get("passed") is not True
            or row.get("effective_offsets_verified") is not True
            or not 5 < row["direction"] * rotation[2] < 18
            or (rotation[0] ** 2 + rotation[1] ** 2) ** 0.5 >= 1
            or not 0 <= row["target_translation_during_twist_m"] < 0.001
            or not any(p["max_impulse_Ns"] > 0 for p in pairs.values())
            or any(
                not math.isfinite(p["min_separation_m"]) or p["min_separation_m"] <= -0.00025
                for p in pairs.values()
            )
        ):
            failures.append(
                "Interface target response/contact did not satisfy the declared diagnostic"
            )
    return failures


def run_interface_checks(
    job, inspect, checks: list[InterfaceCheck], spec: Path, rendered: Path, output: Path
) -> dict:
    reports, failures, images = {}, [], []
    for index, check in enumerate(checks):
        label = f"interface-{index:02d}"
        folder = output / label
        plan = folder / "probe-plan.json"
        write_json(plan, check.model_dump())
        inputs = {
            "source_spec_sha256": digest(spec),
            "source_usd_sha256": digest(rendered / "scene.usda"),
            "probe_plan_sha256": digest(plan),
        }
        job(
            "interface_worker.py",
            [
                str(spec.resolve()),
                str((rendered / "scene.usda").resolve()),
                str(folder.resolve()),
                "--plan",
                str(plan.resolve()),
                "--render-bridge",
            ],
            folder,
            inputs,
        )
        raw = read_json(folder / "physics-probe.json")
        errors = check_interface(raw, check, inputs)
        summary, physical_errors = inspect(
            raw,
            {
                "spec_sha256": inputs["source_spec_sha256"],
                "usd_sha256": inputs["source_usd_sha256"],
            },
        )
        # Snapshots contain large per-collider data; full records remain on disk.
        for state in summary["states"]:
            state.pop("interface_snapshots", None)
        summary["interface_validation"].pop("collider_inventory", None)
        reports[label] = summary
        failures.extend(f"{label}: {error}" for error in [*errors, *physical_errors])
        for item in raw["native_rest_images"]:
            if digest(resolve_member(folder, item["file"])) != item["sha256"]:
                raise IntegrityError("Interface image changed")
        panels = sorted(folder.glob("rest-review-*.jpg"))
        if not panels:
            raise IntegrityError("Interface inspection images missing")
        images.extend(file_record(p, output) for p in panels)
    return {"reports": reports, "failures": failures, "images": images}
