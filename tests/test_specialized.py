"""Adversarial acceptance tests: synthetic evidence never represents a GPU run."""

import copy
import json
from pathlib import Path

import pytest

from video_real2sim.artifacts import digest
from video_real2sim.errors import IntegrityError
from video_real2sim.models import (
    AssemblyCheck,
    InterfaceCheck,
    SceneSpec,
    SurfaceCheck,
    ValidationPlan,
)
from video_real2sim.specialized import check_assembly, check_interface, check_surface, surface_keys

HISTORICAL = Path(__file__).parents[1] / "examples/historical"


def test_historical_ten_geometry_contracts_and_source_manifest():
    manifest = json.loads((HISTORICAL / "manifest.json").read_text())
    assert len(manifest["scenes"]) == 10
    for item in manifest["scenes"]:
        path = HISTORICAL / item["fixture"]
        assert digest(path) == item["fixture_sha256"]
        scene = SceneSpec.model_validate_json(path.read_text())
        assert scene.scene_id == item["scene_id"]
        assert scene.diagnostic_states[0].poses == []


def surface_fixture(mode):
    plan = SurfaceCheck(
        assets=[{"asset_id": "renamed_lid", "slope_radii_m": [0.013, 0.021]}],
        rationale="Synthetic test",
    )
    report = dict(
        native_runtime=True,
        task_body_physics_changed=False,
        all_native_surfaces_within_0_25mm=True,
        all_surface_sites_passed=True,
        first_impulse_measurement=True,
        slope_site_coverage_enforced=True,
        cases=[
            dict(
                site=dict(lid=asset, name=name),
                correct_body=True,
                within_0_25mm=True,
                geometric_offset_m=0.00001,
                contact_found=True,
                measurement=dict(within_tolerance=True, contact_vs_visible_surface_error_m=0.00001),
            )
            for asset, name in sorted(surface_keys(plan, mode))
        ],
    )
    return plan, report


@pytest.mark.parametrize("mode,count", [("flat", 2), ("slopes", 6), ("query", 8)])
def test_surface_full_coverage_and_numerics(mode, count):
    plan, report = surface_fixture(mode)
    assert len(report["cases"]) == count
    assert check_surface(report, plan, mode, {}) == []
    bad = copy.deepcopy(report)
    if mode == "query":
        bad["cases"][0]["geometric_offset_m"] = float("nan")
    else:
        bad["cases"][0]["measurement"]["contact_vs_visible_surface_error_m"] = float("nan")
    assert check_surface(bad, plan, mode, {})
    report["cases"].pop()
    with pytest.raises(IntegrityError, match="cover"):
        check_surface(report, plan, mode, {})


def test_surface_no_duplicate_or_wrong_body():
    plan, report = surface_fixture("query")
    report["cases"][0]["correct_body"] = False
    assert check_surface(report, plan, "query", {})
    report["cases"].append(report["cases"][0])
    with pytest.raises(IntegrityError):
        check_surface(report, plan, "query", {})


def interface_fixture():
    plan = InterfaceCheck(
        tool_id="tool",
        fixture_id="frame",
        support_id="slab",
        fastener_ids=["a", "b"],
        cases=[dict(state_index=2, target_id="b")],
        rationale="Synthetic test",
    )
    rows = [
        dict(
            source_state_index=2,
            target="b",
            direction=d,
            passed=True,
            effective_offsets_verified=True,
            actual_screw_rotation_vector_deg=[0, 0, d * 10],
            target_translation_during_twist_m=0.0001,
            wall_contact_pairs={"wall": dict(max_impulse_Ns=0.001, min_separation_m=-0.00001)},
            entry_clearances={
                g: dict(intersecting_pairs=0)
                for g in ("walls", "floor", "fixture", "plate", "other_fastener")
            },
        )
        for d in (1, -1)
    ]
    report = dict(
        render_transform_checks=[dict(passed=True)],
        interface_validation=dict(
            checks=rows,
            all_checks_passed=True,
            source_geometry_or_filtering_overridden=False,
            other_body_pose_commands_after_reset=0,
            prescribed_body_ids=["tool"],
            render_bridge_enabled=True,
        ),
    )
    return plan, report


def test_interface_needs_both_directions_actual_motion_and_clearance():
    plan, report = interface_fixture()
    assert check_interface(report, plan, {}) == []
    rows = report["interface_validation"]["checks"]
    rows[0]["actual_screw_rotation_vector_deg"] = [0, 0, 0]
    assert check_interface(report, plan, {})
    rows[0]["actual_screw_rotation_vector_deg"] = [0, 0, 10]
    rows[0]["entry_clearances"].pop("other_fastener")
    assert check_interface(report, plan, {})
    rows.pop()
    with pytest.raises(IntegrityError, match="omitted"):
        check_interface(report, plan, {})


@pytest.mark.parametrize("mode", ["hold", "level_then_release"])
def test_assembly_requires_every_attachment_and_measured_release(mode):
    scene = SceneSpec.model_validate_json((HISTORICAL / "assembly_0037.json").read_text())
    plan = AssemblyCheck(state_index=0, mode=mode, rationale="Synthetic test")
    hold = mode == "hold"
    report = dict(
        render_transform_checks=[dict(passed=True)],
        release_clearance=dict(
            all_checks_passed=True, minimum_per_leg_m={a.child: 0.0001 for a in scene.attachments}
        ),
        attachment_retention=dict(
            all_checks_passed=True,
            leg_pose_commands_after_reset=0,
            prescribed_body_ids=[scene.attachments[0].parent],
            all_leg_bodies_dynamic_with_gravity=True,
            hold_only=hold,
            level_before_release=not hold,
            lift_m=plan.lift_m,
            checks=[
                dict(
                    id=a.id,
                    retained=True,
                    enabled_samples=50,
                    maximum_enabled_anchor_error_m=0.0001,
                    maximum_enabled_rotation_error_deg=0.01,
                    independent_after_release=not hold,
                    final_measurement=dict(
                        enabled=hold, anchor_error_m=0.1 if not hold else 0.0001
                    ),
                )
                for a in scene.attachments
            ],
        ),
    )
    assert check_assembly(report, plan, scene) == []
    rows = report["attachment_retention"]["checks"]
    rows[0]["maximum_enabled_anchor_error_m"] = 0.1
    assert check_assembly(report, plan, scene)
    rows.pop()
    with pytest.raises(IntegrityError, match="omitted"):
        check_assembly(report, plan, scene)


def test_freefall_contract_rejects_wrong_shape_and_support(proposal):
    plan = proposal.validation.model_dump()
    sphere = next(a for a in proposal.scene.assets if a.body_type == "dynamic")
    support = sphere.model_copy(update={"id": "test_support", "body_type": "static"})
    proposal.scene.assets.append(support)
    plan["freefall_checks"] = [
        dict(
            asset_id=sphere.id,
            support_asset_id=support.id,
            support_part_id=support.parts[0].id,
            state_indices=[0],
            rationale="Synthetic test",
        )
    ]
    with pytest.raises(ValueError, match="Sphere"):
        ValidationPlan.model_validate(plan).validate_scene(proposal.scene)


def test_surface_radii_checked_against_scene(proposal):
    plan = proposal.validation.model_dump()
    target = next(a for a in proposal.scene.assets if a.body_type == "dynamic")
    plan["surface_checks"] = [
        dict(
            assets=[dict(asset_id=target.id, slope_radii_m=[0.02, 0.01])],
            rationale="Synthetic test",
        )
    ]
    with pytest.raises(ValueError, match="increasing"):
        ValidationPlan.model_validate(plan).validate_scene(proposal.scene)


@pytest.mark.parametrize("case", ["box", "surface", "freefall", "assembly", "interface"])
def test_native_smoke_plans_are_valid_bounded_subsets(case):
    import runpy

    script = Path(__file__).parents[1] / "scripts/gpu_smoke_suite.py"
    proposal, origin = runpy.run_path(str(script))["make_proposal"](case)
    assert len(proposal.scene.diagnostic_states) == len(origin["state_indices"])
    assert origin["sha256"]
    if case != "box":
        assert not proposal.validation.rest_checks
        assert proposal.validation.excluded_states


def test_freefall_contacts_must_match_exact_body_and_support():
    from video_real2sim.models import FreefallCheck
    from video_real2sim.specialized import check_freefall

    scene = SceneSpec.model_validate_json((HISTORICAL / "ego_ball_free.json").read_text())
    plan = FreefallCheck(
        asset_id="tennis_ball",
        support_asset_id="horizontal_surface",
        support_part_id="surface_slab",
        state_indices=[0],
        rationale="Synthetic test",
    )
    support = "/World/Assets/horizontal_surface/p0_surface_slab"
    report = dict(
        release_clearance=dict(
            ground=dict(native_collider=support),
            all_checks_passed=True,
            minimum_per_leg_m={"tennis_ball": -0.000001},
        ),
        states=[
            dict(
                contact_summary=dict(
                    pairs={f"/World/Assets/tennis_ball <-> {support}": dict(max_impulse_Ns=0.01)}
                )
            )
        ],
    )
    assert check_freefall(report, plan, scene) == []
    report["states"][0]["contact_summary"]["pairs"] = {
        f"/World/Assets/tennis_ball_other <-> {support}": dict(max_impulse_Ns=0.01)
    }
    assert check_freefall(report, plan, scene)
    report["release_clearance"]["ground"]["native_collider"] = "/World/Ground"
    with pytest.raises(IntegrityError, match="different support"):
        check_freefall(report, plan, scene)
