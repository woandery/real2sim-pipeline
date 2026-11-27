"""Actual continuous native socket entry/turn/withdrawal; no screw pose control."""

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("spec")
p.add_argument("usd")
p.add_argument("output")
p.add_argument("--render-bridge", action="store_true")
p.add_argument("--plan", required=True)
a = p.parse_args()
probe_plan = json.loads(Path(a.plan).read_text())
tool_id = probe_plan["tool_id"]
fixture_id = probe_plan["fixture_id"]
support_id = probe_plan["support_id"]
fastener_ids = probe_plan["fastener_ids"]
source = Path(a.spec).resolve()
usd = Path(a.usd).resolve()
out = Path(a.output).resolve()
out.mkdir(exist_ok=True)
spec = json.loads(source.read_text())
lock_path = Path(os.environ["REAL2SIM_DEVICE_LOCK"]).expanduser()
lock_path.parent.mkdir(parents=True, exist_ok=True)
lock = lock_path.open("a+")
fcntl.flock(lock, fcntl.LOCK_EX)
if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
    raise RuntimeError("Explicit Isaac Sim EULA acceptance is required")
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "width": 960,
        "height": 540,
        "renderer": "PathTracing",
        "samples_per_pixel_per_frame": 16,
        "max_bounces": 12,
        "extra_args": ["--allow-root"],
    }
)
from importlib.metadata import version

import carb
import numpy as np
import omni.usd
from box_geometry import distance
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.sensors.camera import Camera
from omni.physx import get_physx_simulation_interface
from PIL import Image, ImageDraw
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

if a.render_bridge:
    from render_sync import audit as audit_render
    from render_sync import prepare as prepare_render


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def pose(body):
    position, q = body.get_world_pose()
    return dict(
        position=np.asarray(position).tolist(),
        orientation_wxyz=np.asarray(q).tolist(),
        linear_velocity=np.asarray(body.get_linear_velocity()).tolist(),
        angular_velocity=np.asarray(body.get_angular_velocity()).tolist(),
    )


def smooth(u):
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def offset_subset_matches(actual, shapes):
    """Match declared per-part offsets without assuming native shape-array order."""
    remaining = [float(x) for row in actual for x in row]
    if len(remaining) != len(shapes):
        return False
    for shape in shapes:
        value = shape["authored_contact_offset_m"]
        if value is None:
            continue
        index = next((i for i, x in enumerate(remaining) if abs(x - value) < 1e-10), None)
        if index is None:
            return False
        remaining.pop(index)
    return True


try:
    context = omni.usd.get_context()
    context.open_stage(str(usd))
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    cache = UsdGeom.XformCache()
    inventory = {}
    body_physics = {}
    paths = {}
    for asset in spec["assets"]:
        name = asset["id"]
        path = "/World/Assets/" + name
        prim = stage.GetPrimAtPath(path)
        inverse = cache.GetLocalToWorldTransform(prim).GetInverse()
        shapes = []
        for child in Usd.PrimRange(prim):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            parameters = {}
            size = None
            if child.IsA(UsdGeom.Cube):
                size = float(UsdGeom.Cube(child).GetSizeAttr().Get())
                dimensions = [size] * 3
                parameters = dict(size=size)
                measurement_scope = "Exact analytic native Cube"
            elif child.IsA(UsdGeom.Sphere):
                radius = float(UsdGeom.Sphere(child).GetRadiusAttr().Get())
                dimensions = [2 * radius] * 3
                parameters = dict(radius=radius)
                measurement_scope = "Conservative enclosing box for native Sphere; positive box distance proves separation, zero is inconclusive"
            elif child.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(child)
                radius = float(cylinder.GetRadiusAttr().Get())
                height = float(cylinder.GetHeightAttr().Get())
                axis = cylinder.GetAxisAttr().Get()
                assert axis == "Z"
                dimensions = [2 * radius, 2 * radius, height]
                parameters = dict(radius=radius, height=height, axis=axis)
                measurement_scope = "Conservative enclosing box for native Cylinder; positive box distance proves separation, zero is inconclusive"
            else:
                raise ValueError(
                    "Unsupported native collider "
                    + str(child.GetPath())
                    + " "
                    + child.GetTypeName()
                )
            local = cache.GetLocalToWorldTransform(child) * inverse
            center = local.TransformAffine(Gf.Vec3d(0, 0, 0))
            edges = np.column_stack(
                [
                    local.TransformDir(
                        Gf.Vec3d(*(dimensions[j] if i == j else 0.0 for i in range(3)))
                    )
                    for j in range(3)
                ]
            )
            api = PhysxSchema.PhysxCollisionAPI(child)
            collision = UsdPhysics.CollisionAPI(child)
            shapes.append(
                dict(
                    path=str(child.GetPath()),
                    part_id=child.GetName().split("_", 1)[1],
                    shape_type="native " + child.GetTypeName(),
                    native_parameters=parameters,
                    measurement_scope=measurement_scope,
                    center_local=list(center),
                    edges_local=edges.tolist(),
                    local_to_body_matrix=np.asarray(local).tolist(),
                    collision_enabled=bool(collision.GetCollisionEnabledAttr().Get()),
                    authored_contact_offset_m=finite(api.GetContactOffsetAttr().Get()),
                    authored_rest_offset_m=finite(api.GetRestOffsetAttr().Get()),
                    filtered_pairs=[
                        str(p)
                        for p in UsdPhysics.FilteredPairsAPI(child)
                        .GetFilteredPairsRel()
                        .GetTargets()
                    ]
                    if child.HasAPI(UsdPhysics.FilteredPairsAPI)
                    else [],
                )
            )
        if shapes:
            inventory[name] = shapes
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            paths[name] = path
            rb = UsdPhysics.RigidBodyAPI(prim)
            body_physics[name] = dict(
                kinematic=bool(rb.GetKinematicEnabledAttr().Get()),
                gravity_disabled=bool(
                    PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
                ),
                rigid_body_enabled=bool(rb.GetRigidBodyEnabledAttr().Get()),
                filtered_pairs=[
                    str(p)
                    for p in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                ]
                if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
                else [],
            )
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    assert body_physics[tool_id]["kinematic"]
    assert all(
        not body_physics[n]["kinematic"] and not body_physics[n]["gravity_disabled"]
        for n in [fixture_id, *fastener_ids]
    )
    assert not any(
        x["filtered_pairs"] or not x["collision_enabled"]
        for shapes in inventory.values()
        for x in shapes
    )
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / 240,
        rendering_dt=1 / 30,
        physics_prim_path="/World/PhysicsScene",
    )
    bodies = {
        n: world.scene.add(SingleRigidPrim(prim_path=path, name=n)) for n, path in paths.items()
    }
    camera = Camera(prim_path="/World/InterfaceCamera", resolution=(960, 540))
    native_images = []
    frames = []
    callback_errors = []
    phase = {"case": None, "step": 0}
    raw_count = 0
    render_audits = []
    raw = gzip.open(out / "contacts.jsonl.gz", "wt")

    def contacts(headers, data):
        global raw_count
        if phase["case"] is None:
            return
        try:
            for header in headers:
                actors = [
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                ]
                if paths[tool_id] not in actors:
                    continue
                colliders = [
                    str(PhysicsSchemaTools.intToSdfPath(getattr(header, k)))
                    for k in ["collider0", "collider1"]
                ]
                for i in range(
                    header.contact_data_offset, header.contact_data_offset + header.num_contact_data
                ):
                    c = data[i]
                    impulse = float(np.linalg.norm(c.impulse))
                    separation = float(c.separation)
                    key = " <-> ".join(colliders)
                    stats = phase["stats"]
                    pair = stats["pairs"].setdefault(
                        key, dict(contacts=0, min_separation_m=0.0, max_impulse_Ns=0.0)
                    )
                    pair["contacts"] += 1
                    pair["min_separation_m"] = min(pair["min_separation_m"], separation)
                    pair["max_impulse_Ns"] = max(pair["max_impulse_Ns"], impulse)
                    stats["contacts"] += 1
                    stats["min_separation_m"] = min(stats["min_separation_m"], separation)
                    if phase["step"] >= 780:
                        stats["late_min_separation_m"] = min(
                            stats["late_min_separation_m"], separation
                        )
                    record = dict(
                        case=phase["case"],
                        step=phase["step"],
                        actors=actors,
                        colliders=colliders,
                        position=list(c.position),
                        normal=list(c.normal),
                        impulse_Ns=list(c.impulse),
                        separation_m=separation,
                    )
                    if (
                        impulse > 1e-12
                        and "_" + probe_plan["socket_wall_prefix"] in key
                        and "/" + phase["target"] + "/" in key
                    ):
                        if "first_positive_contact" not in pair:
                            pair["first_positive_contact"] = record
                        if phase["first_wall_step"] is None:
                            phase["first_wall_step"] = phase["step"]
                    raw.write(json.dumps(record) + "\n")
                    raw_count += 1
        except Exception as error:
            callback_errors.append(repr(error))

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)

    def capture(case, step, target):
        before = (world.current_time, world.current_time_step_index)
        bridge_before = prepare_render(stage, bodies, paths, [tool_id]) if a.render_bridge else None

        def v(x):
            return Gf.Vec3d(*map(float, x))

        eye = target + np.array([0.03, -0.045, 0.035])
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/InterfaceCamera"))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(v(eye), v(target), Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/InterfaceCamera"))
        cam.GetHorizontalApertureAttr().Set(36.0)
        cam.GetVerticalApertureAttr().Set(20.25)
        cam.GetFocalLengthAttr().Set(25.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.001, 100))
        for _ in range(20):
            world.render()
        assert before == (world.current_time, world.current_time_step_index)
        if a.render_bridge:
            result = dict(case=case, step=step, **audit_render(stage, bodies, paths, bridge_before))
            render_audits.append(result)
            (out / "render-transform-checks.json").write_text(
                json.dumps(render_audits, indent=2) + "\n"
            )
            assert result["passed"], (
                "Interface physics-to-render bridge audit failed; retain actual audit"
            )
        rgba = np.asarray(camera.get_rgba())
        assert rgba.shape == (540, 960, 4) and rgba[..., :3].std() > 2
        frame = rgba[..., :3].astype(np.uint8)
        path = out / f"rest-interface-{case}-step{step:04d}.png"
        Image.fromarray(frame).save(path)
        native_images.append(dict(file=path.name, sha256=digest(path), case=case, step=step))
        frames.append((f"{case}: actual native step{step}, key-only control", frame))

    def geometry(actual):
        result = {}
        for name, shapes in inventory.items():
            if name in actual:
                position = actual[name]["position"]
                rot = rotation(actual[name]["orientation_wxyz"])
            else:
                asset = next(x for x in spec["assets"] if x["id"] == name)
                position = asset["position"]
                rot = Rotation.from_euler("xyz", asset["rotation_deg"], degrees=True)
            result[name] = [
                {
                    **shape,
                    "center_world": (rot.apply(shape["center_local"]) + position).tolist(),
                    "edges_world": (rot.as_matrix() @ np.asarray(shape["edges_local"])).tolist(),
                }
                for shape in shapes
            ]
        return result

    results = []
    checks = []
    offset_readbacks = []
    for state_index, target_name in [
        (item["state_index"], item["target_id"]) for item in probe_plan["cases"]
    ]:
        state = spec["diagnostic_states"][state_index]
        for direction in [1, -1]:
            case = f"S{state_index:02d}-{target_name}-" + (
                "positive" if direction == 1 else "negative"
            )
            phase["case"] = None
            world.reset()
            camera.initialize()
            requested = {
                x["id"]: dict(position=x["position"], rotation_deg=x["rotation_deg"])
                for x in spec["assets"]
            }
            requested.update({x["asset_id"]: x for x in state["poses"]})
            held = requested[tool_id]
            held_rotation = Rotation.from_euler("xyz", held["rotation_deg"], degrees=True)
            held_position = np.array(held["position"])
            for name, body in bodies.items():
                item = requested[name]
                q = Rotation.from_euler("xyz", item["rotation_deg"], degrees=True).as_quat()[
                    [3, 0, 1, 2]
                ]
                position = np.array(item["position"]) + (
                    [0, 0, probe_plan["insertion_lift_m"]] if name == tool_id else np.zeros(3)
                )
                body.set_world_pose(position=position, orientation=q)
                if not body_physics[name]["kinematic"]:
                    body.set_linear_velocity(np.zeros(3))
                    body.set_angular_velocity(np.zeros(3))
            views = {
                n: world.physics_sim_view.create_rigid_body_view(path) for n, path in paths.items()
            }
            assert all(view.count == 1 for view in views.values())
            offsets = {
                n: dict(
                    contact_offsets_m=np.asarray(view.get_contact_offsets()).tolist(),
                    rest_offsets_m=np.asarray(view.get_rest_offsets()).tolist(),
                )
                for n, view in views.items()
            }
            offset_readbacks.append(
                dict(
                    case=case,
                    bodies=offsets,
                    shape_order_note="Native per-body shape counts and explicitly authored offset multisets checked without a path-order assumption. Undeclared native defaults are reported but not attributed to particular parts.",
                )
            )
            stats = dict(contacts=0, min_separation_m=0.0, late_min_separation_m=0.0, pairs={})
            phase.update(case=case, stats=stats, target=target_name, first_wall_step=None)
            trace = []
            snapshots = {}
            initial = {n: pose(body) for n, body in bodies.items()}
            clock_origin = float(world.current_time)
            step_origin = int(world.current_time_step_index)
            for step in range(1560):
                insertion = smooth((step - 120) / 240)
                angle = direction * 15 * smooth((step - 480) / 600)
                withdrawal = smooth((step - 1200) / 240)
                position = held_position + np.array(
                    [0, 0, probe_plan["insertion_lift_m"] * (1 - insertion + withdrawal)]
                )
                rot = held_rotation * Rotation.from_euler("z", angle, degrees=True)
                q = rot.as_quat()
                command = np.concatenate([position, q]).astype(np.float32).reshape(1, 7)
                views[tool_id].set_kinematic_targets(command, np.array([0], dtype=np.int32))
                phase["step"] = step
                world.step(render=False)
                assert int(world.current_time_step_index) - step_origin == step + 1
                assert abs(float(world.current_time) - clock_origin - (step + 1) / 240) < 1e-5
                actual = {n: pose(body) for n, body in bodies.items()}
                trace.append(
                    dict(
                        step=step,
                        time_s=(step + 1) / 240,
                        bodies=actual,
                        commanded_tool_position=position.tolist(),
                        commanded_tool_local_yaw_deg=angle,
                        measured_joint_positions={},
                        measured_joint_anchor_errors_m={},
                    )
                )
                first_wall = phase["first_wall_step"] == step
                if step in [119, 479, 1199, 1559] or first_wall:
                    boxes = geometry(actual)
                    long = [
                        box
                        for box in boxes[tool_id]
                        if box["part_id"].startswith(probe_plan["tool_part_prefix"])
                    ]
                    groups = dict(
                        walls=[
                            box
                            for box in boxes[target_name]
                            if box["part_id"].startswith(probe_plan["socket_wall_prefix"])
                        ],
                        floor=[
                            box
                            for box in boxes[target_name]
                            if box["part_id"].startswith(probe_plan["socket_floor_prefix"])
                        ],
                        fixture=boxes[fixture_id],
                        plate=boxes[support_id],
                        other_fastener=[
                            box
                            for other in fastener_ids
                            if other != target_name
                            for box in boxes[other]
                        ],
                    )
                    measurements = {}
                    for group, shapes in groups.items():
                        pairs = [
                            dict(tool=x["path"], other=y["path"], **distance(x, y))
                            for x in long
                            for y in shapes
                        ]
                        measurements[group] = dict(
                            minimum_distance_m=min(x["distance_m"] for x in pairs),
                            minimum_sat_margin_m=min(x["sat_margin_m"] for x in pairs),
                            intersecting_pairs=sum(x["intersects"] for x in pairs),
                            distance_scope="Exact convex-box distance"
                            if all(x["shape_type"] == "native Cube" for x in long + shapes)
                            else "Conservative lower bound using boxes enclosing native Cylinder/Sphere features; positive distance proves primitive separation, zero is inconclusive",
                            pairs=pairs,
                        )
                    snapshots[str(step)] = dict(
                        bodies=actual, geometry=boxes, clearances=measurements
                    )
                if step in [119, 479, 759, 1199, 1559] or first_wall:
                    capture(
                        case,
                        step,
                        np.array(requested[target_name]["position"]) + np.array([0, 0, 0.003]),
                    )
            start = trace[479]["bodies"][target_name]
            last = trace[1199]["bodies"][target_name]
            delta = (
                rotation(start["orientation_wxyz"]).inv() * rotation(last["orientation_wxyz"])
            ).as_rotvec(degrees=True)
            shift = float(np.linalg.norm(np.array(last["position"]) - start["position"]))
            wall_pairs = {
                k: v
                for k, v in stats["pairs"].items()
                if "/" + target_name + "/" in k and "_" + probe_plan["socket_wall_prefix"] in k
            }
            entry = snapshots["479"]["clearances"]
            native_offsets_ok = all(
                offset_subset_matches(offsets[name]["contact_offsets_m"], inventory[name])
                for name in [tool_id, *fastener_ids]
            )
            passed = (
                native_offsets_ok
                and 5 < direction * delta[2] < 18
                and np.linalg.norm(delta[:2]) < 1
                and shift < 0.001
                and bool(wall_pairs)
                and all(x["min_separation_m"] > -0.00025 for x in wall_pairs.values())
                and all(
                    entry[group]["intersecting_pairs"] == 0
                    for group in ["walls", "floor", "fixture", "plate", "other_fastener"]
                )
            )
            check = dict(
                case=case,
                source_state_index=state_index,
                direction=direction,
                target=target_name,
                actual_screw_rotation_vector_deg=delta.tolist(),
                target_translation_during_twist_m=shift,
                first_positive_wall_contact_step=phase["first_wall_step"],
                wall_contact_pairs=wall_pairs,
                entry_clearances={
                    group: {k: v for k, v in m.items() if k != "pairs"}
                    for group, m in entry.items()
                },
                effective_offsets_verified=native_offsets_ok,
                passed=bool(passed),
            )
            checks.append(check)
            drift = {
                n: dict(
                    translation_m=float(
                        np.linalg.norm(
                            np.array(trace[-1]["bodies"][n]["position"]) - value["position"]
                        )
                    ),
                    rotation_deg=float(
                        (
                            rotation(value["orientation_wxyz"]).inv()
                            * rotation(trace[-1]["bodies"][n]["orientation_wxyz"])
                        ).magnitude()
                        * 180
                        / math.pi
                    ),
                    finite=bool(np.isfinite(trace[-1]["bodies"][n]["position"]).all()),
                )
                for n, value in initial.items()
            }
            results.append(
                dict(
                    name=case,
                    kind="reveal",
                    steps=1560,
                    integration_clock=dict(
                        native_steps=int(world.current_time_step_index) - step_origin,
                        native_duration_seconds=float(world.current_time) - clock_origin,
                        expected_native_steps=1560,
                        expected_duration_seconds=6.5,
                        all_explicit_steps_checked=True,
                    ),
                    requested_joint_positions={},
                    body_drift=drift,
                    contact_summary=stats,
                    trace=trace,
                    interface_snapshots=snapshots,
                )
            )
            raw.flush()
            (out / f"interface-{case}.json").write_text(
                json.dumps(dict(check=check, snapshots=snapshots), indent=2) + "\n"
            )
            print("NATIVE_INTERFACE_CASE", case, check["passed"], delta.tolist(), flush=True)
    phase["case"] = None
    raw.close()
    for first in range(0, len(frames), 6):
        sheet = Image.new("RGB", (1920, 1692), "#101820")
        draw = ImageDraw.Draw(sheet)
        for j, (label, frame) in enumerate(frames[first : first + 6]):
            x = j % 2 * 960
            y = j // 2 * 564
            draw.text((x + 8, y + 3), label, fill="white")
            sheet.paste(Image.fromarray(frame), (x, y + 24))
        sheet.save(out / f"rest-review-{first // 6:02d}.jpg", quality=94)
    report = dict(
        scene_id=spec["scene_id"],
        native_runtime=True,
        isaacsim=version("isaacsim"),
        source_spec_sha256=digest(source),
        probe_plan_sha256=digest(Path(a.plan)),
        source_usd_sha256=digest(usd),
        body_physics=body_physics,
        joint_drives={},
        joint_collision_settings={},
        contact_capture_errors=callback_errors,
        image_capture_errors=[],
        native_rest_images=native_images,
        contact_records=raw_count,
        raw_contact_limit=None,
        extra_source_files={"box_geometry.py": digest(Path(__file__).parent / "box_geometry.py")},
        states=results,
        interface_validation=dict(
            checks=checks,
            all_checks_passed=all(c["passed"] for c in checks),
            native_offset_readbacks=offset_readbacks,
            collider_inventory=inventory,
            physics_dt=1 / 240,
            held_body_motion_api="native RigidBodyView.set_kinematic_targets",
            prescribed_body_ids=[tool_id],
            other_body_pose_commands_after_reset=0,
            initial_tool_lift_m=probe_plan["insertion_lift_m"],
            twist_degrees=15,
            maximum_commanded_twist_speed_degrees_per_s=9,
            source_geometry_or_filtering_overridden=False,
            scope="Configured independent entry/positive-or-negative turn/withdrawal cases. Actual dynamic screws/fixture and native contact offsets; no threads, torque calibration or robot policy. Box distances independently measured from actual collider geometry and post-step body poses.",
        ),
    )
    report["render_transform_checks"] = render_audits
    if a.render_bridge:
        report["extra_source_files"]["render_sync.py"] = digest(
            Path(__file__).parent / "render_sync.py"
        )
    report["interface_validation"]["render_bridge_enabled"] = a.render_bridge
    (out / "physics-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_INTERFACE_DONE", report["interface_validation"]["all_checks_passed"], flush=True)
except BaseException as error:
    (out / "failure.json").write_text(
        json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2) + "\n"
    )
    print(traceback.format_exc(), flush=True)
    raise
finally:
    app.close()
