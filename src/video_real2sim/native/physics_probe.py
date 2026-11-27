"""Native per-state contact/pose/joint evidence; no robot policy qualification.

Uses PhysX contact reports (actual separation/impulse), not AABB overlap as a
substitute for collision evidence. Optional passive variant is explicitly logged.
"""

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
p.add_argument("scene_usd")
p.add_argument("output")
p.add_argument("--indices", type=int, nargs="*")
p.add_argument("--steps", type=int, default=180)
p.add_argument("--passive", action="store_true")
p.add_argument("--enable-joint-collisions", action="store_true")
p.add_argument("--release-assets", nargs="*", default=[])
p.add_argument("--sweep-joint")
p.add_argument("--sweep-end", type=float)
p.add_argument("--sweep-delay", type=int, default=60)
p.add_argument("--sweep-duration", type=int, default=360)
p.add_argument("--attachment-retention", action="store_true")
p.add_argument("--attachment-hold-only", action="store_true")
p.add_argument("--attachment-level-before-release", action="store_true")
p.add_argument("--attachment-lift", type=float, default=0.12)
p.add_argument("--enable-ccd", action="store_true")
p.add_argument("--substeps", type=int, default=1)
p.add_argument("--fine-after-step", type=int, default=0)
p.add_argument("--twist-asset")
p.add_argument("--twist-degrees", type=float, default=20)
p.add_argument("--kinematic-targets", action="store_true")
p.add_argument("--sync-physics-render", action="store_true")
p.add_argument("--mirror-kinematic-usd", action="store_true")
p.add_argument("--focus-asset")
p.add_argument("--support-collider")
a = p.parse_args()
assert 1 <= a.substeps <= 16
assert 0 <= a.fine_after_step < a.steps
initial_substeps = 1 if a.fine_after_step else a.substeps
if a.attachment_hold_only:
    assert a.attachment_retention
if a.attachment_level_before_release:
    assert a.attachment_retention and not a.attachment_hold_only
assert math.isfinite(a.attachment_lift) and 0.05 <= a.attachment_lift <= 0.5
if a.mirror_kinematic_usd:
    assert a.kinematic_targets and a.sync_physics_render
if a.twist_asset:
    assert (
        not a.sweep_joint
        and not a.attachment_retention
        and math.isfinite(a.twist_degrees)
        and 0 < abs(a.twist_degrees) <= 30
        and a.steps > 300
    )
if a.sweep_joint:
    assert not a.passive and a.sweep_end is not None
    assert (
        a.sweep_delay >= 0 and a.sweep_duration > 0 and a.steps > a.sweep_delay + a.sweep_duration
    )
spec_path = Path(a.spec).resolve()
scene_path = Path(a.scene_usd).resolve()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
spec = json.loads(spec_path.read_text())
lock_path = Path(os.environ["REAL2SIM_DEVICE_LOCK"]).expanduser()
lock_path.parent.mkdir(parents=True, exist_ok=True)
device_lock = lock_path.open("a+")
fcntl.flock(device_lock, fcntl.LOCK_EX)
if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
    raise RuntimeError("Explicit Isaac Sim EULA acceptance is required")
for candidate in ["/etc/vulkan/icd.d/nvidia_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"]:
    if Path(candidate).is_file() and Path(candidate).stat().st_size:
        os.environ["VK_DRIVER_FILES"] = candidate
        os.environ["VK_ICD_FILENAMES"] = candidate
        break
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "width": 960,
        "height": 540,
        "renderer": "PathTracing",
        "samples_per_pixel_per_frame": 16,
        "max_bounces": 12,
        "max_specular_transmission_bounces": 16,
        "extra_args": ["--allow-root"],
    }
)
import carb
import numpy as np
import omni.usd

carb.settings.get_settings().set_bool("/app/viewport/grid/enabled", False)
carb.settings.get_settings().set_bool("/app/viewport/defaults/guide/grid/visible", False)
from importlib.metadata import version

from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.sensors.camera import Camera
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx.bindings._physx import SimulationEvent
from PIL import Image, ImageDraw
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

if a.mirror_kinematic_usd:
    from render_sync import audit as audit_render
    from render_sync import prepare as prepare_render
if spec.get("attachments"):
    from scene_attachments import build_attachments, measure_attachments, set_enabled_attachments


def quat_to_rotation(q):
    return Rotation.from_quat(
        [q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()]
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pose(body):
    pos, q = body.get_world_pose()
    return np.asarray(pos, float), np.asarray(q, float)


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def scalar_quat(euler):
    return Rotation.from_euler("xyz", euler, degrees=True).as_quat()[[3, 0, 1, 2]]


try:
    context = omni.usd.get_context()
    context.open_stage(str(scene_path))
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    body_paths = {}
    body_physics = {}
    retention_panel = None
    if a.attachment_retention:
        assert spec.get("attachments") and a.steps >= 900 and not a.sweep_joint
        parents = {c["parent"] for c in spec["attachments"]}
        assert len(parents) == 1
        retention_panel = next(iter(parents))
        UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath("/World/Assets/" + retention_panel)
        ).GetKinematicEnabledAttr().Set(True)
    # A new render product avoids reusing serialized /Render sensor bindings
    # from the exported scene's previous Kit process.
    assert stage.GetPrimAtPath("/World/SourceCamera")
    camera_path = "/World/PhysicsInspectionCamera"
    UsdGeom.Camera.Define(stage, camera_path).SetFromCamera(
        UsdGeom.Camera(stage.GetPrimAtPath("/World/SourceCamera")).GetCamera()
    )
    inspection_camera = Camera(prim_path=camera_path, resolution=(960, 540), frequency=30)
    native_rest_images = []
    review_frames = []
    image_capture_errors = []
    render_clock_checks = []
    render_transform_checks = []
    cup_tops = {}
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for asset in spec["assets"]:
        if asset["id"].startswith("cup_"):
            hi = (
                bounds.ComputeWorldBound(stage.GetPrimAtPath("/World/Assets/" + asset["id"]))
                .ComputeAlignedRange()
                .GetMax()
            )
            cup_tops[asset["id"]] = float(hi[2] - asset["position"][2])

    def camera_view(eye, target, fov=36):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
        xf.ClearXformOpOrder()

        def v(x):
            return Gf.Vec3d(*map(float, x))

        xf.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(v(eye), v(target), Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        cam.GetProjectionAttr().Set("perspective")
        cam.GetHorizontalApertureAttr().Set(36.0)
        cam.GetVerticalApertureAttr().Set(20.25)
        cam.GetFocalLengthAttr().Set(20.25 / (2 * math.tan(math.radians(fov) / 2)))

    def capture_rest(state_index, label, completed_steps=None):
        before_time = float(world.current_time)
        before_step = int(world.current_time_step_index)
        before_dt = float(world.get_physics_dt())
        actual_before = (
            {name: read_pose(body) for name, body in bodies.items()}
            if a.sync_physics_render
            else {}
        )
        bridge_before = (
            prepare_render(stage, bodies, body_paths, held_ids) if a.mirror_kinematic_usd else None
        )
        if a.sync_physics_render and not a.mirror_kinematic_usd:
            get_physx_interface().update_transformations(True, True, True, False)
        for _ in range(20):
            world.render()
        after_time = float(world.current_time)
        after_step = int(world.current_time_step_index)
        after_dt = float(world.get_physics_dt())
        render_clock_checks.append(
            dict(
                state_index=state_index,
                view=label,
                before_time=before_time,
                after_time=after_time,
                before_step=before_step,
                after_step=after_step,
                before_dt=before_dt,
                after_dt=after_dt,
            )
        )
        assert (
            before_time == after_time
            and before_step == after_step
            and abs(after_dt - before_dt) < 1e-10
        ), "Render changed native simulation clock or integration rate"
        if a.sync_physics_render:
            cache = UsdGeom.XformCache()
            checks = {}
            for name, body in bodies.items():
                position, q = read_pose(body)
                matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_paths[name]))
                render_position = np.array(matrix.ExtractTranslation())
                render_rotation = quat_to_rotation(matrix.ExtractRotationQuat())
                error = float(np.linalg.norm(render_position - position))
                angle = float(np.degrees((rotation(q).inv() * render_rotation).magnitude()))
                before_position, before_q = actual_before[name]
                unchanged = (
                    float(np.linalg.norm(position - before_position)) < 1e-8
                    and (rotation(before_q).inv() * rotation(q)).magnitude() < 1e-8
                )
                checks[name] = dict(
                    actual_position=position.tolist(),
                    actual_orientation_wxyz=q.tolist(),
                    usd_position=render_position.tolist(),
                    usd_orientation_xyzw=render_rotation.as_quat().tolist(),
                    translation_error_m=error,
                    rotation_error_deg=angle,
                    actual_pose_unchanged_during_render=unchanged,
                )
            result = dict(
                state_index=state_index,
                view=label,
                method="Native PhysX.update_transformations(True,True,True,False); no body pose setters",
                bodies=checks,
                passed=all(
                    c["translation_error_m"] < 0.0001
                    and c["rotation_error_deg"] < 0.05
                    and c["actual_pose_unchanged_during_render"]
                    for c in checks.values()
                ),
            )
            if a.mirror_kinematic_usd:
                result.update(
                    audit_render(stage, bodies, body_paths, bridge_before),
                    mirrored_external_kinematic_paths=[body_paths[name] for name in held_ids],
                    method="Native dynamic writeback plus measured external kinematic root USD mirror; no dynamic body pose setters",
                )
            render_transform_checks.append(result)
            (out / "render-transform-checks.json").write_text(
                json.dumps(render_transform_checks, indent=2) + "\n"
            )
            assert result["passed"], (
                "Native physics-to-USD render synchronization remains inconsistent; preserve audit and do not qualify images"
            )
        frame = np.asarray(inspection_camera.get_rgba())
        if frame.shape != (540, 960, 4) or frame[..., :3].std() <= 2:
            image_capture_errors.append(
                dict(
                    state_index=state_index,
                    view=label,
                    shape=list(frame.shape),
                    rgb_std=float(frame[..., :3].std()) if frame.shape == (540, 960, 4) else None,
                )
            )
            if frame.shape == (540, 960, 4):
                Image.fromarray(frame[..., :3].astype(np.uint8)).save(
                    out / f"failed-render-S{state_index:02d}-{label}.png"
                )
            return
        frame = frame[..., :3].astype(np.uint8)
        path = out / f"rest-S{state_index:02d}-{label}.png"
        Image.fromarray(frame).save(path)
        completed_steps = a.steps if completed_steps is None else completed_steps
        completed_native_steps = int(world.current_time_step_index) - loop_step_origin
        native_rest_images.append(
            dict(
                state_index=state_index,
                view=label,
                file=path.name,
                sha256=digest(path),
                completed_physics_steps=completed_native_steps,
                completed_reference_steps=completed_steps,
                scope="Native render of actual PhysX state; native steps measured since requested pose initialization; reference steps are1/120s schedule indices. Render calls do not advance simulation.",
            )
        )
        review_frames.append(
            (
                f"S{state_index:02d} {label} | {completed_native_steps} native steps; reference{completed_steps}",
                frame,
            )
        )

    for asset in spec["assets"]:
        path = "/World/Assets/" + asset["id"]
        prim = stage.GetPrimAtPath(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            if asset["id"] in a.release_assets:
                rb.GetKinematicEnabledAttr().Set(False)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            body_paths[asset["id"]] = path
            body_physics[asset["id"]] = dict(
                kinematic=rb.GetKinematicEnabledAttr().Get(),
                rigid_body_enabled=rb.GetRigidBodyEnabledAttr().Get(),
                gravity_disabled=PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get(),
                mass_kg=UsdPhysics.MassAPI(prim).GetMassAttr().Get(),
                collider_count=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim)),
            )
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / (120 * initial_substeps),
        rendering_dt=1 / 30,
        physics_prim_path="/World/PhysicsScene",
    )
    ccd_bodies = []
    ground_check = None
    leg_corners = {}
    focus_sphere = None
    if a.enable_ccd:
        PhysxSchema.PhysxSceneAPI.Apply(
            stage.GetPrimAtPath("/World/PhysicsScene")
        ).CreateEnableCCDAttr(True)
        for name, path in body_paths.items():
            if body_physics[name]["kinematic"]:
                continue
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
            api.CreateEnableCCDAttr(True)
            api.CreateEnableSpeculativeCCDAttr(True)
            ccd_bodies.append(name)
        if a.attachment_retention:
            cache = UsdGeom.XformCache()
            ground = UsdGeom.Cube(stage.GetPrimAtPath("/World/Ground"))
            assert ground
            h = float(ground.GetSizeAttr().Get()) / 2
            matrix = cache.GetLocalToWorldTransform(ground.GetPrim())
            corners = np.array(
                [
                    matrix.TransformAffine(Gf.Vec3d(x, y, z))
                    for x in [-h, h]
                    for y in [-h, h]
                    for z in [-h, h]
                ]
            )
            ground_check = dict(
                top_z_m=float(corners[:, 2].max()),
                xy_min=corners[:, :2].min(axis=0).tolist(),
                xy_max=corners[:, :2].max(axis=0).tolist(),
                scope="Exact transformed analytic native Cube vertices against compiled horizontal ground top; actual after-World.step poses",
            )
            for connection in spec["attachments"]:
                name = connection["child"]
                root_prim = stage.GetPrimAtPath(body_paths[name])
                inverse = cache.GetLocalToWorldTransform(root_prim).GetInverse()
                local = []
                for prim in Usd.PrimRange(root_prim):
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    assert prim.IsA(UsdGeom.Cube), (
                        "Extend exact clearance method before using it for non-Cube leg colliders"
                    )
                    half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                    transform = cache.GetLocalToWorldTransform(prim)
                    local.extend(
                        inverse.TransformAffine(transform.TransformAffine(Gf.Vec3d(x, y, z)))
                        for x in [-half, half]
                        for y in [-half, half]
                        for z in [-half, half]
                    )
                assert local
                leg_corners[name] = np.array(local)
    if a.focus_asset:
        assert a.focus_asset in body_paths and not body_physics[a.focus_asset]["kinematic"]
        cache = UsdGeom.XformCache()
        root_prim = stage.GetPrimAtPath(body_paths[a.focus_asset])
        colliders = [p for p in Usd.PrimRange(root_prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(colliders) == 1 and colliders[0].IsA(UsdGeom.Sphere)
        sphere = UsdGeom.Sphere(colliders[0])
        matrix = (
            cache.GetLocalToWorldTransform(colliders[0])
            * cache.GetLocalToWorldTransform(root_prim).GetInverse()
        )
        r = float(sphere.GetRadiusAttr().Get())
        radii = [
            np.linalg.norm(matrix.TransformDir(Gf.Vec3d(*(r if i == j else 0.0 for i in range(3)))))
            for j in range(3)
        ]
        assert max(radii) - min(radii) < 1e-8
        focus_sphere = dict(
            center_local=list(matrix.TransformAffine(Gf.Vec3d(0))),
            radius_m=float(radii[0]),
            native_collider=str(colliders[0].GetPath()),
        )
        support_path = a.support_collider or "/World/Ground"
        support_prim = stage.GetPrimAtPath(support_path)
        ground = UsdGeom.Cube(support_prim)
        assert ground and support_prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.CollisionAPI(support_prim).GetCollisionEnabledAttr().Get()
        ancestor = support_prim
        while ancestor:
            assert not ancestor.HasAPI(UsdPhysics.RigidBodyAPI), "Support must remain static"
            ancestor = ancestor.GetParent()
        h = float(ground.GetSizeAttr().Get()) / 2
        gm = cache.GetLocalToWorldTransform(ground.GetPrim())
        linear = np.asarray(gm)[:3, :3]
        assert np.allclose(linear, np.diag(np.diag(linear)), atol=1e-10)
        corners = np.array(
            [
                gm.TransformAffine(Gf.Vec3d(x, y, z))
                for x in [-h, h]
                for y in [-h, h]
                for z in [-h, h]
            ]
        )
        ground_check = dict(
            native_collider=support_path,
            top_z_m=float(corners[:, 2].max()),
            xy_min=corners[:, :2].min(axis=0).tolist(),
            xy_max=corners[:, :2].max(axis=0).tolist(),
            scope="Exact compiledhorizontalnativeCube ground top; focusedbodyuses actualnativeSpherecenter/radius, not AABB approximation",
        )
    bodies = {
        k: world.scene.add(SingleRigidPrim(prim_path=v, name=k)) for k, v in body_paths.items()
    }
    if a.twist_asset:
        assert a.twist_asset in bodies and body_physics[a.twist_asset]["kinematic"], (
            "Only the source-held key may be prescribed; screws and fixture remain dynamic"
        )
    joint_data = {}
    for j in spec.get("joints", []):
        prim = stage.GetPrimAtPath("/World/Joints/" + j["id"])
        base = UsdPhysics.Joint(prim)
        if a.enable_joint_collisions:
            base.GetCollisionEnabledAttr().Set(True)
        drive = (
            UsdPhysics.DriveAPI.Get(prim, "linear" if j["type"] == "prismatic" else "angular")
            if j["type"] != "fixed"
            else None
        )
        settings = {}
        if drive:
            settings = dict(
                stiffness=drive.GetStiffnessAttr().Get(),
                damping=drive.GetDampingAttr().Get(),
                max_force=drive.GetMaxForceAttr().Get(),
            )
            if a.passive:
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(0.0)
        joint_data[j["id"]] = dict(
            spec=j,
            base=base,
            drive=drive,
            original_drive=settings,
            local0=quat_to_rotation(Gf.Quatd(base.GetLocalRot0Attr().Get())),
            local1=quat_to_rotation(Gf.Quatd(base.GetLocalRot1Attr().Get())),
        )
    phase = {"name": None, "step": 0, "stats": None}
    callback_errors = []
    raw_count = 0
    native_joint_break_events = []

    def simulation_event(event):
        if event.type == int(SimulationEvent.JOINT_BREAK):
            try:
                native_joint_break_events.append(
                    dict(
                        event="JOINT_BREAK",
                        encoded_joint_path=[int(v) for v in event.payload["jointPath"]],
                        delivery_state=phase["name"],
                        delivery_reference_step=phase["step"],
                        delivery_substep=phase.get("substep", 0),
                        timestamp_scope="Native event, observed on event-stream delivery; delivery step need not be exact generation step",
                    )
                )
            except Exception as e:
                callback_errors.append("joint_break_event:" + repr(e))

    simulation_subscription = (
        get_physx_interface()
        .get_simulation_event_stream_v2()
        .create_subscription_to_pop(simulation_event, name="Native attachment break evidence")
    )
    raw = gzip.open(out / "contacts.jsonl.gz", "wt", encoding="utf8")

    def contacts(headers, data):
        global raw_count
        if phase["name"] is None:
            return
        try:
            for h in headers:
                actors = [
                    str(PhysicsSchemaTools.intToSdfPath(h.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(h.actor1)),
                ]
                colliders = [
                    str(PhysicsSchemaTools.intToSdfPath(getattr(h, k))) if hasattr(h, k) else None
                    for k in ["collider0", "collider1"]
                ]
                for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    c = data[i]
                    sep = float(c.separation)
                    impulse = [float(x) for x in c.impulse]
                    pair = " <-> ".join(actors)
                    stats = phase["stats"]
                    stats["contacts"] += 1
                    if sep < stats["min_separation_m"]:
                        stats["worst_contact"] = dict(
                            step=phase["step"],
                            actors=actors,
                            colliders=colliders,
                            position=[float(x) for x in c.position],
                            separation_m=sep,
                        )
                    stats["min_separation_m"] = min(stats["min_separation_m"], sep)
                    if phase["step"] >= a.steps // 2:
                        stats["late_min_separation_m"] = min(stats["late_min_separation_m"], sep)
                    pair_stat = stats["pairs"].setdefault(
                        pair, dict(contacts=0, min_separation_m=0.0, max_impulse_Ns=0.0)
                    )
                    pair_stat["contacts"] += 1
                    pair_stat["min_separation_m"] = min(pair_stat["min_separation_m"], sep)
                    pair_stat["max_impulse_Ns"] = max(
                        pair_stat["max_impulse_Ns"], float(np.linalg.norm(impulse))
                    )
                    if raw_count < 300000:
                        raw.write(
                            json.dumps(
                                dict(
                                    state=phase["name"],
                                    step=phase["step"],
                                    substep=phase.get("substep", 0),
                                    actors=actors,
                                    colliders=colliders,
                                    position=[float(x) for x in c.position],
                                    normal=[float(x) for x in c.normal],
                                    impulse_Ns=impulse,
                                    separation_m=sep,
                                )
                            )
                            + "\n"
                        )
                        raw_count += 1
        except Exception as e:
            if len(callback_errors) < 10:
                callback_errors.append(repr(e))

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    ds = spec.get("diagnostic_states", [])
    if a.indices is None:
        indices = [0] if ds and ds[0]["kind"] == "initial" else []
        for i, state in enumerate(ds):
            if state["kind"] == "reveal" and any(
                abs(x["value"] - joint_data[x["joint_id"]]["spec"]["initial_position"]) > 0.01
                for x in state.get("joint_positions", [])
            ):
                indices.append(i)
                break
        indices += [i for i, x in enumerate(ds) if x["kind"] == "final"][-1:]
    else:
        indices = [i if i >= 0 else len(ds) + i for i in a.indices]
    selected_indices = list(dict.fromkeys(indices))
    states = [ds[i] for i in selected_indices]
    if not states:
        selected_indices = [0]
        states = [dict(name_zh="initial", kind="initial", poses=[], joint_positions=[])]
    if a.sweep_joint:
        assert len(states) == 1, "A sweep starts from exactly one specified state"
        entry = joint_data[a.sweep_joint]
        assert entry["drive"] and entry["spec"]["type"] != "fixed"
        assert entry["spec"]["lower"] <= a.sweep_end <= entry["spec"]["upper"]
    if a.attachment_retention:
        assert len(states) == 1 and set(states[0]["enabled_attachments"]) == {
            connection["id"] for connection in spec["attachments"]
        }
    results = []
    for probe_index, state in enumerate(states):
        state_index = selected_indices[probe_index]
        phase["name"] = None
        # A source state may move from separated parts to a connected assembly.
        # Never enable a constraint while reset still holds the separated
        # default poses: it can break before the requested state is initialized.
        # Recreate only these declared connections after all body poses are set.
        if spec.get("attachments"):
            stage.RemovePrim("/World/Attachments")
        enabled_attachments = []
        world.set_simulation_dt(physics_dt=1 / (120 * initial_substeps), rendering_dt=1 / 30)
        world.reset()
        if a.enable_ccd:
            assert (
                PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath("/World/PhysicsScene"))
                .GetEnableCCDAttr()
                .Get()
            )
            assert all(
                PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(body_paths[name]))
                .GetEnableCCDAttr()
                .Get()
                and PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(body_paths[name]))
                .GetEnableSpeculativeCCDAttr()
                .Get()
                for name in ccd_bodies
            )
        inspection_camera.initialize()
        poses = {
            asset["id"]: dict(position=asset["position"], rotation_deg=asset["rotation_deg"])
            for asset in spec["assets"]
        }
        for pose in state.get("poses", []):
            poses[pose["asset_id"]] = pose
        target_q = {j["id"]: j["initial_position"] for j in spec.get("joints", [])}
        target_q.update({x["joint_id"]: x["value"] for x in state.get("joint_positions", [])})
        command_q = dict(target_q)
        explicit = {x["asset_id"] for x in state.get("poses", [])}
        for jid, entry in joint_data.items():
            j = entry["spec"]
            drive = entry["drive"]
            if drive:
                drive.GetTargetPositionAttr().Set(float(target_q[jid]))
            if j["child"] not in explicit and j["type"] != "fixed":
                parent = poses[j["parent"]]
                rp = Rotation.from_euler("xyz", parent["rotation_deg"], degrees=True)
                p0 = np.array(parent["position"]) + rp.apply(j["parent_anchor"])
                r0 = rp * entry["local0"]
                axis = np.eye(3)["XYZ".index(j["axis"])]
                rr = (
                    Rotation.from_rotvec(axis * np.deg2rad(target_q[jid]))
                    if j["type"] == "revolute"
                    else Rotation.identity()
                )
                rc = r0 * rr * entry["local1"].inv()
                p1 = (
                    p0
                    + (r0.apply(axis) * target_q[jid] if j["type"] == "prismatic" else 0)
                    - rc.apply(j["child_anchor"])
                )
                poses[j["child"]] = dict(
                    position=p1.tolist(), rotation_deg=rc.as_euler("xyz", degrees=True).tolist()
                )
        requested = {}
        for name, body in bodies.items():
            pose = poses[name]
            q = scalar_quat(pose["rotation_deg"])
            pos = np.array(pose["position"], float)
            body.set_world_pose(position=pos, orientation=q)
            if not body_physics[name]["kinematic"]:
                body.set_linear_velocity(np.zeros(3))
                body.set_angular_velocity(np.zeros(3))
            requested[name] = (pos, q)
        if spec.get("attachments"):
            build_attachments(stage, spec)
            enabled_attachments = set_enabled_attachments(stage, spec, state)
        held_ids = ([retention_panel] if a.attachment_retention else []) + (
            [a.twist_asset] if a.twist_asset else []
        )
        if a.kinematic_targets:
            assert held_ids and all(body_physics[name]["kinematic"] for name in held_ids)
        held_views = (
            {
                name: world.physics_sim_view.create_rigid_body_view(body_paths[name])
                for name in held_ids
            }
            if a.kinematic_targets
            else {}
        )
        assert all(view.count == 1 for view in held_views.values())

        def command_held(name, position, orientation, held_views=held_views):
            if a.kinematic_targets:
                # Native kinematic target computes motion over this time step.
                # set_world_pose is an immediate teleport, reserved for reset.
                data = (
                    np.concatenate([position, np.asarray(orientation)[[1, 2, 3, 0]]])
                    .astype(np.float32)
                    .reshape(1, 7)
                )
                held_views[name].set_kinematic_targets(data, np.array([0], dtype=np.int32))
            else:
                bodies[name].set_world_pose(position=position, orientation=orientation)

        stats = dict(contacts=0, min_separation_m=0.0, late_min_separation_m=0.0, pairs={})
        phase.update(name=state["name_zh"], step=0, stats=stats)
        trace = []
        first_focus_impulse_step = None
        loop_time_origin = float(world.current_time)
        loop_step_origin = int(world.current_time_step_index)
        schedule = (
            (step, substep)
            for step in range(a.steps)
            for substep in range(a.substeps if step >= a.fine_after_step else 1)
        )
        for micro_step, (step, substep) in enumerate(schedule):
            active_substeps = a.substeps if step >= a.fine_after_step else 1
            if step == a.fine_after_step and substep == 0 and a.fine_after_step:
                world.set_simulation_dt(physics_dt=1 / (120 * a.substeps), rendering_dt=1 / 30)
            assert abs(world.get_physics_dt() - 1 / (120 * active_substeps)) < 1e-10
            motion_step = step + substep / active_substeps
            if a.twist_asset:
                u = float(np.clip((motion_step - 60) / 240, 0.0, 1.0))
                u = u * u * (3 - 2 * u)
                held = poses[a.twist_asset]
                orientation = Rotation.from_euler(
                    "xyz", held["rotation_deg"], degrees=True
                ) * Rotation.from_euler("z", a.twist_degrees * u, degrees=True)
                command_held(
                    a.twist_asset, np.array(held["position"]), orientation.as_quat()[[3, 0, 1, 2]]
                )
            if a.attachment_retention:
                # Only the explicitly externally held panel receives pose
                # commands. Legs are never repositioned after initialization.
                u = float(np.clip((motion_step - 60) / 240, 0.0, 1.0))
                u = u * u * (3 - 2 * u)
                angle = 25 * u
                if a.attachment_level_before_release:
                    # Keep the full lift/25-degree retention test, then bring
                    # the assembled table back to its upright reference before
                    # unlocking. This differs from the retained tilted
                    # unsupported-fall stress test; it is not hand support.
                    back = float(np.clip((motion_step - 300) / 50, 0.0, 1.0))
                    back = back * back * (3 - 2 * back)
                    angle *= 1 - back
                pose = poses[retention_panel]
                r = Rotation.from_euler(
                    "xyz", pose["rotation_deg"], degrees=True
                ) * Rotation.from_euler("x", angle, degrees=True)
                command_held(
                    retention_panel,
                    np.array(pose["position"]) + np.array([0, 0, a.attachment_lift * u]),
                    r.as_quat()[[3, 0, 1, 2]],
                )
                for index, c in enumerate(spec["attachments"]):
                    if not a.attachment_hold_only and substep == 0 and step == 360 + 120 * index:
                        UsdPhysics.Joint(
                            stage.GetPrimAtPath("/World/Attachments/" + c["id"])
                        ).GetJointEnabledAttr().Set(False)
            if a.sweep_joint:
                u = float(np.clip((motion_step - a.sweep_delay) / a.sweep_duration, 0.0, 1.0))
                u = u * u * (3 - 2 * u)
                command_q[a.sweep_joint] = (
                    target_q[a.sweep_joint] + (a.sweep_end - target_q[a.sweep_joint]) * u
                )
                joint_data[a.sweep_joint]["drive"].GetTargetPositionAttr().Set(
                    command_q[a.sweep_joint]
                )
            # In this installed Isaac5SDK, World.step(render=True) delegates to
            # app.update and may integrate multiple physics substeps per render
            # interval. One diagnostic iteration must always advance exactly
            # one explicit physics step; camera warm-up uses render() separately.
            phase.update(step=step, substep=substep)
            world.step(render=False)
            measured_native_count = int(world.current_time_step_index) - loop_step_origin
            measured_elapsed = float(world.current_time) - loop_time_origin
            expected_elapsed = (step + (substep + 1) / active_substeps) / 120
            assert measured_native_count == micro_step + 1, (
                "An explicit integration call advanced an unexpected number of native steps"
            )
            assert abs(measured_elapsed - expected_elapsed) < 0.00001, (
                "Native simulation clock diverged from the declared motion/release schedule"
            )
            if (
                a.attachment_retention
                or a.focus_asset
                or (
                    substep == active_substeps - 1
                    and (
                        step % (1 if a.sweep_joint or a.twist_asset else 10) == 0
                        or step == a.steps - 1
                    )
                )
            ):
                values = {}
                for name, body in bodies.items():
                    pos, q = read_pose(body)
                    values[name] = dict(
                        position=pos.tolist(),
                        orientation_wxyz=q.tolist(),
                        linear_velocity=np.asarray(body.get_linear_velocity()).tolist(),
                        angular_velocity=np.asarray(body.get_angular_velocity()).tolist(),
                    )
                measured = {}
                anchor_errors = {}
                for jid, entry in joint_data.items():
                    j = entry["spec"]
                    parent = values[j["parent"]]
                    child = values[j["child"]]
                    rp = rotation(parent["orientation_wxyz"])
                    rc = rotation(child["orientation_wxyz"])
                    axis = np.eye(3)["XYZ".index(j["axis"])]
                    r0 = rp * entry["local0"]
                    if j["type"] == "revolute":
                        measured[jid] = float(
                            np.rad2deg(
                                np.dot(((r0.inv() * (rc * entry["local1"])).as_rotvec()), axis)
                            )
                        )
                    elif j["type"] == "prismatic":
                        measured[jid] = float(
                            np.dot(
                                np.array(child["position"])
                                + rc.apply(j["child_anchor"])
                                - np.array(parent["position"])
                                - rp.apply(j["parent_anchor"]),
                                r0.apply(axis),
                            )
                        )
                    delta = (
                        np.array(child["position"])
                        + rc.apply(j["child_anchor"])
                        - np.array(parent["position"])
                        - rp.apply(j["parent_anchor"])
                    )
                    if j["type"] == "prismatic":
                        delta -= r0.apply(axis) * measured[jid]
                    anchor_errors[jid] = float(np.linalg.norm(delta))
                clearance = {}
                for name, corners in leg_corners.items():
                    body = values[name]
                    vertices = rotation(body["orientation_wxyz"]).apply(corners) + body["position"]
                    assert np.all(vertices[:, :2] >= np.array(ground_check["xy_min"])) and np.all(
                        vertices[:, :2] <= np.array(ground_check["xy_max"])
                    )
                    clearance[name] = float(vertices[:, 2].min() - ground_check["top_z_m"])
                if focus_sphere:
                    value = values[a.focus_asset]
                    center = (
                        rotation(value["orientation_wxyz"]).apply(focus_sphere["center_local"])
                        + value["position"]
                    )
                    radius = focus_sphere["radius_m"]
                    assert np.all(center[:2] - radius >= ground_check["xy_min"]) and np.all(
                        center[:2] + radius <= ground_check["xy_max"]
                    )
                    clearance[a.focus_asset] = float(center[2] - radius - ground_check["top_z_m"])
                trace.append(
                    dict(
                        step=step,
                        substep=substep,
                        native_step=micro_step,
                        time_s=expected_elapsed,
                        measured_native_time_since_pose_reset_s=measured_elapsed,
                        measured_native_steps=measured_native_count,
                        physics_dt=1 / (120 * active_substeps),
                        bodies=values,
                        measured_joint_positions=measured,
                        commanded_joint_positions=dict(command_q),
                        measured_joint_anchor_errors_m=anchor_errors,
                        attachments=measure_attachments(stage, spec, values)
                        if spec.get("attachments")
                        else {},
                        post_step_ground_clearance_m=clearance,
                    )
                )
            if a.focus_asset:
                first = first_focus_impulse_step is None and any(
                    body_paths[a.focus_asset] in pair and s["max_impulse_Ns"] > 1e-10
                    for pair, s in stats["pairs"].items()
                )
                if first:
                    first_focus_impulse_step = dict(
                        reference_step=step,
                        substep=substep,
                        native_step=micro_step,
                        time_s=measured_elapsed,
                    )
                regular = substep == active_substeps - 1 and step in {
                    round((a.steps - 1) * fraction) for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]
                }
                if first or regular:
                    position, _ = read_pose(bodies[a.focus_asset])
                    camera_view(position + np.array([0.22, -0.28, 0.12]), position, 30)
                    capture_rest(
                        state_index,
                        "first-impulse" if first else f"focused-step{step:04d}",
                        step + 1,
                    )
            if (
                substep == active_substeps - 1
                and a.sweep_joint
                and step
                in {
                    a.sweep_delay + round(a.sweep_duration * f) for f in [0.0, 0.25, 0.5, 0.75, 1.0]
                }
            ):
                cam = spec["camera"]
                camera_view(cam["position"], cam["target"], cam["vertical_fov_deg"])
                capture_rest(state_index, f"sweep-step{step:04d}", step + 1)
            if (
                substep == active_substeps - 1
                and a.attachment_retention
                and step in [59, 239, 359, 479, 599, 719, 899]
            ):
                cam = spec["camera"]
                camera_view(cam["position"], cam["target"], cam["vertical_fov_deg"])
                capture_rest(state_index, f"attachment-step{step:04d}", step + 1)
            if (
                substep == active_substeps - 1
                and a.twist_asset
                and step in [59, 119, 179, 239, 299, a.steps - 1]
            ):
                target = np.array(poses[a.twist_asset]["position"]) + np.array([0, 0, -0.06])
                camera_view(target + np.array([0.12, -0.16, 0.13]), target, 32)
                capture_rest(state_index, f"twist-step{step:04d}", step + 1)
            if (
                substep == active_substeps - 1
                and a.attachment_retention
                and not a.attachment_hold_only
                and a.enable_ccd
                and step in [645, 647, 648, 650, 766, 768, 769, 770, 772]
            ):
                name = spec["attachments"][2 if step < 700 else 3]["child"]
                pos, _ = read_pose(bodies[name])
                target = np.array([pos[0], pos[1], ground_check["top_z_m"] + 0.06])
                camera_view(target + np.array([0.35, -0.5, 0.15]), target, 36)
                capture_rest(state_index, f"impact-{name}-step{step:04d}", step + 1)
        drift = {}
        for name, (pos, q) in requested.items():
            final = trace[-1]["bodies"][name]
            delta = rotation(q).inv() * rotation(final["orientation_wxyz"])
            drift[name] = dict(
                translation_m=float(np.linalg.norm(np.array(final["position"]) - pos)),
                rotation_deg=float(np.rad2deg(delta.magnitude())),
                finite=bool(np.isfinite(final["position"]).all()),
            )
        sweep = (
            dict(
                joint_id=a.sweep_joint,
                start=target_q[a.sweep_joint],
                end=a.sweep_end,
                delay_steps=a.sweep_delay,
                duration_steps=a.sweep_duration,
                mechanism="Native force-drive target ramp; no scene-body pose updates after reset; not a robot manipulation",
            )
            if a.sweep_joint
            else None
        )
        clock = dict(
            native_steps=measured_native_count,
            native_duration_seconds=measured_elapsed,
            expected_native_steps=a.fine_after_step + (a.steps - a.fine_after_step) * a.substeps,
            expected_duration_seconds=a.steps / 120,
            maximum_allowed_clock_error_s=0.00001,
            all_explicit_steps_checked=True,
            pose_initialization_time_origin=loop_time_origin,
            pose_initialization_native_step_origin=loop_step_origin,
        )
        results.append(
            dict(
                name=state["name_zh"],
                kind=state["kind"],
                source_evidence_frames=state.get("evidence_frames", []),
                enabled_attachments=enabled_attachments,
                steps=a.steps,
                integration_clock=clock,
                first_focus_positive_impulse=first_focus_impulse_step,
                requested_joint_positions=target_q,
                body_drift=drift,
                contact_summary=stats,
                trace=trace,
                continuous_joint_sweep=sweep,
            )
        )
        raw.flush()
        print("NATIVE_PHYSICS_PROBED", state["name_zh"], stats["contacts"], flush=True)
        (out / "partial-physics.json").write_text(
            json.dumps(
                dict(
                    native_runtime=True,
                    source_spec_sha256=digest(spec_path),
                    source_usd_sha256=digest(scene_path),
                    states=results,
                    scope="Completed numerical state probes saved before image capture; partial workflow evidence, never a full camera/validation pass.",
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        cam = spec["camera"]
        camera_view(cam["position"], cam["target"], cam["vertical_fov_deg"])
        capture_rest(state_index, "source")
        for name, height in cup_tops.items():
            position, orientation = read_pose(bodies[name])
            target = position + rotation(orientation).apply([0, 0, height])
            camera_view(target + np.array([0, -0.14, 0.01]), target, 20)
            capture_rest(state_index, name + "-interface")
    phase["name"] = None
    raw.close()
    for first in range(0, len(review_frames), 6):
        sheet = Image.new("RGB", (1920, 1692), "#101820")
        draw = ImageDraw.Draw(sheet)
        for j, (label, frame) in enumerate(review_frames[first : first + 6]):
            x = j % 2 * 960
            y = j // 2 * 564
            draw.text((x + 8, y + 3), label, fill="white")
            sheet.paste(Image.fromarray(frame), (x, y + 24))
        sheet.save(out / f"rest-review-{first // 6:02d}.jpg", quality=94)
    result = dict(
        scene_id=spec["scene_id"],
        isaacsim=version("isaacsim"),
        native_runtime=True,
        source_spec_sha256=digest(spec_path),
        source_usd_sha256=digest(scene_path),
        passive_variant=a.passive,
        adjacent_joint_collisions_enabled=a.enable_joint_collisions,
        released_assets=a.release_assets,
        body_physics=body_physics,
        joint_collision_settings={
            k: v["base"].GetCollisionEnabledAttr().Get() for k, v in joint_data.items()
        },
        joint_drives={k: v["original_drive"] for k, v in joint_data.items()},
        joint_types={j["id"]: j["type"] for j in spec.get("joints", [])},
        contact_capture_errors=callback_errors,
        contact_records=raw_count,
        raw_contact_limit=300000,
        states=results,
        scope="Independent native PhysX settling/contact probes at supplied source-conditioned states. Reset poses are not executed manipulation. No robot policy or true material calibration is claimed. Passive variant disables authored joint spring/damping; reaction torque is not measured. Only enabled pairs generate contacts; adjacent-joint clearance is a separate explicit variant.",
    )
    result["native_rest_images"] = native_rest_images
    result["held_body_motion_api"] = (
        "native tensor RigidBodyView.set_kinematic_targets; set_world_pose used only for initial state reset"
        if a.kinematic_targets
        else "legacy immediate set_world_pose teleport for held-body commands; retained as a distinct diagnostic"
    )
    result["native_joint_break_events"] = native_joint_break_events
    if a.twist_asset:
        measurements = []
        for state in results:
            responses = {}
            for name, initial in state["trace"][0]["bodies"].items():
                if name == a.twist_asset or body_physics[name]["kinematic"]:
                    continue
                last = state["trace"][-1]["bodies"][name]
                delta = rotation(initial["orientation_wxyz"]).inv() * rotation(
                    last["orientation_wxyz"]
                )
                pairs = {
                    pair: value
                    for pair, value in state["contact_summary"]["pairs"].items()
                    if body_paths[a.twist_asset] in pair and body_paths[name] in pair
                }
                responses[name] = dict(
                    relative_rotation_vector_deg=np.rad2deg(delta.as_rotvec()).tolist(),
                    translation_m=float(
                        np.linalg.norm(np.array(last["position"]) - initial["position"])
                    ),
                    tool_contact_pairs=pairs,
                )
            measurements.append(dict(source_state=state["name"], dynamic_body_responses=responses))
        result["tool_twist"] = dict(
            prescribed_body_ids=[a.twist_asset],
            local_axis="Z",
            angle_degrees=a.twist_degrees,
            delay_seconds=0.5,
            ramp_seconds=2.0,
            other_body_pose_commands_after_reset=0,
            measurements=measurements,
            scope="Actual native key-only rotation and independently measured dynamic screw/fixture response. No prescribed screw rotation, thread model, torque calibration, or robot-policy success.",
        )
    result["adjacent_joint_collision_override_requested"] = a.enable_joint_collisions
    settings = result["joint_collision_settings"]
    result["adjacent_joint_collisions_enabled"] = bool(settings) and all(settings.values())
    result["adjacent_joint_collision_mode"] = (
        "no_joints"
        if not settings
        else "all"
        if all(settings.values())
        else "mixed"
        if any(settings.values())
        else "none"
    )
    result["collision_flag_definition"] = (
        "adjacent_joint_collisions_enabled describes effective settings for all authored joints; override_requested is only the CLI request. See per-joint settings for mixed configurations."
    )
    result["actual_render_mode"] = carb.settings.get_settings().get("/rtx/rendermode")
    result["image_capture_errors"] = image_capture_errors
    result["runtime_ccd"] = dict(
        requested=a.enable_ccd,
        scene_enabled=bool(
            PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath("/World/PhysicsScene"))
            .GetEnableCCDAttr()
            .Get()
        ),
        hybrid_sweep_and_speculative=a.enable_ccd,
        dynamic_bodies=ccd_bodies,
        initial_physics_dt=1 / (120 * initial_substeps),
        final_physics_dt=1 / (120 * a.substeps),
        fine_after_reference_step=a.fine_after_step,
        substeps_per_reference_step=a.substeps,
        reference_dt=1 / 120,
        reference_steps_per_state=a.steps,
        native_steps_per_state=a.fine_after_step + (a.steps - a.fine_after_step) * a.substeps,
        duration_seconds=a.steps / 120,
        geometry_mass_gravity_changed=False,
        scope="Explicit diagnostic/runtime collision-integration schedule; same physical duration, motion and timed releases as baseline. Do not attribute this result to the coarse default configuration. Future fast-motion drivers must use the qualified configuration.",
    )
    result["render_clock_checks"] = render_clock_checks
    result["render_transform_checks"] = render_transform_checks
    result["integration_step_method"] = (
        "World.step(render=False) for every explicit native integration call; camera capture uses World.render() with native clock and body-state invariance checks. Older retained diagnostics used rendered stepping near endpoints and must not be interpreted as measured exact native-step counts."
    )
    if ground_check:
        minima = {
            name: min(
                row["post_step_ground_clearance_m"][name]
                for state in results
                for row in state["trace"]
            )
            for name in list(leg_corners) + ([a.focus_asset] if focus_sphere else [])
        }
        result["release_clearance"] = dict(
            ground=ground_check,
            minimum_per_leg_m=minima,
            tolerance_m=0.001,
            all_checks_passed=all(value >= -0.001 for value in minima.values()),
            scope="Continuous per-step analytic Cube/ground clearance at the new explicitly enabled hybrid-CCD configuration.1mm criterion declared before this diagnostic; old failed contacts/poses remain retained.",
        )
        if focus_sphere:
            result["release_clearance"].update(
                focused_native_sphere=focus_sphere,
                scope="Continuous actualnativeSphere/compiledCube-plane clearance,1mm limit; nativebodygeometry/mass/gravity unchanged, explicit integration/CCD configuration. Free fall from resetheld/airborne poses is not a source throw/catch replay.",
            )
    result["extra_source_files"] = (
        {"scene_attachments.py": digest(out / "scene_attachments.py")}
        if spec.get("attachments")
        else {}
    )
    if a.mirror_kinematic_usd:
        result["extra_source_files"]["render_sync.py"] = digest(
            Path(__file__).parent / "render_sync.py"
        )
    result["attachment_initialization"] = (
        "Connections recreated only after all requested body poses and velocities have been reset; disabled for separated parts"
        if spec.get("attachments")
        else None
    )
    if a.attachment_retention:
        checks = []
        trace = results[0]["trace"]
        for index, c in enumerate(spec["attachments"]):
            release_step = a.steps if a.attachment_hold_only else 360 + 120 * index
            retained = [
                row["attachments"][c["id"]] for row in trace if 60 <= row["step"] < release_step
            ]
            final = trace[-1]["attachments"][c["id"]]
            anchor = max(row["anchor_error_m"] for row in retained)
            angle = max(row["relative_rotation_error_deg"] for row in retained)
            checks.append(
                dict(
                    id=c["id"],
                    release_step=None if a.attachment_hold_only else release_step,
                    enabled_samples=len(retained),
                    maximum_enabled_anchor_error_m=anchor,
                    maximum_enabled_rotation_error_deg=angle,
                    final_measurement=final,
                    retained=all(row["enabled"] for row in retained)
                    and anchor < 0.002
                    and angle < 2.0,
                    independent_after_release=None
                    if a.attachment_hold_only
                    else not final["enabled"] and final["anchor_error_m"] > 0.02,
                )
            )
        legs_dynamic = all(
            not body_physics[c["child"]]["kinematic"]
            and body_physics[c["child"]]["rigid_body_enabled"]
            and not body_physics[c["child"]]["gravity_disabled"]
            for c in spec["attachments"]
        )
        result["attachment_retention"] = dict(
            externally_held_panel=retention_panel,
            panel_kinematic_override=True,
            all_leg_bodies_dynamic_with_gravity=legs_dynamic,
            prescribed_body_ids=[retention_panel],
            leg_pose_commands_after_reset=0,
            lift_m=a.attachment_lift,
            rotation_deg=25,
            rotation_axis_panel_local="X",
            hold_only=a.attachment_hold_only,
            release_schedule_steps=[]
            if a.attachment_hold_only
            else [360 + 120 * i for i in range(len(spec["attachments"]))],
            checks=checks,
            all_checks_passed=legs_dynamic
            and all(
                c["retained"] and (a.attachment_hold_only or c["independent_after_release"])
                for c in checks
            ),
            scope="Scripted native panel-only lift/tilt; optional individual joint removal explicitly declared. No claim of robot execution or exact source removal trajectory. Base delivered assets remain dynamic; only this observed external-hold diagnostic makes the panel kinematic. Invertedhold and uprightrelease are separate functional conditions, not a claim that unsupported inverted falling-leg impacts cannot break finite-strength connections.",
        )
        result["attachment_retention"]["level_before_release"] = a.attachment_level_before_release
        result["attachment_retention"]["tilt_return_reference_interval"] = (
            [300, 350] if a.attachment_level_before_release else None
        )
        if a.attachment_level_before_release:
            result["attachment_retention"]["scope"] += (
                " In this distinct condition the held panel returns to the upright reference angle before unlocking eachleg; unsupported tilted releases remain negative stress evidence. No external hand force or grasp is applied to any leg."
            )
    result["effective_joint_drives"] = {
        k: dict(
            stiffness=v["drive"].GetStiffnessAttr().Get(),
            damping=v["drive"].GetDampingAttr().Get(),
            max_force=v["drive"].GetMaxForceAttr().Get(),
            drive_provenance=v["base"].GetPrim().GetCustomDataByKey("drive_provenance"),
        )
        for k, v in joint_data.items()
        if v["drive"]
    }
    result["angular_drive_units"] = dict(
        stiffness="N*m/degree",
        damping="N*m*s/degree",
        max_force="N*m",
        target_position="degrees",
        reference="https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html",
    )
    (out / "physics-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if image_capture_errors:
        raise RuntimeError(
            f"Native state image capture failed for {len(image_capture_errors)} views; complete numerical evidence and failed frames retained"
        )
    print("NATIVE_PHYSICS_PROBE_DONE", flush=True)
except BaseException as e:
    (out / "failure.json").write_text(
        json.dumps(dict(error=repr(e), traceback=traceback.format_exc()), indent=2)
    )
    print(traceback.format_exc(), flush=True)
    raise
finally:
    app.close()
