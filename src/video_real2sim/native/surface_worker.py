"""Native from-above contact checks against visible lid surfaces.

A small dynamic spherical contactor is a diagnostic device, not a robot. All
task bodies retain their authored mass, gravity and collision participation.
"""

import argparse
import fcntl
import hashlib
import json
import os
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("spec")
p.add_argument("scene")
p.add_argument("output")
p.add_argument("--limit", type=int)
p.add_argument("--sites", choices=["flat", "slopes"], default="flat")
p.add_argument("--diagnostic-hull-vertices", type=int)
p.add_argument("--physics-dt", type=float, default=1 / 120)
p.add_argument("--first-impulse", action="store_true")
p.add_argument("--plan", required=True)
a = p.parse_args()
probe_plan = json.loads(Path(a.plan).read_text())
lids = [x["asset_id"] for x in probe_plan["assets"]]
slope_radii = {x["asset_id"]: x["slope_radii_m"] for x in probe_plan["assets"]}
assert 0.0005 <= a.physics_dt <= 1 / 120
if a.diagnostic_hull_vertices is not None:
    assert 64 < a.diagnostic_hull_vertices <= 255
spec_path = Path(a.spec).resolve()
scene_path = Path(a.scene).resolve()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
spec = json.loads(spec_path.read_text())
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
        "max_specular_transmission_bounces": 16,
        "extra_args": ["--allow-root"],
    }
)
print("SURFACE_SIMULATION_APP_READY", flush=True)
from importlib.metadata import version

import carb
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.sensors.camera import Camera
from omni.physx import get_physx_simulation_interface
from PIL import Image, ImageDraw
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from surface_geometry import slope_sites, surface_sites, top_height, top_surface, visible_triangles


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rot(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


try:
    carb.settings.get_settings().set_bool("/app/viewport/grid/enabled", False)
    carb.settings.get_settings().set_bool("/app/viewport/defaults/guide/grid/visible", False)
    context = omni.usd.get_context()
    context.open_stage(str(scene_path))
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    radius = 0.001
    probe_path = "/World/SurfaceProbe"
    sphere = UsdGeom.Sphere.Define(stage, probe_path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.05, 0.05)])
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(0.00004)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
    collision.CreateContactOffsetAttr(0.0001)
    collision.CreateRestOffsetAttr(0.0)
    PhysxSchema.PhysxContactReportAPI.Apply(sphere.GetPrim()).CreateThresholdAttr().Set(0.0)
    paths = {
        asset["id"]: "/World/Assets/" + asset["id"]
        for asset in spec["assets"]
        if stage.GetPrimAtPath("/World/Assets/" + asset["id"]).HasAPI(UsdPhysics.RigidBodyAPI)
    }
    assert all(lid in paths for lid in lids)
    cooking_overrides = []
    for lid in lids:
        rb = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(paths[lid]))
        assert not rb.GetKinematicEnabledAttr().Get()
        PhysxSchema.PhysxContactReportAPI.Apply(rb.GetPrim()).CreateThresholdAttr().Set(0.0)
        if a.diagnostic_hull_vertices:
            prim = next(
                p for p in Usd.PrimRange(rb.GetPrim()) if p.GetName().endswith("upper_pressing_cap")
            )
            api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            previous = api.GetHullVertexLimitAttr().Get()
            api.CreateHullVertexLimitAttr(a.diagnostic_hull_vertices)
            cooking_overrides.append(
                dict(
                    prim_path=str(prim.GetPath()),
                    previous_hull_vertex_limit=previous,
                    new_hull_vertex_limit=a.diagnostic_hull_vertices,
                    scope="Diagnostic-only cooking intervention; original deliverable unchanged; cannot qualify as unmodified source scene",
                )
            )
    if cooking_overrides:
        stage.GetRootLayer().Export(str(out / "effective-diagnostic-scene.usda"))
    triangles = {lid: visible_triangles(stage, paths[lid]) for lid in lids}
    sites = [
        dict(lid=lid, **site)
        for lid in lids
        for site in (
            slope_sites(slope_radii[lid], triangles[lid])
            if a.sites == "slopes"
            else surface_sites(triangles[lid])
        )
    ]
    if a.limit is not None:
        sites = sites[: a.limit]
    assert sites
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=a.physics_dt,
        rendering_dt=1 / 30,
        physics_prim_path="/World/PhysicsScene",
    )
    bodies = {
        name: world.scene.add(SingleRigidPrim(prim_path=path, name=name))
        for name, path in paths.items()
    }
    probe = world.scene.add(SingleRigidPrim(prim_path=probe_path, name="surface_contactor"))
    camera_path = "/World/SurfaceInspectionCamera"
    UsdGeom.Camera.Define(stage, camera_path)
    camera = Camera(prim_path=camera_path, resolution=(960, 540), frequency=30)
    phase = dict(lid=None, contacts=[])
    callback_errors = []

    def contacts(headers, data):
        if phase["lid"] is None:
            return
        try:
            target = paths[phase["lid"]]
            for h in headers:
                actors = [
                    str(PhysicsSchemaTools.intToSdfPath(h.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(h.actor1)),
                ]
                if probe_path not in actors or target not in actors:
                    continue
                for index in range(
                    h.contact_data_offset, h.contact_data_offset + h.num_contact_data
                ):
                    c = data[index]
                    impulse = float(np.linalg.norm(c.impulse))
                    if impulse > 1e-9:
                        phase["contacts"].append(
                            dict(
                                position=[float(x) for x in c.position],
                                normal=[float(x) for x in c.normal],
                                separation_m=float(c.separation),
                                impulse_Ns=impulse,
                                actors=actors,
                                colliders=[
                                    str(PhysicsSchemaTools.intToSdfPath(getattr(h, k)))
                                    if hasattr(h, k)
                                    else None
                                    for k in ["collider0", "collider1"]
                                ],
                            )
                        )
        except Exception as e:
            if len(callback_errors) < 10:
                callback_errors.append(repr(e))

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    results = []
    frames = []
    for case_index, site in enumerate(sites):
        phase["lid"] = None
        world.reset()
        camera.initialize()
        probe.set_world_pose(position=np.array([0.0, 0.0, 10.0]))
        probe.set_linear_velocity(np.zeros(3))
        probe.set_angular_velocity(np.zeros(3))
        for _ in range(round(0.5 / a.physics_dt)):
            world.step(render=False)
        lid = site["lid"]
        body = bodies[lid]
        position, q = body.get_world_pose()
        rotation = rot(q)
        initial_position = np.asarray(position).copy()
        initial_rotation = rotation
        point = np.asarray(position) + rotation.apply([*site["xy"], site["visible_z"]])
        probe.set_world_pose(position=point + np.array([0, 0, radius + 0.0008]))
        probe.set_linear_velocity(np.zeros(3))
        probe.set_angular_velocity(np.zeros(3))
        phase["lid"] = lid
        found = None
        trace = []
        for step in range(round(0.3 / a.physics_dt)):
            phase["contacts"] = []
            world.step(render=False)
            position, q = body.get_world_pose()
            rotation = rot(q)
            center = probe.get_world_pose()[0]
            trace.append(
                dict(
                    step=step,
                    lid_position=np.asarray(position).tolist(),
                    lid_orientation_wxyz=np.asarray(q).tolist(),
                    probe_position=np.asarray(center).tolist(),
                    contacts=list(phase["contacts"]),
                )
            )
            eligible = (
                phase["contacts"]
                if a.first_impulse
                else [c for c in phase["contacts"] if abs(c["separation_m"]) < 0.0002]
            )
            if eligible:
                contact = (
                    max(eligible, key=lambda c: c["impulse_Ns"])
                    if a.first_impulse
                    else min(eligible, key=lambda c: abs(c["separation_m"]))
                )
                local = rotation.inv().apply(np.array(contact["position"]) - position)
                visible = top_surface(triangles[lid], float(local[0]), float(local[1]))
                if visible is not None:
                    expected, normal, triangle_index = visible
                    error = float(local[2] - expected)
                    actual_normal = rotation.inv().apply(contact["normal"])
                    actual_normal /= np.linalg.norm(actual_normal)
                    if actual_normal[2] < 0:
                        actual_normal = -actual_normal
                    normal_angle = float(
                        np.degrees(np.arccos(np.clip(np.dot(actual_normal, normal), -1, 1)))
                    )
                    probe_local = rotation.inv().apply(center - position)
                    sphere_surface = probe_local - radius * normal
                    plane_gap = float(
                        np.dot(sphere_surface - np.array([local[0], local[1], expected]), normal)
                    )
                    tolerance = 0.00025 if a.sites == "slopes" else 0.0005
                    radial_shift = float(np.linalg.norm(local[:2]) - np.linalg.norm(site["xy"]))
                    sampled_requested_slope = a.sites != "slopes" or (
                        abs(radial_shift) < 0.00075
                        and normal[2] < 0.999
                        and abs(expected - site["visible_z"]) < 0.001
                    )
                    passed = abs(error) < tolerance and (
                        a.sites != "slopes" or (abs(plane_gap) < tolerance and normal_angle < 15.0)
                    )
                    if a.first_impulse:
                        passed = (
                            passed
                            and abs(contact["separation_m"]) < 0.0002
                            and sampled_requested_slope
                        )
                    found = dict(
                        step=step,
                        contact=contact,
                        contact_asset_local=local.tolist(),
                        visible_surface_z_at_contact=expected,
                        contact_vs_visible_surface_error_m=error,
                        within_0_5mm=bool(abs(error) < 0.0005),
                        within_tolerance=bool(passed),
                        tolerance_m=tolerance,
                        visible_normal_local=normal.tolist(),
                        actual_normal_local=actual_normal.tolist(),
                        normal_error_deg=normal_angle,
                        visible_triangle_index=triangle_index,
                        probe_center_local=probe_local.tolist(),
                        sphere_surface_vs_visible_plane_gap_m=plane_gap,
                        lid_translation_m=float(np.linalg.norm(position - initial_position)),
                        lid_rotation_deg=float(
                            np.degrees((initial_rotation.inv() * rotation).magnitude())
                        ),
                        radial_shift_from_requested_site_m=radial_shift,
                        sampled_requested_slope=bool(sampled_requested_slope),
                    )
                    break
        phase["lid"] = None
        target = point
        eye = target + np.array([0, -0.12, 0.06])
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
        xf.ClearXformOpOrder()

        def v(x):
            return Gf.Vec3d(*map(float, x))

        xf.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(v(eye), v(target), Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        cam.GetHorizontalApertureAttr().Set(36.0)
        cam.GetVerticalApertureAttr().Set(20.25)
        cam.GetFocalLengthAttr().Set(28.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.001, 100))
        for _ in range(24):
            world.render()
        rgba = np.asarray(camera.get_rgba())
        assert rgba.shape == (540, 960, 4) and rgba[..., :3].std() > 2
        image = rgba[..., :3].astype(np.uint8)
        image_path = out / f"surface-{case_index:02d}.png"
        Image.fromarray(image).save(image_path)
        frames.append((f"{lid} | {site['name']} | actual dynamic contactor, no robot", image))
        results.append(
            dict(
                site=site,
                contact_found=found is not None,
                measurement=found,
                approach=dict(
                    direction_world=[0, 0, -1],
                    initial_drop_height_above_vertical_radius_m=0.0008,
                    initial_velocity_world=[0, 0, 0],
                    gravity_m_s2=9.81,
                    physics_dt=a.physics_dt,
                    contact_offset_m=0.0001,
                    rest_offset_m=0.0,
                    minimum_impulse_Ns=1e-9,
                    maximum_abs_separation_m=0.0002,
                    selection="First target contact with positive impulse; largest impulse if multiple simultaneous contacts"
                    if a.first_impulse
                    else "Legacy first near-zero separation positive-impulse contact; may be later than first impact",
                ),
                trace=trace,
                image=image_path.name,
                image_sha256=digest(image_path),
            )
        )
        (out / "partial-surface.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            "SURFACE_SITE_DONE",
            lid,
            site["name"],
            found["contact_vs_visible_surface_error_m"] if found else "no contact",
            flush=True,
        )
    for first in range(0, len(frames), 6):
        sheet = Image.new("RGB", (1920, 1692), "#101820")
        draw = ImageDraw.Draw(sheet)
        for j, (label, frame) in enumerate(frames[first : first + 6]):
            x = j % 2 * 960
            y = j // 2 * 564
            draw.text((x + 8, y + 3), label, fill="white")
            sheet.paste(Image.fromarray(frame), (x, y + 24))
        sheet.save(out / f"rest-review-{first // 6:02d}.jpg", quality=94)
    sections = []
    for lid in lids:
        for angle in [15.0, 135.0, 255.0]:
            phi = np.deg2rad(angle)
            maximum = float(np.linalg.norm(triangles[lid][..., :2], axis=2).max())
            sections.append(
                dict(
                    lid=lid,
                    azimuth_deg=angle,
                    visible_radial_section=[
                        [
                            float(r),
                            top_height(
                                triangles[lid], float(r * np.cos(phi)), float(r * np.sin(phi))
                            ),
                        ]
                        for r in np.linspace(0, maximum * 0.999, 300)
                    ],
                )
            )
    result = dict(
        scene_id=spec["scene_id"],
        native_runtime=True,
        isaacsim=version("isaacsim"),
        source_spec_sha256=digest(spec_path),
        probe_plan_sha256=digest(Path(a.plan)),
        source_usd_sha256=digest(scene_path),
        site_mode=a.sites,
        planned_case_count=len(sites),
        diagnostic_contactor=dict(
            shape="sphere", radius_m=radius, mass_kg=0.00004, gravity_enabled=True
        ),
        task_body_physics_changed=bool(cooking_overrides),
        collision_cooking_overrides=cooking_overrides,
        effective_diagnostic_usd_sha256=digest(out / "effective-diagnostic-scene.usda")
        if cooking_overrides
        else None,
        callback_errors=callback_errors,
        cases=results,
        visible_sections=sections,
        all_surface_sites_passed=not callback_errors
        and all(
            r["measurement"] is not None and r["measurement"]["within_tolerance"] for r in results
        ),
        scope="Actual native contact locations/normals and sphere surface compared with compiled visible USD triangles, excluding invisible collision proxies. Slopes target0.25mm and15degrees normal error; flat sites0.5mm. Not a robot policy, calibrated pressing force or snap/seal validation.",
    )
    result.update(
        physics_dt=a.physics_dt,
        first_impulse_measurement=a.first_impulse,
        diagnostic_integration_changed=a.physics_dt != 1 / 120,
        slope_site_coverage_enforced=a.first_impulse,
    )
    (out / "surface-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SURFACE_PROBE_DONE", result["all_surface_sites_passed"], flush=True)
except BaseException as e:
    (out / "failure.json").write_text(
        json.dumps(dict(error=repr(e), traceback=traceback.format_exc()), indent=2)
    )
    print(traceback.format_exc(), flush=True)
    raise
finally:
    app.close()
