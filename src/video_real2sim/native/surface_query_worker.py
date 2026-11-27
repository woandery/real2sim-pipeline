"""Independent native PhysX collider queries, without a contactor or collider override."""

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
p.add_argument("--plan", required=True)
a = p.parse_args()
probe_plan = json.loads(Path(a.plan).read_text())
lids = [x["asset_id"] for x in probe_plan["assets"]]
slope_radii = {x["asset_id"]: x["slope_radii_m"] for x in probe_plan["assets"]}
spec_path = Path(a.spec).resolve()
usd = Path(a.scene).resolve()
out = Path(a.output).resolve()
out.mkdir(exist_ok=True)
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
from importlib.metadata import version

import carb
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from omni.physx import get_physx_scene_query_interface
from pxr import PhysxSchema, Usd, UsdPhysics
from scipy.spatial.transform import Rotation
from surface_geometry import slope_sites, surface_sites, top_surface, visible_triangles


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


try:
    context = omni.usd.get_context()
    context.open_stage(str(usd))
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    triangles = {lid: visible_triangles(stage, "/World/Assets/" + lid) for lid in lids}
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / 120,
        rendering_dt=1 / 30,
        physics_prim_path="/World/PhysicsScene",
    )
    bodies = {}
    body_physics = {}
    for asset in spec["assets"]:
        prim = stage.GetPrimAtPath("/World/Assets/" + asset["id"])
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        body_physics[asset["id"]] = dict(
            kinematic=bool(rb.GetKinematicEnabledAttr().Get()),
            gravity_disabled=bool(
                PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
            ),
        )
        bodies[asset["id"]] = world.scene.add(
            SingleRigidPrim(prim_path=str(prim.GetPath()), name=asset["id"])
        )
    world.reset()
    for _ in range(60):
        world.step(render=False)
    query = get_physx_scene_query_interface()
    cases = []
    for lid in lids:
        assert not body_physics[lid]["kinematic"] and not body_physics[lid]["gravity_disabled"]
        position, q = bodies[lid].get_world_pose()
        rotation = Rotation.from_quat(np.array(q)[[1, 2, 3, 0]])
        for site in surface_sites(triangles[lid]) + slope_sites(slope_radii[lid], triangles[lid]):
            origin = np.array(position) + rotation.apply([*site["xy"], site["visible_z"] + 0.02])
            direction = rotation.apply([0, 0, -1])
            hit = query.raycast_closest(
                carb.Float3(*map(float, origin)), carb.Float3(*map(float, direction)), 0.1
            )
            result = dict(
                site=dict(lid=lid, **site),
                origin_world=origin.tolist(),
                direction_world=direction.tolist(),
                hit=bool(hit.get("hit")),
                correct_body=False,
            )
            if hit.get("hit"):
                local = rotation.inv().apply(np.array(hit["position"]) - position)
                normal = rotation.inv().apply(np.array(hit["normal"]))
                normal /= np.linalg.norm(normal)
                if normal[2] < 0:
                    normal = -normal
                expected, visible_normal, triangle_index = top_surface(
                    triangles[lid], float(local[0]), float(local[1])
                )
                error = float(local[2] - expected)
                angle = float(np.degrees(np.arccos(np.clip(np.dot(normal, visible_normal), -1, 1))))
                result.update(
                    correct_body=str(hit["rigidBody"]) == "/World/Assets/" + lid,
                    collider=str(hit["collision"]),
                    body=str(hit["rigidBody"]),
                    face_index=int(hit["faceIndex"]),
                    position_world=[float(v) for v in hit["position"]],
                    local_position=local.tolist(),
                    native_normal_local=normal.tolist(),
                    visible_normal_local=visible_normal.tolist(),
                    visible_z=expected,
                    geometric_offset_m=error,
                    normal_error_deg=angle,
                    visible_triangle_index=triangle_index,
                    within_0_25mm=abs(error) < 0.00025,
                )
            cases.append(result)
    report = dict(
        scene_id=spec["scene_id"],
        native_runtime=True,
        isaacsim=version("isaacsim"),
        source_spec_sha256=digest(spec_path),
        probe_plan_sha256=digest(Path(a.plan)),
        source_usd_sha256=digest(usd),
        task_body_physics_changed=False,
        body_physics=body_physics,
        physics_steps=60,
        physics_dt=1 / 120,
        cases=cases,
        all_native_surfaces_within_0_25mm=all(
            c["correct_body"] and c.get("within_0_25mm", False) for c in cases
        ),
        method="omni.physx.get_physx_scene_query_interface().raycast_closest against actual native colliders; independent of contact-report point generation; compiled visible USD triangle intersection is the reference",
        api_reference="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html",
        scope="Native geometric query after free settling, with original dynamic bodies/gravity/colliders and no contactor. Does not demonstrate contact-force accuracy, robot execution or exact calibrated source geometry.",
    )
    (out / "surface-query.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SURFACE_QUERY_DONE", report["all_native_surfaces_within_0_25mm"], flush=True)
except BaseException as error:
    (out / "failure.json").write_text(
        json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2) + "\n"
    )
    raise
finally:
    app.close()
