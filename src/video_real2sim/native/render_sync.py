"""Bridge measured native kinematic poses to USD at camera capture boundaries.

PhysX writes dynamic transforms back; tensor-driven kinematics require explicit
USD authoring for the renderer. Never prescribe any dynamic body through this
bridge. All body poses and velocities are audited unchanged after rendering.
"""

import numpy as np
from omni.physx import get_physx_interface
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation


def read_bodies(bodies):
    values = {}
    for name, body in bodies.items():
        p, q = body.get_world_pose()
        values[name] = dict(
            position=np.asarray(p).copy(),
            orientation=np.asarray(q).copy(),
            velocity=np.concatenate(
                [body.get_linear_velocity(), body.get_angular_velocity()]
            ).copy(),
        )
    return values


def prepare(stage, bodies, paths, held_names):
    before = read_bodies(bodies)
    get_physx_interface().update_transformations(True, True, True, False)
    cache = UsdGeom.XformCache()
    for name in held_names:
        prim = stage.GetPrimAtPath(paths[name])
        value = before[name]
        q = value["orientation"]
        quat = Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))).GetNormalized()
        matrix = Gf.Matrix4d().SetRotate(quat)
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float, value["position"])))
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        local = matrix * parent.GetInverse()
        # This authors the already achieved pose of the externally controlled
        # kinematic body for rendering. It is not a dynamic-body pose setter.
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(local)
    return before


def audit(stage, bodies, paths, before):
    after = read_bodies(bodies)
    cache = UsdGeom.XformCache()
    checks = {}
    for name, value in after.items():
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(paths[name]))
        usd_position = np.array(matrix.ExtractTranslation())
        q = matrix.ExtractRotationQuat()
        usd_rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()])
        rotation = Rotation.from_quat(value["orientation"][[1, 2, 3, 0]])
        old = before[name]
        old_rotation = Rotation.from_quat(old["orientation"][[1, 2, 3, 0]])
        translation_error = float(np.linalg.norm(usd_position - value["position"]))
        angle = float(np.degrees((rotation.inv() * usd_rotation).magnitude()))
        pose_changed = (
            float(np.linalg.norm(value["position"] - old["position"])) > 1e-8
            or (old_rotation.inv() * rotation).magnitude() > 1e-8
        )
        velocity_change = float(np.max(np.abs(value["velocity"] - old["velocity"])))
        checks[name] = dict(
            actual_position=value["position"].tolist(),
            usd_position=usd_position.tolist(),
            translation_error_m=translation_error,
            rotation_error_deg=angle,
            actual_pose_unchanged_during_render=not pose_changed,
            maximum_actual_velocity_change=velocity_change,
        )
    return dict(
        bodies=checks,
        passed=all(
            c["translation_error_m"] < 0.0001
            and c["rotation_error_deg"] < 0.05
            and c["actual_pose_unchanged_during_render"]
            and c["maximum_actual_velocity_change"] < 1e-7
            for c in checks.values()
        ),
        scope="Physics-to-render bridge, all actual native bodies/velocities audited unchanged; only measured externally controlled kinematic root USD transforms authored, never dynamic leg/screw poses.",
    )
