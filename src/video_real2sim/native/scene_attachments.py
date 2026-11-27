"""Explicit detachable fixed-joint approximation for source-observed assemblies.

Import only after SimulationApp starts on Isaac. No snap-to-target or leg pose
controller is implemented here; enabling a connection only toggles its joint.
"""

import numpy as np
from pxr import Gf, UsdPhysics
from scipy.spatial.transform import Rotation


def poses_for(spec, state):
    poses = {
        a["id"]: dict(position=a["position"], rotation_deg=a["rotation_deg"])
        for a in spec["assets"]
    }
    poses.update({p["asset_id"]: p for p in state.get("poses", [])})
    return poses


def frame(spec, connection):
    poses = poses_for(spec, spec["diagnostic_states"][connection["reference_state_index"]])
    parent = poses[connection["parent"]]
    child = poses[connection["child"]]
    rp = Rotation.from_euler("xyz", parent["rotation_deg"], degrees=True)
    rc = Rotation.from_euler("xyz", child["rotation_deg"], degrees=True)
    error = (
        np.array(parent["position"])
        + rp.apply(connection["parent_anchor"])
        - np.array(child["position"])
        - rc.apply(connection["child_anchor"])
    )
    assert np.linalg.norm(error) < 0.0005, "Assembly anchors must match the explicit reference pose"
    return rp.inv() * rc, float(np.linalg.norm(error))


def path(connection):
    return "/World/Attachments/" + connection["id"]


def build_attachments(stage, spec):
    report = []
    for connection in spec.get("attachments", []):
        local, error = frame(spec, connection)
        q = local.as_quat()
        joint = UsdPhysics.FixedJoint.Define(stage, path(connection))
        joint.CreateBody0Rel().SetTargets(["/World/Assets/" + connection["parent"]])
        joint.CreateBody1Rel().SetTargets(["/World/Assets/" + connection["child"]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*connection["parent_anchor"]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*connection["child_anchor"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        joint.CreateCollisionEnabledAttr(connection["collision_enabled"])
        joint.CreateJointEnabledAttr(False)
        joint.CreateBreakForceAttr(float(connection["break_force_N"]))
        joint.CreateBreakTorqueAttr(float(connection["break_torque_Nm"]))
        joint.GetPrim().SetCustomDataByKey(
            "source_assembly_approximation", connection["approximation_note"]
        )
        report.append(
            dict(
                id=connection["id"],
                parent=connection["parent"],
                child=connection["child"],
                reference_state_index=connection["reference_state_index"],
                reference_anchor_error_m=error,
                initially_enabled=False,
                collision_enabled=connection["collision_enabled"],
                mechanism="Individually enabled/disabled native PhysX fixed joint; abstract fastening, no recovered screw thread/torque",
            )
        )
    return report


def set_enabled_attachments(stage, spec, state):
    enabled = set(state.get("enabled_attachments", []))
    known = {c["id"] for c in spec.get("attachments", [])}
    assert enabled <= known
    for connection in spec.get("attachments", []):
        UsdPhysics.Joint(stage.GetPrimAtPath(path(connection))).GetJointEnabledAttr().Set(
            connection["id"] in enabled
        )
    return sorted(enabled)


def measure_attachments(stage, spec, bodies):
    result = {}
    for connection in spec.get("attachments", []):
        parent = bodies[connection["parent"]]
        child = bodies[connection["child"]]
        rp = Rotation.from_quat(np.array(parent["orientation_wxyz"])[[1, 2, 3, 0]])
        rc = Rotation.from_quat(np.array(child["orientation_wxyz"])[[1, 2, 3, 0]])
        local, _ = frame(spec, connection)
        delta = (
            np.array(parent["position"])
            + rp.apply(connection["parent_anchor"])
            - np.array(child["position"])
            - rc.apply(connection["child_anchor"])
        )
        result[connection["id"]] = dict(
            enabled=bool(
                UsdPhysics.Joint(stage.GetPrimAtPath(path(connection))).GetJointEnabledAttr().Get()
            ),
            anchor_error_m=float(np.linalg.norm(delta)),
            relative_rotation_error_deg=float(np.degrees(((rp * local).inv() * rc).magnitude())),
        )
    return result
