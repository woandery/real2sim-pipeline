"""Compile declarative geometry and capture native Isaac Sim evidence.

Diagnostic source states are explicitly posed inspections, never robot execution.
Run with the separately configured Isaac Sim Python interpreter.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("spec")
p.add_argument("output")
p.add_argument("--record", action="store_true")
a = p.parse_args()
spec_path = Path(a.spec).resolve()
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
        try:
            if (
                "nvidia"
                not in json.loads(Path(candidate).read_text())["ICD"]["library_path"].lower()
            ):
                continue
        except Exception:
            continue
        os.environ["VK_DRIVER_FILES"] = candidate
        os.environ["VK_ICD_FILENAMES"] = candidate
        break
from isaacsim import SimulationApp

render_settings = spec.get("render_settings", {})
renderer = render_settings.get("renderer", "PathTracing")
assert renderer in ["RayTracedLighting", "PathTracing"]
app = SimulationApp(
    {
        "headless": True,
        "width": 960,
        "height": 540,
        "renderer": renderer,
        "samples_per_pixel_per_frame": int(render_settings.get("samples_per_pixel", 16)),
        "max_bounces": 12,
        "max_specular_transmission_bounces": 16,
        "extra_args": ["--allow-root"],
    }
)
import carb
import imageio.v2 as imageio
import numpy as np
import omni.usd

carb.settings.get_settings().set_bool("/app/viewport/grid/enabled", False)
carb.settings.get_settings().set_bool("/app/viewport/defaults/guide/grid/visible", False)
from importlib.metadata import version

from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scene_attachments import build_attachments, set_enabled_attachments
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

GLASS_TEMPLATE_LAYER = None
GLASS_TEMPLATE_PATH = "/World/Materials/NativeGlassTemplate"


def vec(x):
    return Gf.Vec3d(*map(float, x))


def transform(prim, position, rotation, size=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(vec(position))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rotation)))
    if size is not None:
        xf.AddScaleOp().Set(vec(size))


def lathe(stage, path, profile, segments=48):
    points = []
    counts = []
    indices = []
    for r, z in profile:
        for j in range(segments):
            points.append(
                (
                    r * math.cos(j * 2 * math.pi / segments),
                    r * math.sin(j * 2 * math.pi / segments),
                    z,
                )
            )
    for i in range(len(profile) - 1):
        for j in range(segments):
            nj = (j + 1) % segments
            counts.append(4)
            indices.extend(
                [
                    i * segments + j,
                    i * segments + nj,
                    (i + 1) * segments + nj,
                    (i + 1) * segments + j,
                ]
            )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def lathe_sector(stage, path, profile, start, end):
    """Closed convex radial-profile wedge, with coincident axis points merged.

    Each wedge is cooked independently. No global convex hull bridges or
    decimates the complete circumferential pressing surface.
    """
    assert profile[0] == profile[-1] and 0 < end - start <= math.pi / 2
    ring = profile[:-1]
    points = []
    lookup = {}
    edges = []
    for angle in [start, end]:
        edge = []
        for radius, z in ring:
            point = (radius * math.cos(angle), radius * math.sin(angle), z)
            if point not in lookup:
                lookup[point] = len(points)
                points.append(point)
            edge.append(lookup[point])
        edges.append(edge)
    faces = [edges[0][::-1], edges[1]]
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        faces.append([edges[0][i], edges[0][j], edges[1][j], edges[1][i]])
    faces = [list(dict.fromkeys(face)) for face in faces]
    faces = [face for face in faces if len(face) >= 3]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def material(stage, color, opacity, index, root="/World"):
    path = f"{root}/Materials/m{index}"
    if 0 < float(opacity) < 0.999 and GLASS_TEMPLATE_LAYER is not None:
        # Fractional PreviewSurface alpha produced verified black/depth artifacts
        # even over opaque cabinet tops. Use actual MDL transmission instead.
        UsdGeom.Scope.Define(stage, f"{root}/Materials")
        Sdf.CopySpec(GLASS_TEMPLATE_LAYER, GLASS_TEMPLATE_PATH, stage.GetRootLayer(), path)
        mat = UsdShade.Material(stage.GetPrimAtPath(path))
        shader = next(
            UsdShade.Shader(p) for p in mat.GetPrim().GetChildren() if p.IsA(UsdShade.Shader)
        )
        tint = np.clip(np.array(color, float) * float(opacity) + 0.99 * (1 - float(opacity)), 0, 1)
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*map(float, tint))
        )
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.49)
        shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.001)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.12)
        mat.GetPrim().SetCustomDataByKey("video_alpha_hint", float(opacity))
        return mat
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def make_part(stage, path, part, index):
    shape = part["shape"]
    size = part["size"]
    scale = None
    if shape == "box":
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1)
        scale = size
    elif shape == "sphere":
        if max(size) - min(size) > 1e-8:
            profile = [
                (float(size[0]) / 2 * math.sin(t), float(size[2]) / 2 * math.cos(t))
                for t in np.linspace(0, math.pi, 13)
            ]
            geom = lathe(stage, path, profile, segments=24)
            points = geom.GetPointsAttr().Get()
            geom.GetPointsAttr().Set(
                [
                    Gf.Vec3f(
                        float(q[0]), float(q[1]) * float(size[1]) / float(size[0]), float(q[2])
                    )
                    for q in points
                ]
            )
        else:
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(0.5)
            scale = size
    elif shape in ["cylinder", "cone", "capsule"]:
        cls = {"cylinder": UsdGeom.Cylinder, "cone": UsdGeom.Cone, "capsule": UsdGeom.Capsule}[
            shape
        ]
        if abs(float(size[1]) - float(size[0])) > 1e-8 or (
            shape == "capsule" and size[2] < size[0]
        ):
            r = float(size[0]) / 2
            h = float(size[2])
            if shape == "capsule":
                center = max(0.0, (h - 2 * r) / 2)
                zscale = min(1.0, h / (2 * r))
                profile = [
                    (r * math.sin(t), (center + r * math.cos(t)) * zscale)
                    for t in np.linspace(0, math.pi / 2, 7)
                ]
                profile += [
                    (r * math.sin(t), (-center + r * math.cos(t)) * zscale)
                    for t in np.linspace(math.pi / 2, math.pi, 7)
                ]
            elif shape == "cone":
                profile = [(0, h / 2), (r, -h / 2), (0, -h / 2)]
            else:
                profile = [(0, -h / 2), (r, -h / 2), (r, h / 2), (0, h / 2)]
            geom = lathe(stage, path, profile, segments=24)
            points = geom.GetPointsAttr().Get()
            geom.GetPointsAttr().Set(
                [
                    Gf.Vec3f(
                        float(q[0]), float(q[1]) * float(size[1]) / float(size[0]), float(q[2])
                    )
                    for q in points
                ]
            )
        else:
            geom = cls.Define(stage, path)
            geom.CreateRadiusAttr(float(size[0]) / 2)
            height = (
                max(0.0, float(size[2]) - float(size[0])) if shape == "capsule" else float(size[2])
            )
            geom.CreateHeightAttr(height)
            geom.CreateAxisAttr("Z")
    elif shape == "lathe" and part.get("convex_sectors"):
        assert part["collision"] and float(part.get("opacity", 1)) == 0
        geom = UsdGeom.Xform.Define(stage, path)
        count = int(part["convex_sectors"])
        for sector in range(count):
            mesh = lathe_sector(
                stage,
                f"{path}/sector_{sector:03d}",
                part["profile"],
                sector * 2 * math.pi / count,
                (sector + 1) * 2 * math.pi / count,
            )
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            settings = part.get("convex_hull", {"hull_vertex_limit": 64, "min_thickness": 0.0001})
            api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim())
            api.CreateHullVertexLimitAttr(int(settings["hull_vertex_limit"]))
            api.CreateMinThicknessAttr(float(settings["min_thickness"]))
        geom.GetPrim().SetCustomDataByKey(
            "collision_construction",
            "Independently cooked closed radial-profile convex sectors; explicit compiler4 representation",
        )
    else:
        geom = lathe(stage, path, part["profile"], segments=int(part.get("segments", 48)))
    transform(geom.GetPrim(), part["position"], part["rotation_deg"], scale)
    root = "/" + path.strip("/").split("/")[0]
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
        material(stage, part["color"], part.get("opacity", 1), index, root)
    )
    if float(part.get("opacity", 1)) <= 0:
        UsdGeom.Imageable(geom.GetPrim()).MakeInvisible()
    if part.get("collision", True) and not part.get("convex_sectors"):
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        if geom.GetPrim().IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
            if "convex_hull" in part:
                cooking = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(geom.GetPrim())
                settings = part["convex_hull"]
                cooking.CreateHullVertexLimitAttr(int(settings["hull_vertex_limit"]))
                cooking.CreateMinThicknessAttr(float(settings["min_thickness"]))
    if "contact_offset_m" in part or "rest_offset_m" in part:
        assert part.get("collision", True)
        for prim in Usd.PrimRange(geom.GetPrim()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if "contact_offset_m" in part:
                api.CreateContactOffsetAttr(float(part["contact_offset_m"]))
            if "rest_offset_m" in part:
                api.CreateRestOffsetAttr(float(part["rest_offset_m"]))
            prim.SetCustomDataByKey(
                "collision_offsets_provenance",
                "Explicit part fields in compiled source specification; runtime tensor readback remains required",
            )
    return geom


def set_camera(stage, path, eye, target, fov, up=None):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(vec(eye), vec(target), vec(up or [0, 0, 1])).GetInverse()
    )
    cam = UsdGeom.Camera(prim)
    cam.GetProjectionAttr().Set("perspective")
    cam.GetHorizontalApertureAttr().Set(36.0)
    cam.GetVerticalApertureAttr().Set(20.25)
    cam.GetFocalLengthAttr().Set(20.25 / (2 * math.tan(math.radians(fov) / 2)))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.005, 1000))


def set_asset_camera(stage, path, eye, target, vertical_extent, up=None):
    set_camera(stage, path, eye, target, 42, up)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
    cam.GetProjectionAttr().Set("orthographic")
    # USD apertures are tenths of a scene unit; this stage uses metres.
    cam.GetVerticalApertureAttr().Set(float(vertical_extent) * 10)
    cam.GetHorizontalApertureAttr().Set(float(vertical_extent) * 10 * 960 / 540)


def review_sheet(items, path, label):
    """Keep every native 960x540 pixel; no anisotropic review resizing."""
    sheet = Image.new("RGB", (1920, 1692), "#101820")
    draw = ImageDraw.Draw(sheet)
    for j, item in enumerate(items):
        x = j % 2 * 960
        y = j // 2 * 564
        frame = item[-1]
        assert frame.shape == (540, 960, 3)
        draw.text((x + 8, y + 3), label(item), fill="white")
        sheet.paste(Image.fromarray(frame), (x, y + 24))
    sheet.save(path, quality=94)


def collider_wire(stage, asset_path, width):
    """Authored primitive/convex-hull wire; native cooking may simplify it."""
    vertices = []
    counts = []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(asset_path)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        points = []
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        elif prim.IsA(UsdGeom.Cube):
            h = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            points = np.array([[x, y, z] for x in [-h, h] for y in [-h, h] for z in [-h, h]])
        elif prim.IsA(UsdGeom.Sphere):
            r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            points = np.array(
                [
                    [r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)]
                    for t in np.linspace(0, math.pi, 9)
                    for p in np.linspace(0, 2 * math.pi, 16, endpoint=False)
                ]
            )
        elif any(prim.IsA(c) for c in [UsdGeom.Cylinder, UsdGeom.Cone, UsdGeom.Capsule]):
            geom = (
                UsdGeom.Cylinder(prim)
                if prim.IsA(UsdGeom.Cylinder)
                else UsdGeom.Cone(prim)
                if prim.IsA(UsdGeom.Cone)
                else UsdGeom.Capsule(prim)
            )
            r = float(geom.GetRadiusAttr().Get())
            h = float(geom.GetHeightAttr().Get())
            profile = [(r, -h / 2), (r, h / 2)]
            if prim.IsA(UsdGeom.Cone):
                profile = [(r, -h / 2), (0, h / 2)]
            if prim.IsA(UsdGeom.Capsule):
                profile = [
                    (r * math.sin(t), h / 2 + r * math.cos(t))
                    for t in np.linspace(0, math.pi / 2, 7)
                ] + [
                    (r * math.sin(t), -h / 2 + r * math.cos(t))
                    for t in np.linspace(math.pi / 2, math.pi, 7)
                ]
            points = np.array(
                [
                    [rad * math.cos(p), rad * math.sin(p), z]
                    for rad, z in profile
                    for p in np.linspace(0, 2 * math.pi, 16, endpoint=False)
                ]
            )
        if len(points) < 4:
            continue
        points = np.unique(np.round(points, 10), axis=0)
        try:
            faces = ConvexHull(points).simplices
        except Exception:
            continue
        edges = {
            tuple(sorted((int(face[k]), int(face[(k + 1) % 3]))))
            for face in faces
            for k in range(3)
        }
        matrix = cache.GetLocalToWorldTransform(prim)
        for i, j in edges:
            vertices.extend(
                [matrix.TransformAffine(vec(points[i])), matrix.TransformAffine(vec(points[j]))]
            )
            counts.append(2)
    curve = UsdGeom.BasisCurves.Define(stage, "/World/InspectionColliderWire")
    curve.CreateTypeAttr("linear")
    curve.CreatePointsAttr(vertices)
    curve.CreateCurveVertexCountsAttr(counts)
    curve.CreateWidthsAttr([float(width)])
    curve.SetWidthsInterpolation("constant")
    mat = material(stage, [0.05, 0.95, 0.1], 1, 999999)
    shader = UsdShade.Shader(stage.GetPrimAtPath(str(mat.GetPath()) + "/surface"))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.9, 0.1))
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(mat)
    return len(counts)


started = time.time()
try:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    if any(
        0 < float(p.get("opacity", 1)) < 0.999 for asset in spec["assets"] for p in asset["parts"]
    ):
        from isaacsim.core.api.materials import OmniGlass

        native_glass = OmniGlass(
            prim_path=GLASS_TEMPLATE_PATH,
            color=np.array([0.98, 0.98, 0.98]),
            ior=1.49,
            depth=0.001,
            thin_walled=True,
        )
        GLASS_TEMPLATE_LAYER = stage.GetRootLayer()
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    light = UsdLux.DomeLight.Define(stage, "/World/Dome")
    light.CreateIntensityAttr(750.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Key")
    sun.CreateIntensityAttr(500.0)
    sun.CreateAngleAttr(20.0)
    transform(sun.GetPrim(), [0, 0, 4], [-35, -25, 0])
    g = spec["ground"]
    make_part(
        stage,
        "/World/Ground",
        dict(
            shape="box",
            size=g["size"],
            position=g["position"],
            rotation_deg=[0, 0, 0],
            color=g["color"],
            collision=True,
        ),
        0,
    )
    assets = {}
    parts_count = 0
    joint_parents = {j["parent"] for j in spec.get("joints", [])}
    for ai, asset in enumerate(spec["assets"]):
        path = f"/World/Assets/{asset['id']}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        transform(prim, asset["position"], asset["rotation_deg"])
        assets[asset["id"]] = path
        if asset["body_type"] != "static" or asset["id"] in joint_parents:
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateKinematicEnabledAttr(asset["body_type"] != "dynamic")
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(max(0.001, float(asset["mass_kg"])))
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(16)
        for pi, part in enumerate(asset["parts"]):
            make_part(stage, f"{path}/p{pi}_{part['id']}", part, 1 + ai * 1000 + pi)
            parts_count += 1
    drives = {}
    joint_anchor_checks = []
    authored_joint_drives = {}
    for j in spec.get("joints", []):
        path = "/World/Joints/" + j["id"]
        cls = {
            "prismatic": UsdPhysics.PrismaticJoint,
            "revolute": UsdPhysics.RevoluteJoint,
            "fixed": UsdPhysics.FixedJoint,
        }[j["type"]]
        joint = cls.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([assets[j["parent"]]])
        joint.CreateBody1Rel().SetTargets([assets[j["child"]]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, j["parent_anchor"])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, j["child_anchor"])))
        # Preserve the supplied initial orientations. Identity local frames would
        # twist a flipped box lid to match its upside-down parent on reset.
        parent = next(x for x in spec["assets"] if x["id"] == j["parent"])
        child = next(x for x in spec["assets"] if x["id"] == j["child"])

        def rotation_of(euler):
            xf = (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), float(euler[0]))
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(euler[1]))
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(euler[2]))
            )
            return xf

        rp = (
            UsdGeom.Xformable(stage.GetPrimAtPath(assets[j["parent"]]))
            .GetLocalTransformation()
            .ExtractRotation()
        )
        rc = (
            UsdGeom.Xformable(stage.GetPrimAtPath(assets[j["child"]]))
            .GetLocalTransformation()
            .ExtractRotation()
        )
        axis = Gf.Vec3d(*[1 if k == "XYZ".index(j["axis"]) else 0 for k in range(3)])
        offset = (
            Gf.Rotation(axis, float(j["initial_position"]))
            if j["type"] == "revolute"
            else Gf.Rotation()
        )
        relative = rc.GetQuat().GetInverse() * rp.GetQuat() * offset.GetQuat()
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(relative))
        p0 = vec(parent["position"]) + rp.TransformDir(vec(j["parent_anchor"]))
        p1 = vec(child["position"]) + rc.TransformDir(vec(j["child_anchor"]))
        delta = p1 - p0
        if j["type"] == "prismatic":
            delta -= rp.TransformDir(axis) * float(j["initial_position"])
        joint_anchor_checks.append(dict(joint_id=j["id"], anchor_error_m=delta.GetLength()))
        joint.CreateCollisionEnabledAttr(bool(j.get("collision_enabled", False)))
        if j["type"] != "fixed":
            joint.CreateAxisAttr(j["axis"])
            joint.CreateLowerLimitAttr(float(j["lower"]))
            joint.CreateUpperLimitAttr(float(j["upper"]))
            drive = UsdPhysics.DriveAPI.Apply(
                joint.GetPrim(), "linear" if j["type"] == "prismatic" else "angular"
            )
            settings = j.get("drive", dict(stiffness=250.0, damping=40.0, max_force=1000.0))
            source = (
                "explicit_spec" if "drive" in j else "legacy_compiler_diagnostic_hold_unmeasured"
            )
            if settings is None:
                settings = dict(stiffness=0.0, damping=0.0, max_force=0.0)
                source = "explicit_passive"
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(settings["stiffness"]))
            drive.CreateDampingAttr(float(settings["damping"]))
            drive.CreateMaxForceAttr(float(settings["max_force"]))
            drive.CreateTargetPositionAttr(float(j["initial_position"]))
            drives[j["id"]] = drive
            joint.GetPrim().SetCustomDataByKey("drive_provenance", source)
            units = (
                dict(
                    stiffness="N*m/degree",
                    damping="N*m*s/degree",
                    max_force="N*m",
                    target_position="degrees",
                )
                if j["type"] == "revolute"
                else dict(stiffness="N/m", damping="N*s/m", max_force="N", target_position="metres")
            )
            authored_joint_drives[j["id"]] = dict(
                **settings,
                target_position=float(j["initial_position"]),
                provenance=source,
                units=units,
                calibrated_from_video=False,
            )
    camera_path = "/World/SourceCamera"
    UsdGeom.Camera.Define(stage, camera_path)
    cam = spec["camera"]
    set_camera(stage, camera_path, cam["position"], cam["target"], cam["vertical_fov_deg"])
    asset_dir = out / "assets"
    asset_dir.mkdir(exist_ok=True)
    for asset in spec["assets"]:
        s = Usd.Stage.CreateNew(str(asset_dir / f"{asset['id']}.usda"))
        prim = UsdGeom.Xform.Define(s, "/Asset").GetPrim()
        s.SetDefaultPrim(prim)
        if asset["body_type"] != "static":
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(
                asset["body_type"] == "kinematic"
            )
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(max(0.001, float(asset["mass_kg"])))
        for pi, part in enumerate(asset["parts"]):
            make_part(s, f"/Asset/p{pi}_{part['id']}", part, pi)
        UsdGeom.SetStageMetersPerUnit(s, 1)
        UsdGeom.SetStageUpAxis(s, "Z")
        s.GetRootLayer().Save()
    stage.GetRootLayer().Export(str(out / "scene.usda"))
    carb.settings.get_settings().set("/physics/updateToUsd", True)
    attachment_bindings = build_attachments(stage, spec)
    set_enabled_attachments(stage, spec, spec["diagnostic_states"][0])
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1 / 120,
        rendering_dt=1 / 30,
        physics_prim_path="/World/PhysicsScene",
    )
    trackers = {}
    for asset in spec["assets"]:
        if asset["body_type"] != "static" or asset["id"] in joint_parents:
            trackers[asset["id"]] = world.scene.add(
                SingleRigidPrim(prim_path=assets[asset["id"]], name=asset["id"])
            )
    camera = Camera(prim_path=camera_path, resolution=(960, 540), frequency=30)
    world.reset()
    camera.initialize()
    source_positions = {k: np.asarray(v.get_world_pose()[0]).tolist() for k, v in trackers.items()}
    for _ in range(24):
        world.step(render=False)
    initial_positions = {k: np.asarray(v.get_world_pose()[0]).tolist() for k, v in trackers.items()}
    for _ in range(120):
        world.step(render=False)
    rest = []
    for k, v in trackers.items():
        pos = np.asarray(v.get_world_pose()[0])
        rest.append(
            dict(
                asset_id=k,
                position=pos.tolist(),
                displacement_from_source_initial=float(np.linalg.norm(pos - source_positions[k])),
                displacement_after_first_24_steps=float(np.linalg.norm(pos - initial_positions[k])),
                finite=bool(np.isfinite(pos).all()),
            )
        )
    world.stop()
    world.reset()
    world.stop()
    for _ in range(12):
        world.render()

    def capture(path=None, warmup=12):
        for _ in range(warmup):
            world.render()
        rgba = camera.get_rgba()
        if rgba is None or np.asarray(rgba).shape != (540, 960, 4):
            raise RuntimeError("No native Isaac camera frame")
        frame = np.asarray(rgba)[..., :3].astype(np.uint8)
        if frame.std() < 2:
            raise RuntimeError("Blank native render")
        if path:
            imageio.imwrite(path, frame)
        return frame

    images = []
    asset_bounds = {}
    asset_inspections = []
    original = {asset["id"]: (asset["position"], asset["rotation_deg"]) for asset in spec["assets"]}
    states = spec["diagnostic_states"]

    def apply_state(state):
        poses = {k: dict(position=pos, rotation_deg=rot) for k, (pos, rot) in original.items()}
        explicit = {p["asset_id"] for p in state.get("poses", [])}
        poses.update({p["asset_id"]: p for p in state.get("poses", [])})
        targets = {j["id"]: j["initial_position"] for j in spec.get("joints", [])}
        targets.update({q["joint_id"]: q["value"] for q in state.get("joint_positions", [])})
        # Use the same complete joint frames as the independent physics probe.
        # Rotating around an asset origin alone loses offset hinge anchors.
        for j in spec.get("joints", []):
            if j["child"] in explicit:
                continue
            joint = UsdPhysics.Joint(stage.GetPrimAtPath("/World/Joints/" + j["id"]))

            def local_rotation(q):
                return Rotation.from_quat([*q.GetImaginary(), q.GetReal()])

            parent = poses[j["parent"]]
            rp = Rotation.from_euler("xyz", parent["rotation_deg"], degrees=True)
            r0 = rp * local_rotation(joint.GetLocalRot0Attr().Get())
            axis = np.eye(3)["XYZ".index(j["axis"])]
            rr = (
                Rotation.from_rotvec(axis * np.deg2rad(targets[j["id"]]))
                if j["type"] == "revolute"
                else Rotation.identity()
            )
            rc = r0 * rr * local_rotation(joint.GetLocalRot1Attr().Get()).inv()
            p0 = np.array(parent["position"]) + rp.apply(j["parent_anchor"])
            p1 = (
                p0
                + (r0.apply(axis) * targets[j["id"]] if j["type"] == "prismatic" else 0)
                - rc.apply(j["child_anchor"])
            )
            poses[j["child"]] = dict(
                position=p1.tolist(), rotation_deg=rc.as_euler("xyz", degrees=True).tolist()
            )
        for k, pose in poses.items():
            transform(stage.GetPrimAtPath(assets[k]), pose["position"], pose["rotation_deg"])
        set_enabled_attachments(stage, spec, state)

    for si, state in enumerate(states):
        apply_state(state)
        for vi, az in enumerate([None, -2.1, -0.9]):
            if az is None:
                set_camera(
                    stage, camera_path, cam["position"], cam["target"], cam["vertical_fov_deg"]
                )
            else:
                orb = spec["orbit"]
                target = np.array(orb["target"])
                eye = target + np.array(
                    [orb["radius"] * math.cos(az), orb["radius"] * math.sin(az), orb["height"]]
                )
                set_camera(stage, camera_path, eye, target, 48)
            frame = capture(out / f"state-{si:02d}-{state['kind']}-view{vi}.png")
            label = f"S{si:02d} {state['kind']} | source frames " + ",".join(
                map(str, state.get("evidence_frames", []))
            )
            images.append((label, vi, frame))
    for first in range(0, len(images), 6):
        review_sheet(
            images[first : first + 6],
            out / f"review-{first // 6:02d}.jpg",
            lambda item: (
                f"{spec['scene_id']} | {item[0]} | view {item[1]} | posed diagnostic, no policy"
            ),
        )
    # Canonical asset views distinguish true geometry from source camera foreshortening.
    # Other assets are hidden temporarily; all original poses/visibility are restored.
    asset_images = []
    saved_visibility = {
        k: UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Get()
        for k, path in assets.items()
    }
    ground_visibility = (
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Ground")).GetVisibilityAttr().Get()
    )
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Ground")).MakeInvisible()
    for path in assets.values():
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    for asset in spec["assets"]:
        asset_id = asset["id"]
        path = assets[asset_id]
        prim = stage.GetPrimAtPath(path)
        transform(prim, [0, 0, 0], [0, 0, 0])
        UsdGeom.Imageable(prim).MakeVisible()
        bbox = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(prim)
            .ComputeAlignedRange()
        )
        lo = np.array(bbox.GetMin())
        hi = np.array(bbox.GetMax())
        extent = hi - lo
        center = (hi + lo) / 2
        asset_bounds[asset_id] = dict(
            local_min_m=lo.tolist(),
            local_max_m=hi.tolist(),
            extent_m=extent.tolist(),
            asset_root_scale=[1.0, 1.0, 1.0],
            measurement="Native BBoxCache with root translation/rotation normalized; descendant transforms applied once",
        )
        is_environment = any(
            word in asset_id
            for word in [
                "wall",
                "floor",
                "ground",
                "tabletop",
                "slab",
                "backdrop",
                "background",
                "cloth",
                "surround",
            ]
        )
        inspect = asset["body_type"] != "static" or (max(extent) < 0.65 and not is_environment)
        if inspect:
            radius = max(0.035, float(max(extent)) * 1.8)
            for view, offset in [
                ("front", [0, -radius, 0]),
                ("side", [radius, 0, 0]),
                ("top", [0, 0, radius]),
            ]:
                visible_width = extent[1] if view == "side" else extent[0]
                visible_height = extent[1] if view == "top" else extent[2]
                vertical_extent = (
                    max(0.02, float(visible_height), float(visible_width) * 540 / 960) * 1.25
                )
                set_asset_camera(
                    stage,
                    camera_path,
                    center + np.array(offset),
                    center,
                    vertical_extent,
                    [0, 1, 0] if view == "top" else None,
                )
                frame = capture(out / f"asset-{asset_id}-{view}.png")
                asset_images.append((asset_id, view + " orthographic", frame))
            edges = collider_wire(stage, path, max(0.00003, float(max(extent)) * 0.002))
            frame = capture(out / f"asset-{asset_id}-colliders.png")
            asset_images.append((asset_id, "authored convex-collider wire", frame))
            stage.RemovePrim("/World/InspectionColliderWire")
            asset_inspections.append(
                dict(
                    asset_id=asset_id,
                    views=[
                        "orthographic_front",
                        "orthographic_side",
                        "orthographic_top",
                        "authored_collider_wire",
                    ],
                    collider_wire_segments=edges,
                    scope="Isolated canonical geometry inspection with ground and other assets hidden; hull wire is authored representation, native cooking may simplify.",
                )
            )
        transform(prim, *original[asset_id])
        UsdGeom.Imageable(prim).MakeInvisible()
    for k, path in assets.items():
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Set(saved_visibility[k])
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Ground")).GetVisibilityAttr().Set(
        ground_visibility
    )
    for first in range(0, len(asset_images), 6):
        review_sheet(
            asset_images[first : first + 6],
            out / f"review-asset-{first // 6:02d}.jpg",
            lambda item: (
                f"ISOLATED {item[0]} | {item[1]} | bounds "
                + "x".join(f"{v:.3f}" for v in asset_bounds[item[0]]["extent_m"])
                + " m"
            ),
        )
    # Restore and export initial stage, never accidentally deliver the last posed inspection.
    for asset_id, (pos, rot) in original.items():
        transform(stage.GetPrimAtPath(assets[asset_id]), pos, rot)
    set_enabled_attachments(stage, spec, states[0])
    set_camera(stage, camera_path, cam["position"], cam["target"], cam["vertical_fov_deg"])
    stage.GetRootLayer().Export(str(out / "scene.usda"))
    if a.record:
        orb = spec["orbit"]
        target = np.array(orb["target"])
        video = out / "isaac-scene-orbit.mp4"
        with imageio.get_writer(
            video, fps=24, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
        ) as writer:
            for si, state in enumerate(states):
                apply_state(state)
                for _ in range(24):
                    world.render()
                for i, az in enumerate(np.linspace(-2.35, -0.65, 96)):
                    if i < 24:
                        set_camera(
                            stage,
                            camera_path,
                            cam["position"],
                            cam["target"],
                            cam["vertical_fov_deg"],
                        )
                    else:
                        eye = target + np.array(
                            [
                                orb["radius"] * math.cos(az),
                                orb["radius"] * math.sin(az),
                                orb["height"],
                            ]
                        )
                        set_camera(stage, camera_path, eye, target, 48)
                    frame = Image.fromarray(capture(warmup=3))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 34), fill="#101820")
                    draw.text(
                        (8, 4),
                        f"Native Isaac Sim | {spec['scene_id']} | state {si}: {state['kind']}",
                        fill="white",
                    )
                    draw.text(
                        (8, 18),
                        "Scene inspection | source-conditioned posed states | no robot policy or exact action replay",
                        fill="#cbd5e1",
                    )
                    writer.append_data(np.asarray(frame))
                    if i % 48 == 0:
                        print("ISAAC_RECORD_FRAME", si * 96 + i, flush=True)
        set_camera(stage, camera_path, cam["position"], cam["target"], cam["vertical_fov_deg"])
    report = dict(
        compiler_version=5,
        attachment_bindings=attachment_bindings,
        extra_source_files={
            "scene_attachments.py": hashlib.sha256(
                (out / "scene_attachments.py").read_bytes()
            ).hexdigest()
        },
        scene_id=spec["scene_id"],
        isaacsim=version("isaacsim"),
        hostname=os.uname().nodename,
        driver_icd=os.environ.get("VK_DRIVER_FILES"),
        native_runtime=True,
        requested_renderer=renderer,
        actual_render_mode=carb.settings.get_settings().get("/rtx/rendermode"),
        assets=len(assets),
        geometry_parts=parts_count,
        rigid_bodies=len(trackers),
        joints=len(spec.get("joints", [])),
        initial_joint_anchor_checks=joint_anchor_checks,
        physics_steps=144,
        physics_dt=1 / 120,
        rest_diagnostics=rest,
        asset_local_bounds=asset_bounds,
        asset_inspections=asset_inspections,
        render_images=len(images),
        asset_render_images=len(asset_images),
        recorded=a.record,
        video_frames=len(states) * 96 if a.record else 0,
        video_fps=24 if a.record else None,
        inspection_states=[s["kind"] for s in states],
        scope="Source-conditioned geometry and native Isaac Sim scene inspection. Extra source states are explicitly posed diagnostics, not reconstructed exact motion or learned task execution.",
        elapsed_s=time.time() - started,
        source_spec=str(spec_path),
    )
    report.update(
        source_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        frozen_worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        camera_resolution=[960, 540],
        review_panel_resolution=[960, 540],
        review_panel_resize="none",
        orthographic_aperture_units_reference="https://openusd.org/dev/api/class_usd_geom_camera.html",
        viewport_grid_enabled=carb.settings.get_settings().get("/app/viewport/grid/enabled"),
        authored_joint_drives=authored_joint_drives,
    )
    (out / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ISAAC_NATIVE_SCENE_DONE", spec["scene_id"], flush=True)
except BaseException as error:
    failure = dict(
        scene_id=spec.get("scene_id"),
        error=repr(error),
        traceback=traceback.format_exc(),
        time=time.time(),
    )
    (out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
    print("ISAAC_NATIVE_SCENE_FAILED", failure["traceback"], flush=True)
    raise
finally:
    app.close()
