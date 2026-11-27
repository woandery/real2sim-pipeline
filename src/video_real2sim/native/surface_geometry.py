"""Read visible compiled USD triangles independently of collision proxies."""

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom


def visible_triangles(stage, asset_path):
    root = stage.GetPrimAtPath(asset_path)
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    triangles = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        points = []
        faces = []
        if prim.IsA(UsdGeom.Mesh):
            g = UsdGeom.Mesh(prim)
            points = np.asarray(g.GetPointsAttr().Get(), float)
            indices = list(g.GetFaceVertexIndicesAttr().Get())
            offset = 0
            for count in g.GetFaceVertexCountsAttr().Get():
                face = indices[offset : offset + count]
                offset += count
                faces.extend([[face[0], face[i], face[i + 1]] for i in range(1, count - 1)])
        elif prim.IsA(UsdGeom.Cylinder):
            g = UsdGeom.Cylinder(prim)
            assert g.GetAxisAttr().Get() == "Z"
            rad = float(g.GetRadiusAttr().Get())
            half = float(g.GetHeightAttr().Get()) / 2
            points = [
                [rad * math.cos(a), rad * math.sin(a), z]
                for z in [-half, half]
                for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)
            ]
            points.extend([[0, 0, -half], [0, 0, half]])
            for i in range(48):
                j = (i + 1) % 48
                faces.extend(
                    [[96, j, i], [97, 48 + i, 48 + j], [i, j, 48 + j], [i, 48 + j, 48 + i]]
                )
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            points = [
                [x, y, z] for x in [-half, half] for y in [-half, half] for z in [-half, half]
            ]
            for face in [
                [0, 1, 3, 2],
                [4, 6, 7, 5],
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 2, 6, 4],
                [1, 5, 7, 3],
            ]:
                faces.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
        else:
            raise ValueError(
                f"Unsupported visible surface geometry: {prim.GetPath()} {prim.GetTypeName()}"
            )
        matrix = cache.GetLocalToWorldTransform(prim)
        points = np.array(
            [
                inverse.TransformAffine(matrix.TransformAffine(Gf.Vec3d(*map(float, p))))
                for p in points
            ],
            float,
        )
        triangles.extend(points[np.asarray(faces, int)])
    assert triangles, "No visible triangles"
    return np.asarray(triangles)


def top_height(triangles, x, y):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    valid = np.abs(den) > 1e-14
    safe = np.where(valid, den, 1.0)
    u = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / safe
    v = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / safe
    w = 1 - u - v
    valid &= (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    if not valid.any():
        return None
    return float(np.max((u * a[:, 2] + v * b[:, 2] + w * c[:, 2])[valid]))


def top_surface(triangles, x, y):
    """Independent visible triangle hit and its outward/upward normal."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    valid = np.abs(den) > 1e-14
    safe = np.where(valid, den, 1.0)
    u = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / safe
    v = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / safe
    w = 1 - u - v
    valid &= (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return None
    heights = u * a[:, 2] + v * b[:, 2] + w * c[:, 2]
    index = int(ids[np.argmax(heights[ids])])
    normal = np.cross(b[index] - a[index], c[index] - a[index])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    return float(heights[index]), normal, index


def slope_sites(radii, triangles):
    if len(radii) != 2 or not 0 < radii[0] < radii[1]:
        raise ValueError("Expected two increasing positive slope radii")
    result = []
    for slope, radius in enumerate(radii, 1):
        for angle in [15.0, 135.0, 255.0]:
            phi = math.radians(angle)
            xy = [radius * math.cos(phi), radius * math.sin(phi)]
            z, normal, index = top_surface(triangles, *xy)
            assert 0.1 < normal[2] < 0.999, "Site must lie on a visible slope"
            result.append(
                dict(
                    name=f"slope{slope}-az{int(angle)}",
                    xy=xy,
                    visible_z=z,
                    slope_index=slope,
                    azimuth_deg=angle,
                    radius_m=radius,
                    visible_normal_local=normal.tolist(),
                    triangle_index=index,
                )
            )
    return result


def surface_sites(triangles):
    central = top_height(triangles, 0.0, 0.0)
    assert central is not None
    radius = float(np.linalg.norm(triangles[..., :2], axis=2).max())
    samples = []
    for x in np.linspace(radius * 0.5, radius * 0.99, 300):
        z = top_height(triangles, float(x), 0.0)
        if z is not None and central - 0.006 < z < central - 0.0002:
            samples.append((float(x), z))
    groups = []
    for value in samples:
        if (
            not groups
            or abs(value[1] - groups[-1][-1][1]) > 1e-5
            or value[0] - groups[-1][-1][0] > radius * 0.01
        ):
            groups.append([])
        groups[-1].append(value)
    groups = [g for g in groups if g[-1][0] - g[0][0] > 0.0022]
    if groups:
        chosen = max(groups, key=lambda g: g[-1][0] - g[0][0])
        x = (chosen[0][0] + chosen[-1][0]) / 2
    else:
        x = radius * 0.85
    shoulder = top_height(triangles, x, 0.0)
    assert shoulder is not None
    return [
        dict(name="central", xy=[0.0, 0.0], visible_z=central),
        dict(name="shoulder", xy=[x, 0.0], visible_z=shoulder),
    ]
