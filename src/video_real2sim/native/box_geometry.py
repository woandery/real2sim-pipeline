"""Independent convex-box geometry measurements, never contact-force evidence."""

import itertools

import numpy as np
from scipy.optimize import lsq_linear

SIGNS = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
EDGES = [
    (i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1
]


def vertices(box):
    return SIGNS @ np.asarray(box["edges_world"]).T + np.asarray(box["center_world"])


def distance(first, second):
    """Exact convex distance from bounded least squares; SAT supplies overlap sign.

    edge matrices contain full-length edge vectors as columns. The optimization
    is a convex quadratic on two closed unit boxes, independent of PhysX.
    """
    a = np.asarray(first["edges_world"])
    b = np.asarray(second["edges_world"])
    delta = np.asarray(second["center_world"]) - np.asarray(first["center_world"])
    solution = lsq_linear(
        np.column_stack([a, -b]), delta, bounds=(-0.5, 0.5), method="bvls", tol=1e-12, max_iter=200
    )
    assert solution.success
    axes = [a[:, i] for i in range(3)] + [b[:, i] for i in range(3)]
    axes += [np.cross(a[:, i], b[:, j]) for i in range(3) for j in range(3)]
    margins = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-15:
            continue
        axis = axis / norm
        margins.append(
            float(abs(delta @ axis) - 0.5 * (np.abs(axis @ a).sum() + np.abs(axis @ b).sum()))
        )
    return dict(
        distance_m=float(np.linalg.norm(solution.fun)),
        sat_margin_m=max(margins),
        intersects=max(margins) < -1e-9,
    )


def plane_section(box, z):
    points = vertices(box)
    hits = []
    for i, j in EDGES:
        p, q = points[i], points[j]
        for endpoint in [p, q]:
            if abs(endpoint[2] - z) < 1e-10:
                hits.append(endpoint[:2])
        if (p[2] - z) * (q[2] - z) < 0:
            hits.append((p + (q - p) * (z - p[2]) / (q[2] - p[2]))[:2])
    if not hits:
        return []
    unique = np.unique(np.round(hits, 12), axis=0)
    if len(unique) < 3:
        return []
    center = unique.mean(axis=0)
    angles = np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
    return unique[np.argsort(angles)].tolist()
