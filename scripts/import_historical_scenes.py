"""Extract geometry fixtures from an explicitly supplied historical experiment.

Does not import approvals, videos, account configuration or claim a new native run.
Unknown metadata fields are recorded rather than silently accepted by new contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from video_real2sim.models import SceneSpec


def project_fields(value, schema, definitions, path, removed):
    if "$ref" in schema:
        schema = definitions[schema["$ref"].rsplit("/", 1)[-1]]
    if "anyOf" in schema:
        options = schema["anyOf"]
        kind = (
            "null"
            if value is None
            else "object"
            if isinstance(value, dict)
            else "array"
            if isinstance(value, list)
            else None
        )
        schema = next(
            (s for s in options if s.get("type") == kind or (kind == "object" and "$ref" in s)),
            options[0],
        )
        return project_fields(value, schema, definitions, path, removed)
    if isinstance(value, dict) and "properties" in schema:
        result = {}
        for key, item in value.items():
            if key not in schema["properties"]:
                removed.append(f"{path}.{key}")
            else:
                result[key] = project_fields(
                    item, schema["properties"][key], definitions, f"{path}.{key}", removed
                )
        return result
    if isinstance(value, list) and "items" in schema:
        return [
            project_fields(item, schema["items"], definitions, f"{path}[{i}]", removed)
            for i, item in enumerate(value)
        ]
    return value


def migrate(raw: dict) -> tuple[SceneSpec, list[str], list[str]]:
    schema = SceneSpec.model_json_schema()
    removed, changes = [], []
    data = project_fields(raw, schema, schema["$defs"], "$", removed)
    if data.get("compiler_version") != 5:
        changes.append("compiler_version explicitly set to 5; this requires new native validation")
        data["compiler_version"] = 5
    assets = {a["id"]: a for a in data["assets"]}
    joints = {j["id"]: j for j in data["joints"]}
    for joint in data["joints"]:
        if "drive" not in joint:
            joint["drive"] = (
                {"stiffness": 250.0, "damping": 40.0, "max_force": 1000.0}
                if joint["type"] != "fixed"
                else None
            )
            changes.append(f"{joint['id']}: made historical implicit drive explicit")
        joint.setdefault("collision_enabled", False)
    first = data["diagnostic_states"][0]
    for pose in first["poses"]:
        if any(pose[k] != assets[pose["asset_id"]][k] for k in ("position", "rotation_deg")):
            raise ValueError("Historical initial poses disagree with asset transforms")
    for joint in first["joint_positions"]:
        if joint["value"] != joints[joint["joint_id"]]["initial_position"]:
            raise ValueError("Historical initial joint state disagrees with joint declaration")
    if first["poses"] or first["joint_positions"]:
        changes.append(
            "Removed redundant initial-state pose/joint copies after exact equality check"
        )
    first["poses"], first["joint_positions"] = [], []
    return SceneSpec.model_validate(data), removed, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.experiment / "delivery/manifest.json").read_text())
    rows = manifest["scenes"] if isinstance(manifest, dict) else manifest
    records = []
    for row in rows:
        scene = row["scene_id"]
        path = args.experiment / "scenes" / scene / f"scene-v{row['accepted_revision']}.json"
        model, removed, changes = migrate(json.loads(path.read_text()))
        target = args.output / f"{scene}.json"
        target.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "scene_id": scene,
                "source": str(path.relative_to(args.experiment)),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fixture": target.name,
                "fixture_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "excluded_metadata_fields": removed,
                "explicit_migration_changes": changes,
            }
        )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "scope": "Geometry contract regression fixtures; no API/native validation or historical approval transferred.",
                "scenes": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Imported {len(records)} geometry fixtures")


if __name__ == "__main__":
    main()
