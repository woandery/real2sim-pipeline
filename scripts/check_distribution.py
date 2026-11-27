"""Verify shareable archive contents without exposing excluded local files."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def audit(folder: Path) -> dict:
    wheel, source = next(folder.glob("*.whl")), next(folder.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    with tarfile.open(source) as archive:
        source_files = {}
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts or member.issym() or member.islnk():
                raise ValueError("Unsafe source archive member")
            if member.isfile():
                source_files["/".join(parts[1:])] = archive.extractfile(member).read()
    required_source = {
        "README.md",
        "config.example.toml",
        ".env.example",
        "docs/mentor-guide.zh.md",
        "docs/validation.zh.md",
        "docs/operator-guide.zh.md",
        "tests/test_native_jobs.py",
        "examples/historical/manifest.json",
        "scripts/audit_native_results.py",
    }
    required_wheel = {
        f"video_real2sim/{n}"
        for n in [
            "provider.py",
            "pipeline.py",
            "specialized.py",
            "native/isaac_worker.py",
            "native/physics_probe.py",
            "native/surface_worker.py",
            "native/surface_query_worker.py",
            "native/interface_worker.py",
            "prompts/analyze.txt",
            "prompts/review.txt",
        ]
    }
    assert required_source <= source_files.keys(), required_source - source_files.keys()
    assert required_wheel <= wheel_files.keys(), required_wheel - wheel_files.keys()
    for name, data in {**source_files, **wheel_files}.items():
        path = PurePosixPath(name)
        assert not ({".venv", "runs", "__pycache__", ".git"} & set(path.parts)), name
        assert path.name not in {".env", "config.toml", "config.local.toml", "WORKSTATE.json"}, name
        assert path.suffix not in {".pem", ".key"}, name
        assert path.suffix != ".mp4" or name == "checks/native-box-orbit.mp4", name
        if path.suffix in {".py", ".md", ".json", ".toml", ".txt", ".xml"}:
            # Private installation paths must not leak into a portable research package.
            assert not any(b"/" + root.encode() + b"/" in data for root in ("home", "inspire")), (
                name
            )
    records = [
        dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
        for p in (wheel, source)
    ]
    return dict(
        passed=True,
        wheel_files=len(wheel_files),
        source_files=len(source_files),
        artifacts=records,
        scope="Required code/docs/examples present; credentials, private paths, runs and dataset videos excluded.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    result = audit(args.dist)
    (args.dist / "artifacts.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
