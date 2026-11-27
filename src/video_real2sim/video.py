"""Whole-video coverage, bounded temporal refinement and decoded-output inspection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .artifacts import digest, read_json, verify_files, write_json
from .config import SamplingSettings
from .errors import IntegrityError, NeedsAttention, WorkflowError
from .models import EvidenceRequest


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) <= 0:
        cap.release()
        raise WorkflowError(f"Unreadable video: {path.name}")
    return cap


def extract_frames(video: Path, indices: list[int], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    records = []
    try:
        for index in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise WorkflowError(f"Cannot retrieve source frame {index}")
            path = output / f"frame-{index:07d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise WorkflowError("Cannot save video evidence")
            records.append(
                {"frame": index, "time_s": index / fps, "path": path.name, "sha256": digest(path)}
            )
    finally:
        cap.release()
    return records


def contact_sheets(records: list[dict], frames: Path, output: Path) -> list[dict]:
    sheets = []
    for first in range(0, len(records), 12):
        sheet = Image.new("RGB", (1920, 1200), "#101820")
        draw = ImageDraw.Draw(sheet)
        for i, rec in enumerate(records[first : first + 12]):
            with Image.open(frames / rec["path"]) as original:
                image = original.convert("RGB")
            image.thumbnail((640, 270))
            x, y = i % 3 * 640, i // 3 * 300
            draw.text((x + 8, y + 5), f"Frame {rec['frame']} | {rec['time_s']:.3f}s", fill="white")
            sheet.paste(image, (x + (640 - image.width) // 2, y + 25))
        path = output / f"timeline-{first // 12:02d}.jpg"
        sheet.save(path, quality=94)
        sheets.append({"path": path.name, "sha256": digest(path)})
    return sheets


def preprocess(video: Path, output: Path, settings: SamplingSettings) -> dict:
    source_hash = digest(video)
    manifest = output / "timeline.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["source_sha256"] != source_hash or saved["sampling"] != settings.model_dump():
            raise IntegrityError("Video or sampling settings changed; use a new run")
        verify_files(output, saved["contact_sheets"])
        verify_files(output / "frames", saved["selected_frames"])
        return saved
    output.mkdir(parents=True, exist_ok=True)
    cap = open_video(video)
    fps, declared = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    changes, previous = [], None
    width = height = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(
                np.float32
            )
            changes.append(0.0 if previous is None else float(np.mean(np.abs(small - previous))))
            previous = small
    finally:
        cap.release()
    count = len(changes)
    if count < 2 or (declared > 0 and declared != count):
        raise WorkflowError(f"Incomplete or too-short source video: {count}/{declared} frames")
    uniform = set(
        np.linspace(0, count - 1, min(count, settings.uniform_frames)).round().astype(int).tolist()
    )
    chosen = set(uniform)
    separation = max(1, int(fps * settings.event_separation_seconds))
    for index in np.argsort(-np.asarray(changes), kind="stable"):
        if len(chosen) >= len(uniform) + settings.event_frames:
            break
        if all(abs(int(index) - other) >= separation for other in chosen):
            chosen.add(int(index))
    records = extract_frames(video, sorted(chosen), output / "frames")
    for record in records:
        index = record["frame"]
        record.update(
            reason="uniform" if index in uniform else "motion_peak", change_score=changes[index]
        )
    sheets = contact_sheets(records, output / "frames", output)
    result = {
        "source_sha256": source_hash,
        "fps": fps,
        "width": width,
        "height": height,
        "declared_frames": declared,
        "decoded_frames": count,
        "all_frames_decoded": True,
        "duration_s": count / fps,
        "sampling": settings.model_dump(),
        "selected_frames": records,
        "contact_sheets": sheets,
        "scope": "All frames decoded; semantic understanding uses sampled evidence. Brief or occluded events may remain unresolved.",
    }
    write_json(manifest, result)
    return result


def refine_frames(
    video: Path,
    requests: list[EvidenceRequest],
    output: Path,
    frame_count: int,
    settings: SamplingSettings,
) -> list[dict]:
    chosen: set[int] = set()
    for request in requests:
        start, end = request.interval_frames
        if not 0 <= start <= end < frame_count:
            raise WorkflowError("Refinement interval lies outside video")
        chosen.update(
            np.linspace(start, end, settings.frames_per_interval).round().astype(int).tolist()
        )
    if len(chosen) > settings.max_refinement_frames:
        raise NeedsAttention(
            "Refinement frame budget exceeded; intervals were not silently truncated"
        )
    manifest = output / "manifest.json"
    expected = {
        "source_sha256": digest(video),
        "requests": [r.model_dump() for r in requests],
        "indices": sorted(chosen),
    }
    if manifest.exists():
        saved = read_json(manifest)
        if saved["inputs"] != expected:
            raise IntegrityError("Refinement inputs changed")
        verify_files(output, saved["frames"])
        return saved["frames"]
    records = extract_frames(video, sorted(chosen), output)
    write_json(manifest, {"inputs": expected, "frames": records})
    return records


def inspect_video(video: Path, output: Path, state_count: int) -> dict:
    """Decode every native output frame and retain three actual frames per state."""
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    indices = {i * 96 + k for i in range(state_count) for k in (0, 48, 95)}
    min_std = float("inf")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (540, 960):
                raise WorkflowError("Unexpected native video dimensions")
            # Spatial variation, not variation between RGB channels: a solid red
            # failure frame must not pass merely because red differs from green.
            min_std = min(min_std, float(frame[40:].std(axis=(0, 1)).mean()))
            count += 1
    finally:
        cap.release()
    if count != state_count * 96 or abs(fps - 24) > 0.01 or min_std < 2:
        raise WorkflowError("Native video failed complete decode/length/nonblank checks")
    records = extract_frames(video, sorted(indices), output / "frames")
    sheets = contact_sheets(records, output / "frames", output)
    result = {
        "video_sha256": digest(video),
        "frames": count,
        "fps": fps,
        "minimum_image_std": min_std,
        "samples": records,
        "sheets": sheets,
        "all_frames_decoded": True,
    }
    write_json(output / "video-integrity.json", result)
    return result
