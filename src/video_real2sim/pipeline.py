"""Resumable stages with content-bound, consecutive multi-round review gates."""

from __future__ import annotations

import shutil
from pathlib import Path

from .artifacts import (
    canonical,
    digest,
    exclusive_lock,
    file_record,
    read_json,
    resolve_member,
    utc_now,
    verify_files,
    write_json,
)
from .backend import IsaacBackend, native_sources
from .config import ProjectConfig
from .errors import IntegrityError, NeedsAttention
from .models import Proposal, Review, Understanding
from .provider import OpenAIProvider
from .video import inspect_video, preprocess, refine_frames

PROMPTS = Path(__file__).parent / "prompts"


class Pipeline:
    def __init__(self, config: ProjectConfig, provider=None, backend=None):
        self.config = config
        self.provider = provider or OpenAIProvider(config.api)
        self.backend = backend or IsaacBackend(config.runtime, config.api.key_env)

    def _status(self, run: Path, status: str, **details) -> dict:
        record = {"status": status, "updated_at": utc_now(), **details}
        write_json(run / "status.json", record)
        return record

    def prepare(self, video: Path, run: Path, scene_id: str) -> dict:
        import re

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,95}", scene_id):
            raise ValueError("scene_id must be a safe identifier")
        run.mkdir(parents=True, exist_ok=True)
        source = {"scene_id": scene_id, "source_sha256": digest(video), "original_name": video.name}
        config = self.config.model_dump()
        if (run / "source.json").exists():
            saved = read_json(run / "source.json")
            if saved != source or digest(run / "source.mp4") != source["source_sha256"]:
                raise IntegrityError("Source changed; use a new run directory")
            if read_json(run / "config.json") != config:
                raise IntegrityError("Configuration changed; use a new run directory")
        else:
            if (run / "source.mp4").exists():
                raise IntegrityError("Unregistered video already exists in run directory")
            shutil.copyfile(video, run / "source.mp4")
            write_json(run / "source.json", source)
            write_json(run / "config.json", config)
        return preprocess(run / "source.mp4", run / "source", self.config.sampling)

    def _understand(
        self, run: Path, timeline: dict, retry_uncertain: bool
    ) -> tuple[Understanding, list[Path]]:
        source = read_json(run / "source.json")
        images = [run / "source" / im["path"] for im in timeline["contact_sheets"]]
        observed_frames = {row["frame"] for row in timeline["selected_frames"]}
        context = "\nSOURCE:\n" + canonical(source) + "\nTIMELINE:\n" + canonical(timeline)
        prompt = (PROMPTS / "analyze.txt").read_text() + context
        understanding = self.provider.call(
            "understanding",
            prompt,
            images,
            Understanding,
            run / "calls/understanding-00",
            retry_uncertain=retry_uncertain,
        )
        understanding.validate_video(
            source["scene_id"], timeline["decoded_frames"], observed_frames
        )
        write_json(run / "understanding-00.json", understanding.model_dump())
        for round_index in range(1, self.config.sampling.refinement_rounds + 1):
            if not understanding.missing_evidence:
                break
            directory = run / f"source/refinement-{round_index:02d}"
            records = refine_frames(
                run / "source.mp4",
                understanding.missing_evidence,
                directory,
                timeline["decoded_frames"],
                self.config.sampling,
            )
            images += [directory / rec["path"] for rec in records]
            observed_frames.update(rec["frame"] for rec in records)
            prompt = (PROMPTS / "analyze.txt").read_text() + context
            prompt += "\nREFINEMENT: Keep established objects/actions; resolve uncertainties only from evidence. "
            prompt += "Attached original-resolution frames follow the timeline and earlier refinement images. "
            prompt += "Do not mark unobservable facts as known.\nPREVIOUS:\n" + canonical(
                understanding.model_dump()
            )
            prompt += "\nNEW FRAME INDEX:\n" + canonical(records)
            understanding = self.provider.call(
                f"understanding-refinement-{round_index}",
                prompt,
                images,
                Understanding,
                run / f"calls/understanding-{round_index:02d}",
                retry_uncertain=retry_uncertain,
            )
            understanding.validate_video(
                source["scene_id"], timeline["decoded_frames"], observed_frames
            )
            write_json(run / f"understanding-{round_index:02d}.json", understanding.model_dump())
        write_json(run / "understanding.json", understanding.model_dump())
        write_json(
            run / "source/evidence.json",
            {
                "observed_frames": sorted(observed_frames),
                "images": [file_record(p, run / "source") for p in images],
            },
        )
        if understanding.missing_evidence:
            raise NeedsAttention(
                "Temporal evidence remains unresolved after the configured refinement rounds"
            )
        return understanding, images

    def run(self, video: Path, run: Path, scene_id: str, *, retry_uncertain: bool = False) -> dict:
        run = run.resolve()
        with exclusive_lock(run / ".pipeline.lock"):
            try:
                timeline = self.prepare(video, run, scene_id)
                candidate_path = run / "candidate.json"
                if candidate_path.exists():
                    candidate = read_json(candidate_path)
                    verify_files(run, candidate["files"])
                    if (run / "delivery.json").exists():
                        delivery = read_json(run / "delivery.json")
                        if delivery["candidate_sha256"] != digest(candidate_path):
                            raise IntegrityError("Delivery belongs to a different candidate")
                        verify_files(run, delivery["files"])
                        return self._status(run, "complete", revision=candidate["revision"])
                    return self._status(
                        run,
                        "awaiting_approval"
                        if self.config.review.require_human_approval
                        else "ready_to_record",
                        revision=candidate["revision"],
                    )
                understanding, source_images = self._understand(run, timeline, retry_uncertain)
                observed_frames = set(read_json(run / "source/evidence.json")["observed_frames"])
                feedback = None
                previous = None
                for revision in range(1, self.config.review.max_revisions + 1):
                    directory = run / f"revisions/{revision:03d}"
                    directory.mkdir(parents=True, exist_ok=True)
                    self._status(run, "building", revision=revision)
                    prompt = (PROMPTS / "geometry.txt").read_text()
                    prompt += "\nReturn a Proposal object with scene and validation fields, using the provided schema. "
                    prompt += "The initial diagnostic state must have empty poses/joint_positions and use asset initial transforms. "
                    prompt += "Every asset source_object_id must match an understanding object. Explicitly set drive=null for passive joints. "
                    prompt += "Validation: classify EVERY state into rest_checks or excluded_states with a physical reason "
                    prompt += "(for example human-held). Every moving joint requires a driven continuous sweep with its "
                    prompt += "start state and end_position. Report extra contact/assembly/deformation checks in additional_requirements; "
                    prompt += "do not hide requirements merely because the generic runner cannot satisfy them."
                    prompt += " For detachable fixed-joint assemblies with one common parent, include assembly_checks "
                    prompt += "for both hold and level_then_release, each starting from a state with all connections enabled. "
                    prompt += "These checks prescribe only the externally held parent and verify dynamic child retention/release."
                    prompt += "\nUNDERSTANDING:\n" + canonical(understanding.model_dump())
                    if previous is not None:
                        prompt += "\nPREVIOUS PROPOSAL:\n" + canonical(previous.model_dump())
                        prompt += "\nREPAIR EVIDENCE:\n" + canonical(feedback)
                    proposal = self.provider.call(
                        f"geometry-{revision}",
                        prompt,
                        source_images,
                        Proposal,
                        directory / "geometry-call",
                        retry_uncertain=retry_uncertain,
                    )
                    proposal.scene.validate_evidence(
                        understanding, timeline["decoded_frames"], observed_frames
                    )
                    write_json(directory / "proposal.json", proposal.model_dump())
                    spec = directory / "scene.json"
                    write_json(spec, proposal.scene.compiler_dict())
                    rendered = directory / "render"
                    runtime = self.backend.render(spec, rendered)
                    physics = self.backend.validate(proposal, spec, rendered, directory / "physics")
                    native_images = sorted(rendered.glob("review-*.jpg"))
                    native_images += [directory / "physics" / r["path"] for r in physics["images"]]
                    reports = []
                    for pass_index in range(1, self.config.review.passes_required + 1):
                        self._status(run, "reviewing", revision=revision, review_pass=pass_index)
                        prompt = (PROMPTS / "review.txt").read_text()
                        prompt += f"\nCURRENT_REVIEW_ROUND: {pass_index}. Inspect independently; do not infer approval from earlier rounds."
                        prompt += "\nSOURCE IMAGE ORDER:\n" + canonical(
                            [p.name for p in source_images]
                        )
                        prompt += "\nNATIVE IMAGE ORDER:\n" + canonical(
                            [str(p.relative_to(directory)) for p in native_images]
                        )
                        prompt += "\nUNDERSTANDING:\n" + canonical(understanding.model_dump())
                        prompt += "\nSCENE AND PLAN:\n" + canonical(proposal.model_dump())
                        prompt += "\nNATIVE RUNTIME:\n" + canonical(runtime)
                        prompt += "\nPHYSICS:\n" + canonical(physics)
                        review = self.provider.call(
                            f"review-{revision}-{pass_index}",
                            prompt,
                            source_images + native_images,
                            Review,
                            directory / f"review-{pass_index:02d}",
                            review=True,
                            retry_uncertain=retry_uncertain,
                        )
                        if review.scene_id != scene_id or review.round != pass_index:
                            raise IntegrityError("Reviewer returned the wrong scene or round")
                        review.validate_evidence(observed_frames)
                        reports.append(review.model_dump())
                        if not review.accepted() or not physics["passed"]:
                            break
                    if (
                        len(reports) == self.config.review.passes_required
                        and all(Review.model_validate(r).accepted() for r in reports)
                        and physics["passed"]
                    ):
                        selected = [
                            run / "source.json",
                            run / "source.mp4",
                            run / "config.json",
                            run / "understanding.json",
                        ]
                        selected += [
                            p
                            for folder in (run / "source", directory)
                            for p in folder.rglob("*")
                            if p.is_file()
                            and not p.name.startswith(".")
                            and "__pycache__" not in p.parts
                        ]
                        candidate = {
                            "revision": revision,
                            "scene_id": scene_id,
                            "spec": str(spec.relative_to(run)),
                            "spec_sha256": digest(spec),
                            "review_passes": len(reports),
                            "created_at": utc_now(),
                            "files": [file_record(p, run) for p in sorted(set(selected))],
                            "scope": "Scene reconstruction and inspection only; no policy execution.",
                        }
                        write_json(candidate_path, candidate)
                        return self._status(
                            run,
                            "awaiting_approval"
                            if self.config.review.require_human_approval
                            else "ready_to_record",
                            revision=revision,
                        )
                    feedback = {"reviews": reports, "physics": physics}
                    previous = proposal
                raise NeedsAttention(
                    "Maximum geometry revisions reached without consecutive review passes"
                )
            except Exception as exc:
                self._status(run, "needs_attention", error_type=type(exc).__name__, detail=str(exc))
                raise

    def approve(self, run: Path, reviewer: str, observations: str) -> dict:
        run = run.resolve()
        if not reviewer.strip() or len(observations.strip()) < 20:
            raise ValueError(
                "Approval needs reviewer identity and concrete visual observations (20+ characters)"
            )
        with exclusive_lock(run / ".pipeline.lock"):
            candidate = read_json(run / "candidate.json")
            verify_files(run, candidate["files"])
            approval = {
                "candidate_sha256": digest(run / "candidate.json"),
                "reviewer": reviewer,
                "observations": observations,
                "accepted_at": utc_now(),
                "scope": "Reviewer attests to personally inspecting source, native states, assets and physics evidence.",
            }
            write_json(run / "approval.json", approval)
            return self._status(run, "ready_to_record", revision=candidate["revision"])

    def record(self, run: Path, *, retry_uncertain: bool = False) -> dict:
        run = run.resolve()
        with exclusive_lock(run / ".pipeline.lock"):
            candidate = read_json(run / "candidate.json")
            verify_files(run, candidate["files"])
            if read_json(run / "config.json") != self.config.model_dump():
                raise IntegrityError("Recording configuration differs from the approved run")
            if self.config.review.require_human_approval:
                approval = read_json(run / "approval.json")
                if approval["candidate_sha256"] != digest(run / "candidate.json"):
                    raise IntegrityError("Approval belongs to a different candidate")
            spec = run / candidate["spec"]
            for name, sha in native_sources().items():
                approved_worker = spec.parent / "render" / name
                if not approved_worker.is_file() or digest(approved_worker) != sha:
                    raise IntegrityError(
                        "Native implementation changed since approval; review a new run"
                    )
            proposal = Proposal.model_validate(read_json(spec.parent / "proposal.json"))
            output = run / "recording"
            self.backend.render(spec, output, record=True)
            inspection = inspect_video(
                output / "isaac-scene-orbit.mp4",
                output / "inspection",
                len(proposal.scene.diagnostic_states),
            )
            evidence = read_json(run / "source/evidence.json")
            verify_files(run / "source", evidence["images"])
            source_images = [resolve_member(run / "source", p["path"]) for p in evidence["images"]]
            video_images = [output / "inspection" / p["path"] for p in inspection["sheets"]]
            for index in range(1, self.config.review.passes_required + 1):
                prompt = (PROMPTS / "review.txt").read_text()
                prompt += "\nPOST-RECORDING OVERRIDE: Inspect actual decoded video contact sheets after source sheets. "
                prompt += "Check final video for missing/wrong states, blank frames and correspondence to the approved scene. "
                prompt += "This is a posed-state scene-inspection video, not a policy rollout. "
                prompt += f"Set round={index}.\nSCENE:\n" + canonical(proposal.model_dump())
                prompt += "\nUNDERSTANDING:\n" + canonical(read_json(run / "understanding.json"))
                prompt += "\nVIDEO INTEGRITY:\n" + canonical(inspection)
                review = self.provider.call(
                    f"recording-review-{index}",
                    prompt,
                    source_images + video_images,
                    Review,
                    output / f"review-{index:02d}",
                    review=True,
                    retry_uncertain=retry_uncertain,
                )
                review.validate_evidence(set(evidence["observed_frames"]))
                if (
                    review.scene_id != candidate["scene_id"]
                    or review.round != index
                    or not review.accepted()
                ):
                    return self._status(run, "recording_needs_attention", review_pass=index)
            write_json(
                run / "delivery.json",
                {
                    "scene_id": candidate["scene_id"],
                    "candidate_sha256": digest(run / "candidate.json"),
                    "video": file_record(output / "isaac-scene-orbit.mp4", run),
                    "scene": file_record(output / "scene.usda", run),
                    "recording_reviews": self.config.review.passes_required,
                    "files": [
                        file_record(p, run)
                        for p in sorted(
                            list(output.rglob("*"))
                            + (
                                [run / "approval.json"]
                                if self.config.review.require_human_approval
                                else []
                            )
                        )
                        if p.is_file()
                        and not p.name.startswith(".")
                        and "__pycache__" not in p.parts
                    ],
                    "scope": "Source-conditioned scene and independent assets; scripted state inspection, no robot policy.",
                },
            )
            return self._status(run, "complete", revision=candidate["revision"])
