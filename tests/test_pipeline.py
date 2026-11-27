import pytest
from PIL import Image
from test_video import synthetic_video

from video_real2sim.artifacts import read_json, write_json
from video_real2sim.config import APISettings, ProjectConfig, ReviewSettings, SamplingSettings
from video_real2sim.errors import IntegrityError, NeedsAttention
from video_real2sim.models import Completeness, Proposal, Review, Understanding
from video_real2sim.pipeline import Pipeline


class ScriptedProvider:
    """Explicit offline test double: never produces an API or native provenance receipt."""

    def __init__(self, understanding, proposal, verdicts):
        self.understanding = understanding
        self.proposal = proposal
        self.verdicts = iter(verdicts)
        self.calls = []

    def call(self, stage, prompt, images, schema, directory, **kwargs):
        self.calls.append(stage)
        if schema is Understanding:
            return self.understanding
        if schema is Proposal:
            return self.proposal
        index = int(stage.rsplit("-", 1)[1])
        result = Review(
            scene_id="box_demo",
            round=index,
            verdict=next(self.verdicts),
            findings=[],
            completeness=Completeness(
                later_revealed_objects_accounted_for=True,
                complete_action_sequence=True,
                initial_and_final_state_consistent=True,
                non_target_obstacles_accounted_for=True,
            ),
            remaining_uncertainties=["Test fixture"],
            acceptance_reasoning="Offline test only",
        )
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "result.json", result.model_dump())
        return result


class FakeBackend:
    def __init__(self, passed=True):
        self.passed = passed

    def render(self, spec, output, *, record=False):
        import shutil

        from video_real2sim.backend import NATIVE

        output.mkdir(parents=True, exist_ok=True)
        for path in NATIVE.glob("*.py"):
            shutil.copyfile(path, output / path.name)
        (output / "scene.usda").write_text("# Test fixture only, not native evidence\n")
        Image.new("RGB", (32, 32), "yellow").save(output / "review-00.jpg")
        if record:
            import cv2
            import numpy as np

            writer = cv2.VideoWriter(
                str(output / "isaac-scene-orbit.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                24,
                (960, 540),
            )
            assert writer.isOpened()
            for _ in range(192):
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
                frame[100:400, 100:800] = 200
                cv2.putText(
                    frame,
                    "OFFLINE TEST FIXTURE",
                    (100, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                writer.write(frame)
            writer.release()
        return {"native_runtime": False, "fixture": True}

    def validate(self, proposal, spec, rendered, output):
        output.mkdir(parents=True, exist_ok=True)
        return {
            "passed": self.passed,
            "reports": {},
            "images": [],
            "failures": [] if self.passed else ["penetration"],
        }


def config(revisions=3):
    return ProjectConfig(
        api=APISettings(model="test"),
        sampling=SamplingSettings(uniform_frames=4, event_frames=0),
        review=ReviewSettings(max_revisions=revisions, passes_required=2),
    )


def test_repair_requires_two_passes_on_same_revision(tmp_path, understanding, proposal):
    video = tmp_path / "video.mp4"
    synthetic_video(video)
    provider = ScriptedProvider(
        understanding,
        proposal,
        [
            "acceptable_with_stated_uncertainties",
            "repair",
            "acceptable_with_stated_uncertainties",
            "acceptable_with_stated_uncertainties",
        ],
    )
    pipeline = Pipeline(config(), provider, FakeBackend())
    run = tmp_path / "run"
    result = pipeline.run(video, run, "box_demo")
    assert result["status"] == "awaiting_approval" and result["revision"] == 2
    assert provider.calls[-2:] == ["review-2-1", "review-2-2"]
    pipeline.approve(
        run,
        "test reviewer",
        "Inspected source and native fixture states for this offline test only.",
    )
    assert read_json(run / "status.json")["status"] == "ready_to_record"
    spec = run / "revisions/002/scene.json"
    spec.write_text(spec.read_text() + " ")
    with pytest.raises(IntegrityError):
        pipeline.approve(run, "reviewer", "Changed source must invalidate the prior approval.")


def test_physics_failure_cannot_be_overridden_by_model(tmp_path, understanding, proposal):
    video = tmp_path / "video.mp4"
    synthetic_video(video)
    provider = ScriptedProvider(understanding, proposal, ["acceptable_with_stated_uncertainties"])
    pipeline = Pipeline(config(1), provider, FakeBackend(passed=False))
    with pytest.raises(NeedsAttention, match="Maximum"):
        pipeline.run(video, tmp_path / "run", "box_demo")
    assert not (tmp_path / "run/candidate.json").exists()


def test_source_changed_rejects_cached_candidate(tmp_path, understanding, proposal):
    video = tmp_path / "video.mp4"
    synthetic_video(video)
    provider = ScriptedProvider(
        understanding, proposal, ["acceptable_with_stated_uncertainties"] * 2
    )
    pipeline = Pipeline(config(), provider, FakeBackend())
    pipeline.run(video, tmp_path / "run", "box_demo")
    synthetic_video(video, count=40)
    with pytest.raises(IntegrityError, match="Source changed"):
        pipeline.run(video, tmp_path / "run", "box_demo")


def test_offline_full_workflow_records_and_reviews_actual_output(tmp_path, understanding, proposal):
    video = tmp_path / "video.mp4"
    synthetic_video(video)
    provider = ScriptedProvider(
        understanding, proposal, ["acceptable_with_stated_uncertainties"] * 4
    )
    pipeline = Pipeline(config(), provider, FakeBackend())
    run = tmp_path / "run"
    pipeline.run(video, run, "box_demo")
    pipeline.approve(
        run, "fixture tester", "Inspected only offline fixtures, not native simulation evidence."
    )
    result = pipeline.record(run)
    assert result["status"] == "complete"
    assert provider.calls[-2:] == ["recording-review-1", "recording-review-2"]
    assert read_json(run / "recording/inspection/video-integrity.json")["frames"] == 192
    count = len(provider.calls)
    assert pipeline.run(video, run, "box_demo")["status"] == "complete"
    assert len(provider.calls) == count
    delivery = read_json(run / "delivery.json")
    assert "approval.json" in {r["path"] for r in delivery["files"]}
    (run / "recording/scene.usda").write_text("modified")
    with pytest.raises(IntegrityError):
        pipeline.run(video, run, "box_demo")


def test_modified_compiler_invalidates_recording(tmp_path, understanding, proposal):
    video = tmp_path / "video.mp4"
    synthetic_video(video)
    pipeline = Pipeline(
        config(),
        ScriptedProvider(understanding, proposal, ["acceptable_with_stated_uncertainties"] * 2),
        FakeBackend(),
    )
    run = tmp_path / "run"
    pipeline.run(video, run, "box_demo")
    pipeline.approve(run, "tester", "Inspected offline fixture states and reference panels.")
    worker = run / "revisions/001/render/isaac_worker.py"
    worker.write_text(worker.read_text() + "\n# changed\n")
    with pytest.raises(IntegrityError):
        pipeline.record(run)


def test_refinement_evidence_is_retained_for_final_video_review(tmp_path, understanding, proposal):
    from video_real2sim.models import EvidenceRequest

    class RefiningProvider(ScriptedProvider):
        def __init__(self):
            super().__init__(understanding, proposal, ["acceptable_with_stated_uncertainties"] * 4)
            self.final_images = []

        def call(self, stage, prompt, images, schema, directory, **kwargs):
            if stage == "understanding":
                initial = understanding.model_copy(deep=True)
                initial.missing_evidence = [
                    EvidenceRequest(
                        interval_frames=[5, 6],
                        reason="Resolve occlusion",
                        requested_view="Source frames",
                    )
                ]
                return initial
            if stage.startswith("recording-review"):
                self.final_images = images
            return super().call(stage, prompt, images, schema, directory, **kwargs)

    video = tmp_path / "video.mp4"
    synthetic_video(video)
    provider = RefiningProvider()
    pipeline = Pipeline(config(), provider, FakeBackend())
    run = tmp_path / "run"
    pipeline.run(video, run, "box_demo")
    evidence = read_json(run / "source/evidence.json")
    assert {5, 6} <= set(evidence["observed_frames"])
    pipeline.approve(
        run, "fixture reviewer", "Inspected offline fixtures and refinement frames only."
    )
    assert pipeline.record(run)["status"] == "complete"
    assert {"frame-0000005.jpg", "frame-0000006.jpg"} <= {p.name for p in provider.final_images}
