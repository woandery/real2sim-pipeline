import copy

import pytest

from video_real2sim.backend import check_probe
from video_real2sim.config import RuntimeSettings
from video_real2sim.errors import IntegrityError


def probe():
    return {
        "native_runtime": True,
        "source_spec_sha256": "spec",
        "source_usd_sha256": "usd",
        "native_rest_images": [{"file": "fixture.png", "sha256": "test"}],
        "contact_capture_errors": [],
        "image_capture_errors": [],
        "joint_types": {"hinge": "revolute"},
        "states": [
            {
                "integration_clock": {
                    "all_explicit_steps_checked": True,
                    "native_steps": 180,
                    "expected_native_steps": 180,
                    "native_duration_seconds": 1.5,
                    "expected_duration_seconds": 1.5,
                },
                "body_drift": {
                    "box": {"finite": True, "translation_m": 0.0001, "rotation_deg": 0.01}
                },
                "contact_summary": {"min_separation_m": -0.00001},
                "trace": [
                    {
                        "bodies": {"box": {"position": [0, 0, 0]}},
                        "measured_joint_positions": {"hinge": 0.0},
                    },
                    {
                        "bodies": {"box": {"position": [0, 0, 0]}},
                        "measured_joint_positions": {"hinge": -98.24},
                    },
                ],
            }
        ],
    }


def check(raw, rest=True):
    return check_probe(
        raw, {"spec_sha256": "spec", "usd_sha256": "usd"}, RuntimeSettings(), enforce_rest=rest
    )


def test_physics_clock_must_be_measured():
    data = probe()
    assert check(data)[1] == []
    data["states"][0]["integration_clock"]["native_steps"] = 181
    assert any("clock" in f for f in check(data)[1])


def test_severe_penetration_and_unstable_rest_fail():
    data = probe()
    data["states"][0]["contact_summary"]["min_separation_m"] = -0.01
    data["states"][0]["body_drift"]["box"]["translation_m"] = 0.2
    assert len(check(data)[1]) == 2


def test_sweep_uses_measured_angle_and_units():
    data = probe()
    data["states"][0]["continuous_joint_sweep"] = {"joint_id": "hinge", "end": -98.0}
    assert check(data, rest=False)[1] == []
    stalled = copy.deepcopy(data)
    stalled["states"][0]["trace"][-1]["measured_joint_positions"]["hinge"] = -10.0
    assert any("target" in f for f in check(stalled, rest=False)[1])


def test_wrong_scene_cannot_supply_physics():
    data = probe()
    data["source_spec_sha256"] = "different"
    with pytest.raises(IntegrityError):
        check(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0])
def test_nonfinite_or_empty_clock_never_qualifies(value):
    data = probe()
    data["states"][0]["integration_clock"]["native_duration_seconds"] = value
    assert any("clock" in f for f in check(data)[1])
