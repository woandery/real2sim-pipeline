import pytest
from pydantic import ValidationError

from video_real2sim.models import Proposal


def test_example_matches_video(proposal, understanding):
    understanding.validate_video("box_demo", 30)
    proposal.scene.validate_evidence(understanding, 30)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_sizes_rejected(proposal_data, invalid):
    proposal_data["scene"]["assets"][0]["parts"][0]["size"][0] = invalid
    with pytest.raises(ValidationError):
        Proposal.model_validate(proposal_data)


def test_code_and_paths_rejected(proposal_data):
    proposal_data["scene"]["assets"][0]["id"] = "../../escape"
    proposal_data["python"] = "print('unsafe')"
    with pytest.raises(ValidationError):
        Proposal.model_validate(proposal_data)


def test_passive_drive_survives_serialization(proposal_data):
    proposal_data["scene"]["joints"][0]["drive"] = None
    proposal_data["scene"]["diagnostic_states"][1]["joint_positions"] = []
    proposal_data["validation"]["sweeps"] = []
    data = Proposal.model_validate(proposal_data).scene.compiler_dict()
    assert "drive" in data["joints"][0]
    assert data["joints"][0]["drive"] is None


def test_unknown_references_rejected(proposal_data):
    proposal_data["scene"]["joints"][0]["child"] = "missing"
    with pytest.raises(ValidationError, match="reference"):
        Proposal.model_validate(proposal_data)


def test_sweep_required_for_moving_joint(proposal_data):
    proposal_data["validation"]["sweeps"] = []
    with pytest.raises(ValidationError, match="continuous sweep"):
        Proposal.model_validate(proposal_data)


def test_no_unaccounted_states(proposal_data):
    proposal_data["validation"]["rest_checks"].pop()
    with pytest.raises(ValidationError, match="Every state"):
        Proposal.model_validate(proposal_data)


def test_evidence_frame_bounds(understanding):
    with pytest.raises(ValueError, match="outside"):
        understanding.validate_video("box_demo", 20)


def test_initial_state_cannot_diverge_from_asset_pose(proposal_data):
    proposal_data["scene"]["diagnostic_states"][0]["joint_positions"] = [
        {"joint_id": "lid_hinge", "value": -20.0}
    ]
    with pytest.raises(ValidationError, match="Initial state"):
        Proposal.model_validate(proposal_data)


def test_evidence_must_be_in_actual_model_inputs(understanding, proposal):
    with pytest.raises(ValueError, match="not supplied"):
        understanding.validate_video("box_demo", 30, {0})
    with pytest.raises(ValueError, match="not supplied"):
        proposal.scene.validate_evidence(understanding, 30, {0})
    understanding.validate_video("box_demo", 30, {0, 29})
    proposal.scene.validate_evidence(understanding, 30, {0, 29})
