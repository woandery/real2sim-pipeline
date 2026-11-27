import json
from pathlib import Path

import pytest

from video_real2sim.models import Proposal, Understanding

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def proposal_data():
    return json.loads((EXAMPLES / "box-proposal.json").read_text())


@pytest.fixture
def proposal(proposal_data):
    return Proposal.model_validate(proposal_data)


@pytest.fixture
def understanding():
    return Understanding.model_validate(
        json.loads((EXAMPLES / "box-understanding.json").read_text())
    )
