import json

import httpx
import pytest
from openai import OpenAI
from PIL import Image

from video_real2sim.artifacts import read_json, write_json
from video_real2sim.config import APISettings
from video_real2sim.errors import IntegrityError, NeedsAttention, WorkflowError
from video_real2sim.models import Understanding
from video_real2sim.provider import OpenAIProvider


def response(data, status="completed"):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(data), "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def make_provider(handler):
    client = OpenAI(
        api_key="test-secret-never-log",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return OpenAIProvider(APISettings(model="test-model"), client)


def test_actual_sdk_structured_vision_request_and_cache(tmp_path, understanding):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json=response(understanding.model_dump()))

    provider = make_provider(handler)
    image = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "red").save(image)
    directory = tmp_path / "call"
    first = provider.call("analyze", "inspect scene", [image], Understanding, directory)
    second = provider.call("analyze", "inspect scene", [image], Understanding, directory)
    assert first == second == understanding and len(requests) == 1
    body = requests[0]
    assert body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert "test-secret-never-log" not in "".join(p.read_text() for p in directory.glob("*.json"))
    Image.new("RGB", (20, 20), "blue").save(image)
    with pytest.raises(IntegrityError):
        provider.call("analyze", "inspect scene", [image], Understanding, directory)


def test_tampered_cached_result_rejected(tmp_path, understanding):
    provider = make_provider(
        lambda req: httpx.Response(200, json=response(understanding.model_dump()))
    )
    provider.call("analyze", "prompt", [], Understanding, tmp_path)
    data = read_json(tmp_path / "result.json")
    data["summary_zh"] = "modified"
    write_json(tmp_path / "result.json", data)
    with pytest.raises(IntegrityError):
        provider.call("analyze", "prompt", [], Understanding, tmp_path)


def test_transport_failure_not_repeated_on_resume(tmp_path):
    attempts = []

    def fail(request):
        attempts.append(1)
        raise httpx.ReadTimeout("ambiguous", request=request)

    provider = make_provider(fail)
    for _ in range(2):
        with pytest.raises(NeedsAttention):
            provider.call("analyze", "prompt", [], Understanding, tmp_path)
    assert len(attempts) == 1
    assert read_json(tmp_path / "receipt.json")["status"] == "uncertain"


def test_http_error_redacts_body_and_headers(tmp_path):
    provider = make_provider(
        lambda req: httpx.Response(
            401,
            json={"error": {"message": "test-secret-never-log", "type": "invalid_request_error"}},
        )
    )
    with pytest.raises(WorkflowError, match="401"):
        provider.call("analyze", "prompt", [], Understanding, tmp_path)
    assert "test-secret-never-log" not in (tmp_path / "receipt.json").read_text()


def test_incomplete_response_does_not_pass(tmp_path, understanding):
    provider = make_provider(
        lambda req: httpx.Response(200, json=response(understanding.model_dump(), "incomplete"))
    )
    with pytest.raises(WorkflowError):
        provider.call("analyze", "prompt", [], Understanding, tmp_path)
    assert read_json(tmp_path / "receipt.json")["status"] == "uncertain"


def test_api_smoke_uses_actual_sdk_and_distinguishes_scope(tmp_path):
    from video_real2sim.smoke import api_smoke

    observation = dict(
        visible_objects=["red square"],
        spatial_description="Centered square.",
        uncertainties=["Synthetic image."],
    )
    provider = make_provider(lambda req: httpx.Response(200, json=response(observation)))
    image = tmp_path / "image.png"
    Image.new("RGB", (24, 24), "red").save(image)
    result = api_smoke(provider, image, tmp_path / "call")
    assert result["status"] == "api_smoke_completed"
    assert "no physics" in result["scope"]
    assert result["observation"] == observation


def test_no_key_fails_before_creating_billable_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider(APISettings(model="test-model"))
    with pytest.raises(WorkflowError, match="Set OPENAI_API_KEY"):
        provider.call("test", "test", [], Understanding, tmp_path)
    assert not (tmp_path / "receipt.json").exists()


def test_base64_budget_stops_before_sending(tmp_path):
    image = tmp_path / "large.jpg"
    with image.open("wb") as stream:
        stream.truncate(37_000_000)
    provider = make_provider(lambda req: pytest.fail("Over-budget request must not be sent"))
    with pytest.raises(NeedsAttention, match="Encoded vision request"):
        provider.call("test", "test", [image], Understanding, tmp_path / "call")
    assert not (tmp_path / "call/receipt.json").exists()
