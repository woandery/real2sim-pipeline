"""Exercise subprocess bookkeeping with tiny CPU workers, without pretending Isaac ran."""

import json
import sys

import pytest

from video_real2sim import backend
from video_real2sim.artifacts import read_json
from video_real2sim.config import RuntimeSettings
from video_real2sim.errors import IntegrityError, NeedsAttention, WorkflowError


def runner(tmp_path, monkeypatch, code, timeout=30):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "worker.py").write_text(code)
    monkeypatch.setattr(backend, "NATIVE", sources)
    settings = RuntimeSettings(python=sys.executable, accept_eula=True)
    # Short timeout only in this local test; production configuration has >=30s.
    settings = settings.model_copy(update={"timeout_seconds": timeout})
    return backend.IsaacBackend(settings, key_env="PRIVATE_TOKEN"), tmp_path / "output"


def test_child_secret_isolation_and_hash_bound_cache(tmp_path, monkeypatch):
    for key in ("PRIVATE_TOKEN", "OPENAI_API_KEY", "THIRD_PARTY_API_KEY"):
        monkeypatch.setenv(key, "test-do-not-leak")
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    code = """import os, sys, json
from pathlib import Path
assert not any(k in os.environ for k in ['PRIVATE_TOKEN', 'OPENAI_API_KEY', 'THIRD_PARTY_API_KEY', 'PYTHONOPTIMIZE'])
assert __debug__
Path(sys.argv[1]).write_text(json.dumps({'native_runtime': False, 'cpu_test': True}))
"""
    worker, output = runner(tmp_path, monkeypatch, code)
    args = [str(output / "result.json")]
    worker._job("worker.py", args, output, {})
    receipt = read_json(output / "job.json")
    assert receipt["status"] == "completed"
    worker._job("worker.py", args, output, {})
    assert read_json(output / "job.json") == receipt
    (output / "result.json").write_text("{}")
    with pytest.raises(IntegrityError):
        worker._job("worker.py", args, output, {})


@pytest.mark.parametrize(
    "code,timeout,status",
    [("raise SystemExit(7)", 30, "failed"), ("import time; time.sleep(60)", 1, "timed_out")],
)
def test_failed_or_timed_out_child_not_restarted(tmp_path, monkeypatch, code, timeout, status):
    worker, output = runner(tmp_path, monkeypatch, code, timeout)
    with pytest.raises(WorkflowError):
        worker._job("worker.py", [], output, {})
    assert read_json(output / "job.json")["status"] == status
    with pytest.raises(NeedsAttention):
        worker._job("worker.py", [], output, {})


def test_pending_job_is_not_duplicated(tmp_path, monkeypatch):
    worker, output = runner(tmp_path, monkeypatch, "pass")
    worker._job("worker.py", [], output, {})
    receipt = read_json(output / "job.json")
    receipt["status"] = "running"
    (output / "job.json").write_text(json.dumps(receipt))
    with pytest.raises(NeedsAttention):
        worker._job("worker.py", [], output, {})


def test_worker_code_change_during_execution_cannot_be_registered(tmp_path, monkeypatch):
    code = "from pathlib import Path\np=Path(__file__)\np.write_text(p.read_text()+'\\n# altered\\n')\n"
    worker, output = runner(tmp_path, monkeypatch, code)
    with pytest.raises(IntegrityError, match="changed during execution"):
        worker._job("worker.py", [], output, {})
    assert read_json(output / "job.json")["status"] == "failed"
