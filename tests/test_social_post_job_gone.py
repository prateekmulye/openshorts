"""Social-post route guards with fake key/provider boundaries; no real posts."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

import app as app_module


@pytest.fixture
def social_route(tmp_path, monkeypatch):
    job_id = "job-gone-mid-request"
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "source_metadata.json").write_text("{}")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(app_module, "jobs", {
        job_id: {"status": "completed", "result": {"clips": []}, "user_id": "fixture-owner"},
    })
    state = {"drop": False, "key": "fixture-key", "resolutions": 0,
             "provider_calls": [], "queue_calls": []}

    async def resolve(request, body_key=None):
        state["resolutions"] += 1
        await asyncio.sleep(0)
        if state["drop"]:
            app_module.jobs.pop(job_id, None)
        return state["key"], "fixture-profile"

    def post(*args, **kwargs):
        state["provider_calls"].append(args)
        pytest.fail("guarded request reached the external posting boundary")

    async def enqueue(item):
        state["queue_calls"].append(item)
        pytest.fail("guarded request enqueued work")

    monkeypatch.setattr(app_module, "resolve_upload_post", resolve)
    monkeypatch.setattr(app_module, "_post_video_blocking", post)
    monkeypatch.setattr(app_module.job_queue, "put", enqueue)

    def request():
        async def run():
            transport = httpx.ASGITransport(app=app_module.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/api/social/post", json={
                    "job_id": job_id, "clip_index": 0,
                    "platforms": ["tiktok"], "user_id": "fixture-profile"})
        response = asyncio.run(run())
        assert state["provider_calls"] == []
        assert state["queue_calls"] == []
        return response

    return job_id, state, request


def test_job_dropped_during_key_resolution_is_a_404(social_route):
    job_id, state, request = social_route
    state["drop"] = True
    response = request()
    assert state["resolutions"] == 1
    assert job_id not in app_module.jobs
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_job_absent_at_start_does_not_resolve_key(social_route):
    job_id, state, request = social_route
    app_module.jobs.pop(job_id)
    response = request()
    assert state["resolutions"] == 0
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_missing_key_keeps_existing_error(social_route):
    _, state, request = social_route
    state["key"] = None
    response = request()
    assert response.status_code == 400
    assert response.json() == {"detail": "Missing Upload-Post API key"}


def test_other_owner_still_gets_not_found(social_route, monkeypatch):
    _, _, request = social_route
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

    async def other_user(request):
        return SimpleNamespace(id="different-owner")

    monkeypatch.setattr(app_module, "_user_from_request", other_user)
    response = request()
    assert response.status_code == 404
    assert response.json() == {"detail": "Not found"}


def test_existing_empty_job_is_not_misclassified_as_absent(social_route):
    job_id, _, request = social_route
    app_module.jobs[job_id] = {}
    response = request()
    assert response.status_code == 400
    assert response.json() == {"detail": "Job result not available"}
