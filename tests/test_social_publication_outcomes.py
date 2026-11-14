"""Route contract with a fake external Upload-Post HTTP service; no real posts.

Source: https://docs.upload-post.com/api/upload-status/ (fetched 2026-11-14 UTC).
A completed upload can include skipped platforms or a TikTok inbox fallback.
"""
import asyncio
import json

import httpx
import pytest

import app as app_module


@pytest.fixture
def status_route(monkeypatch):
    original_client = httpx.AsyncClient
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(app_module, "_post_request_owners", {"mine": None, "theirs": "other-owner"})
    monkeypatch.setattr(app_module, "_logged_post_failures", set())
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)
    upstream = {"body": {"status": "completed", "results": []}, "code": 200, "calls": []}

    def handle(request):
        upstream["calls"].append(request)
        return httpx.Response(upstream["code"], content=json.dumps(upstream["body"]),
                              headers={"Content-Type": "application/json"})

    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handle), **kwargs))

    def get(path="/api/social/post-status/mine", headers=None):
        async def run():
            async with original_client(transport=httpx.ASGITransport(app=app_module.app),
                                       base_url="http://testserver") as client:
                return await client.get(path, headers=headers if headers is not None else {
                    "X-Upload-Post-Key": "fixture-caller-key"})
        return asyncio.run(run())

    return upstream, get


def test_known_self_host_request_polls_with_callers_key(status_route):
    upstream, get = status_route
    assert get().json() == {"status": "completed", "results": []}
    assert len(upstream["calls"]) == 1
    request = upstream["calls"][0]
    assert str(request.url) == "https://api.upload-post.com/api/uploadposts/status?request_id=mine"
    assert request.headers["Authorization"] == "Apikey fixture-caller-key"


@pytest.mark.parametrize("path,code,body", [
    ("/api/social/post-status/theirs", 404, {"detail": "Not found"}),
    ("/api/social/post-status/unseen", 200, {"status": "unknown"}),
])
def test_owner_and_unknown_guards_do_not_call_vendor(status_route, path, code, body):
    upstream, get = status_route
    response = get(path)
    assert response.status_code == code
    assert response.json() == body
    assert upstream["calls"] == []


def test_missing_key_does_not_call_vendor(status_route):
    upstream, get = status_route
    assert get(headers={}).json() == {"status": "unknown"}
    assert upstream["calls"] == []


def test_only_allowlisted_outcome_fields_leave_route(status_route):
    upstream, get = status_route
    upstream["body"] = {"status": "completed", "account": "private", "results": [
        {"platform": "youtube", "success": True, "status": "completed", "message": "Published",
         "post_url": "https://private.invalid", "credentials": "must-not-leak"},
        {"platform": "instagram", "success": True, "status": "skipped", "skipped": True,
         "skip_reason": "profile_platform_not_configured"},
        {"platform": "tiktok", "success": True, "status": "completed", "fallback_to_inbox": True},
        {"platform": "x", "success": False, "status": "failed", "skipped": False,
         "fallback_to_inbox": False, "error": "Rejected"},
    ]}
    assert get().json() == {"status": "completed", "results": [
        {"platform": "youtube", "success": True, "status": "completed", "message": "Published"},
        {"platform": "instagram", "success": True, "status": "skipped", "skipped": True, "message": None},
        {"platform": "tiktok", "success": True, "status": "completed", "fallback_to_inbox": True, "message": None},
        {"platform": "x", "success": False, "status": "failed", "skipped": False,
         "fallback_to_inbox": False, "message": "Rejected"},
    ]}


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, [], {}])
def test_success_never_coerces_non_boolean(status_route, value):
    upstream, get = status_route
    upstream["body"] = {"status": "completed", "results": [{"platform": "youtube", "success": value}]}
    assert get().json()["results"][0]["success"] is False


@pytest.mark.parametrize("platform", ["", "   ", "\t\n"])
def test_blank_platform_cannot_be_success(status_route, platform):
    upstream, get = status_route
    upstream["body"] = {"status": "completed", "results": [
        {"platform": platform, "success": True}]}
    assert get().json()["results"][0] == {
        "platform": platform, "success": False, "message": None}


@pytest.mark.parametrize("field", ["skipped", "fallback_to_inbox", "status"])
@pytest.mark.parametrize("value", ["unrecognized", 1, None, [], {}])
def test_malformed_outcome_field_cannot_be_success(status_route, field, value):
    upstream, get = status_route
    upstream["body"] = {"status": "completed", "results": [
        {"platform": "tiktok", "success": True, field: value}]}
    result = get().json()["results"][0]
    assert result["success"] is False
    assert result[field] == ("unknown" if field == "status" else None)


@pytest.mark.parametrize("body", [[], "completed", 1, None,
    {"status": True, "results": []}, {"status": ["completed"], "results": []},
    {"status": "completed", "results": "bad"}, {"status": "completed", "results": 1},
    {"status": "completed", "results": {"youtube": {"success": True}}},
    {"status": "completed", "results": [None]},
])
def test_malformed_envelope_is_unknown(status_route, body):
    upstream, get = status_route
    upstream["body"] = body
    assert get().json() == {"status": "unknown", "results": []}


def test_upstream_error_is_unknown_even_with_completed_body(status_route):
    upstream, get = status_route
    upstream["code"] = 503
    assert get().json() == {"status": "unknown"}


def test_health_advertises_publication_outcomes(status_route):
    _, get = status_route
    response = get("/health")
    assert response.status_code == 200
    # This branch inherits source-range support; stacking must preserve both capabilities.
    assert response.json() == {
        "status": "ok", "capabilities": {"source_ranges": True, "publication_outcomes": True}}
