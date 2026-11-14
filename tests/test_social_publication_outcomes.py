"""Route contract with a fake external Upload-Post HTTP service; no real posts.

Source: https://docs.upload-post.com/api/upload-status/ (fetched 2026-11-14 UTC).
A completed upload can include skipped platforms or a TikTok inbox fallback.
"""
import asyncio
import json
from types import SimpleNamespace

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
        if upstream.get("error"):
            raise upstream["error"]("fixture-private-key in provider exception", request=request)
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


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_upstream_error_is_unknown_even_with_completed_body(status_route, code):
    upstream, get = status_route
    upstream["code"] = code
    assert get().json() == {"status": "unknown"}


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 599])
def test_transient_poll_error_stays_retryable_without_provider_text(status_route, code, capsys):
    # A 200/unknown here permanently holds Wheelman's accepted attempt. A
    # transient observation failure must instead raise at the HTTP client so
    # its publisher retains processing and polls again without another POST.
    # This replaces the old blanket 503->unknown expectation for that reason.
    upstream, get = status_route
    upstream["code"] = code
    upstream["body"] = {"status": "completed", "credentials": "fixture-private-key"}
    response = get()
    assert response.status_code == 503
    assert response.json() == {"detail": "Publication status temporarily unavailable. Try again later."}
    assert app_module._post_request_owners["mine"] is None
    upstream["code"] = 200
    upstream["body"] = {"status": "processing", "results": []}
    assert get().json() == {"status": "processing", "results": []}
    assert [request.method for request in upstream["calls"]] == ["GET", "GET"]
    captured = capsys.readouterr()
    assert "fixture-private-key" not in captured.out + captured.err + response.text


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError])
def test_transport_poll_error_is_retryable_and_redacted(status_route, error, capsys):
    upstream, get = status_route
    upstream["error"] = error
    response = get()
    assert response.status_code == 503
    assert response.json() == {"detail": "Publication status temporarily unavailable. Try again later."}
    assert app_module._post_request_owners["mine"] is None
    assert [request.method for request in upstream["calls"]] == ["GET"]
    captured = capsys.readouterr()
    assert "fixture-private-key" not in captured.out + captured.err + response.text


def test_health_advertises_publication_outcomes(status_route):
    _, get = status_route
    response = get("/health")
    assert response.status_code == 200
    # This branch inherits source-range support; stacking must preserve both capabilities.
    assert response.json() == {
        "status": "ok", "capabilities": {"source_ranges": True, "publication_outcomes": True}}


@pytest.mark.parametrize("request_id", ["mine", "different-request"])
def test_status_preserves_provider_request_id_for_caller_reconciliation(status_route, request_id):
    upstream, get = status_route
    upstream["body"]["request_id"] = request_id
    assert get().json()["request_id"] == request_id


@pytest.mark.parametrize("request_id", [True, 1, [], {}, "", "bad/id"])
def test_status_malformed_request_id_is_unknown(status_route, request_id):
    upstream, get = status_route
    upstream["body"]["request_id"] = request_id
    assert get().json() == {"status": "unknown", "results": []}


@pytest.fixture
def post_route(tmp_path, monkeypatch):
    """Real submit/status routes and multipart upload; only vendor HTTP is fake."""
    original_client, original_async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "_post_request_owners", {})
    monkeypatch.setattr(app_module, "_logged_post_failures", set())
    monkeypatch.setattr(app_module, "_social_post_times", {})
    monkeypatch.setattr(app_module, "_social_post_days", {})
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)
    job_dir = tmp_path / "fixture-job"
    job_dir.mkdir()
    (job_dir / "clip.mp4").write_bytes(b"fixture media; vendor HTTP is fake")
    (job_dir / "clip_metadata.json").write_text("{}")
    monkeypatch.setattr(app_module, "jobs", {"fixture-job": {
        "user_id": None, "result": {"clips": [{"video_url": "/videos/fixture-job/clip.mp4"}]}}})
    vendor = {"submit": {"request_id": "accepted-request", "status": "processing"},
              "status": {"request_id": "accepted-request", "status": "completed", "results": [
                  {"platform": "youtube", "success": True, "status": "completed",
                   "skipped": False, "fallback_to_inbox": False}]}, "calls": []}

    def handle(request):
        vendor["calls"].append(request)
        return httpx.Response(200, json=vendor["submit" if request.method == "POST" else "status"])

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(app_module.httpx, "Client", lambda **kw: original_client(transport=transport, **kw))
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda **kw: original_async_client(transport=transport, **kw))

    def call(method, path, **kwargs):
        async def run():
            async with original_async_client(transport=httpx.ASGITransport(app=app_module.app),
                                             base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(run())
    return vendor, call


def _post(call, **fields):
    return call("POST", "/api/social/post", headers={"X-Upload-Post-Key": "fixture-header-key"}, json={
        "job_id": "fixture-job", "clip_index": 0, "api_key": "fixture-body-key",
        "user_id": "fixture-client-profile", "platforms": ["youtube"], **fields})


def test_submit_then_status_preserves_acceptance_and_outcomes(post_route):
    vendor, call = post_route
    submitted = _post(call)
    assert submitted.status_code == 200, submitted.text
    assert submitted.json() == {"request_id": "accepted-request", "status": "processing"}
    assert app_module._post_request_owners == {"accepted-request": None}
    status = call("GET", "/api/social/post-status/accepted-request",
                  headers={"X-Upload-Post-Key": "fixture-header-key"})
    assert status.json() == {**vendor["status"], "results": [
        {**vendor["status"]["results"][0], "message": None}]}
    assert [r.method for r in vendor["calls"]] == ["POST", "GET"]
    assert all(r.headers["Authorization"] == "Apikey fixture-header-key" for r in vendor["calls"])


@pytest.mark.parametrize("flag", ["skipped", "fallback_to_inbox"])
def test_immediate_submit_preserves_outcome_evidence(post_route, flag):
    vendor, call = post_route
    vendor["submit"] = {"request_id": "accepted-request", "status": "completed", "results": [
        {"platform": "youtube", "success": True, "status": "skipped", flag: True}]}
    assert _post(call).json() == vendor["submit"]


def test_scheduled_submit_keeps_upstream_response_and_schedule_fields(post_route):
    vendor, call = post_route
    vendor["submit"] = {"success": True, "job_id": "scheduled-job", "status": "scheduled"}
    response = _post(call, scheduled_date="2030-01-01T12:00:00Z", timezone="UTC")
    assert response.json() == vendor["submit"]
    body = vendor["calls"][0].content
    assert b'2030-01-01T12:00:00Z' in body
    assert b'name="timezone"' in body


def test_managed_submit_and_poll_force_credentials_and_owner(post_route, monkeypatch):
    vendor, call = post_route
    from cloud import config
    user = SimpleNamespace(id="owner", plan="starter", upload_post_profile="os_owner", entitled=True)

    async def authenticated(request):
        return user

    async def connected(profile, platforms):
        return []  # External account lookup only.

    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "_user_from_request", authenticated)
    monkeypatch.setattr(app_module, "managed_keys", SimpleNamespace(
        has_active_entitlement=lambda u: u.entitled, upload_post_key=lambda: "fixture-managed-key"))
    monkeypatch.setattr(app_module, "_platforms_needing_reconnect", connected)
    app_module.jobs["fixture-job"]["user_id"] = "owner"
    response = _post(call)
    assert response.status_code == 200, response.text
    assert vendor["calls"][0].headers["Authorization"] == "Apikey fixture-managed-key"
    assert b'os_owner' in vendor["calls"][0].content
    assert b'fixture-client-profile' not in vendor["calls"][0].content
    assert app_module._post_request_owners == {"accepted-request": "owner"}
    assert call("GET", "/api/social/post-status/accepted-request",
                headers={"X-Upload-Post-Key": "fixture-attacker-key"}).json()["status"] == "completed"
    assert vendor["calls"][1].headers["Authorization"] == "Apikey fixture-managed-key"
    user.id = "other-owner"
    assert call("GET", "/api/social/post-status/accepted-request").status_code == 404
    assert _post(call).status_code == 404
    assert len(vendor["calls"]) == 2
