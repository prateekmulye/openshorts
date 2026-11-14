"""Agent uploads (POST/PUT /api/uploads -> process with upload_id) and the
per-job captions switch, through the MCP transport where the agent uses them."""
import asyncio
import os
import shutil
import subprocess
import threading

import httpx
import pytest

app_module = pytest.importorskip("app")


def _client():
    transport = httpx.ASGITransport(app=app_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver",
                             headers={"X-Gemini-Key": "test-key"})


def _mcp(tool, arguments):
    async def _do():
        async with _client() as c:
            return await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                              "params": {"name": tool, "arguments": arguments}})
    return asyncio.run(_do())


def _tool_payload(resp):
    import json
    return json.loads(resp.json()["result"]["content"][0]["text"])


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"; up_root = tmp_path / "uploads"
    out_root.mkdir(); up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    # Never spawn main.py: capture the job instead.
    captured = {}
    async def fake_enqueue(*a, **k):
        captured["args"] = a; captured["kwargs"] = k
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda p: 120.0)
    return out_root, up_root, captured


def test_create_upload_is_an_mcp_tool():
    names = [t["name"] for t in __import__("mcp_server").TOOLS]
    assert "create_upload" in names
    pv = next(t for t in __import__("mcp_server").TOOLS if t["name"] == "process_video")
    assert "upload_id" in pv["inputSchema"]["properties"]
    assert "captions" in pv["inputSchema"]["properties"]
    assert "source_url" not in pv["inputSchema"]["required"]


def test_process_video_needs_a_source():
    payload = _tool_payload(_mcp("process_video", {"confirm_rights": True}))
    assert "source_url" in payload["error"] and "upload_id" in payload["error"]


def test_put_then_process_with_upload_id_and_captions_off(dirs, monkeypatch):
    out_root, up_root, _ = dirs
    slot = _tool_payload(_mcp("create_upload", {"filename": "../talk.mp4"}))
    assert slot["upload_url"].endswith(f"/api/uploads/{slot['upload_id']}")
    assert "talk.mp4" in app_module.pending_uploads[slot["upload_id"]]["path"]
    assert ".." not in os.path.basename(app_module.pending_uploads[slot["upload_id"]]["path"])

    async def _put(body):
        async with _client() as c:
            return await c.put(f"/api/uploads/{slot['upload_id']}", content=body)
    assert asyncio.run(_put(b"")).status_code == 400
    resp = asyncio.run(_put(b"\x00" * 5000))
    assert resp.status_code == 200 and resp.json()["bytes"] == 5000

    # Intercept the job instead of running main.py.
    started = {}
    async def fake_put(item):
        started["job"] = item
    monkeypatch.setattr(app_module.job_queue, "put", fake_put)

    payload = _tool_payload(_mcp("process_video", {
        "upload_id": slot["upload_id"], "confirm_rights": True, "captions": False}))
    assert "job_id" in payload, payload
    job = app_module.jobs[payload["job_id"]]
    assert job["env"]["AUTO_CAPTIONS"] == "0"
    assert slot["upload_id"] not in app_module.pending_uploads
    input_path = job["cmd"][job["cmd"].index("-i") + 1]
    assert os.path.exists(input_path) and input_path.startswith(str(up_root))


def test_captions_default_leaves_env_alone(dirs, monkeypatch):
    slot = _tool_payload(_mcp("create_upload", {}))
    async def _put():
        async with _client() as c:
            return await c.put(f"/api/uploads/{slot['upload_id']}", content=b"x" * 100)
    assert asyncio.run(_put()).status_code == 200
    async def fake_put(item): pass
    monkeypatch.setattr(app_module.job_queue, "put", fake_put)
    payload = _tool_payload(_mcp("process_video", {"upload_id": slot["upload_id"], "confirm_rights": True}))
    env = app_module.jobs[payload["job_id"]]["env"]
    assert "AUTO_CAPTIONS" not in env
    assert env["AUTO_HOOK"] == "1"  # MCP default matches the dashboard


def test_unknown_or_unfinished_upload_id(dirs):
    async def _put():
        async with _client() as c:
            return await c.put("/api/uploads/nope", content=b"x")
    assert asyncio.run(_put()).status_code == 404
    slot = _tool_payload(_mcp("create_upload", {}))
    payload = _tool_payload(_mcp("process_video", {"upload_id": slot["upload_id"], "confirm_rights": True}))
    assert "PUT" in payload["error"]


def test_sweep_expires_old_slots(dirs, monkeypatch):
    slot = _tool_payload(_mcp("create_upload", {}))
    path = app_module.pending_uploads[slot["upload_id"]]["path"]
    open(path, "wb").write(b"x")
    assert slot["upload_id"] not in app_module._sweep_pending_uploads(now=__import__("time").time() + 1)
    gone = app_module._sweep_pending_uploads(now=__import__("time").time() + app_module.UPLOAD_TTL_SECONDS + 1)
    assert slot["upload_id"] in gone and not os.path.exists(path)


def test_delete_upload(dirs):
    slot = _tool_payload(_mcp("create_upload", {}))
    async def _del():
        async with _client() as c:
            return await c.delete(f"/api/uploads/{slot['upload_id']}")
    assert asyncio.run(_del()).status_code == 200
    assert slot["upload_id"] not in app_module.pending_uploads


def test_tmpfiles_link_is_refreshed_through_the_page():
    import file_hosts
    stale = "https://tmpfiles.org/dl/1787841143.abc/wAwNGnPW0lXW/video.mp4"
    fresh = "https://tmpfiles.org/dl/1787841582.def/wAwNGnPW0lXW/video.mp4"
    seen = {}
    def fetch(url):
        seen["page"] = url
        return f'<a href="{fresh}">Download</a>'
    assert file_hosts.resolve_tmpfiles(stale, fetch=fetch) == fresh
    assert seen["page"] == "https://tmpfiles.org/wAwNGnPW0lXW/video.mp4"
    assert file_hosts.resolve_tmpfiles("https://tmpfiles.org/wAwNGnPW0lXW/video.mp4", fetch=fetch) == fresh
    assert file_hosts.resolve_tmpfiles(stale, fetch=lambda u: "<html>no link</html>") == stale
    assert not file_hosts.is_tmpfiles("https://youtube.com/watch?v=x")


def test_auto_hook_reaches_the_job_env(dirs, monkeypatch):
    slot = _tool_payload(_mcp("create_upload", {}))
    async def _put():
        async with _client() as c:
            return await c.put(f"/api/uploads/{slot['upload_id']}", content=b"x" * 100)
    assert asyncio.run(_put()).status_code == 200
    async def fake_put(item): pass
    monkeypatch.setattr(app_module.job_queue, "put", fake_put)
    payload = _tool_payload(_mcp("process_video", {"upload_id": slot["upload_id"], "confirm_rights": True,
                                                   "auto_hook": True, "hook_style": "yellow"}))
    env = app_module.jobs[payload["job_id"]]["env"]
    assert env["AUTO_HOOK"] == "1" and env["AUTO_HOOK_STYLE"] == "yellow"


def test_auto_hook_false_opts_out(dirs, monkeypatch):
    slot = _tool_payload(_mcp("create_upload", {}))
    async def _put():
        async with _client() as c:
            return await c.put(f"/api/uploads/{slot['upload_id']}", content=b"x" * 100)
    assert asyncio.run(_put()).status_code == 200
    async def fake_put(item): pass
    monkeypatch.setattr(app_module.job_queue, "put", fake_put)
    payload = _tool_payload(_mcp("process_video", {"upload_id": slot["upload_id"], "confirm_rights": True, "auto_hook": False}))
    assert "AUTO_HOOK" not in app_module.jobs[payload["job_id"]]["env"]


@pytest.fixture(params=[
    (b"x" * (1024 * 1024 + 1), 413, "File too large. Max size 1MB"),
    (b"", 400, "Empty body"),
    (b"not-a-video", 400, "The body is not a readable video file"),
], ids=["oversized", "empty", "unreadable"])
def rejected_upload(dirs, monkeypatch, request):
    """Real upload files with the external media probe returning unreadable."""
    monkeypatch.setattr(app_module, "MAX_FILE_SIZE_MB", 1)
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 0.0)
    slot = _tool_payload(_mcp("create_upload", {}))
    return slot["upload_id"], request.param


def _put_upload(upload_id, body):
    async def put():
        async with _client() as client:
            return await client.put(f"/api/uploads/{upload_id}", content=body)
    return asyncio.run(put())


def test_failed_replacement_upload_invalidates_prior_success(rejected_upload, monkeypatch):
    upload_id, (body, status, detail) = rejected_upload
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 120.0)
    assert _put_upload(upload_id, b"previous-video").status_code == 200
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 0.0)
    response = _put_upload(upload_id, body)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    slot = app_module.pending_uploads[upload_id]
    assert slot["complete"] is False and slot["bytes"] == 0
    assert not os.path.exists(slot["path"])


@pytest.mark.parametrize("previously_complete", [False, True])
def test_failed_upload_cleanup_tolerates_disappearance(rejected_upload, monkeypatch,
                                                       previously_complete):
    upload_id, (body, status, detail) = rejected_upload
    slot = app_module.pending_uploads[upload_id]
    if previously_complete:
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 120.0)
        assert _put_upload(upload_id, b"previous-video").status_code == 200
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 0.0)
    remove = os.remove
    calls = []

    def remove_after_disappearance(path):
        # Deterministic race at the OS boundary: another actor removes the
        # real temporary file just before this cleanup tries to remove it.
        assert path == slot["path"]
        calls.append(path)
        remove(path)
        remove(path)

    monkeypatch.setattr(app_module.os, "remove", remove_after_disappearance)
    response = _put_upload(upload_id, body)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert calls == [slot["path"]]
    assert slot["complete"] is False
    assert slot["bytes"] == 0
    assert not os.path.exists(slot["path"])


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("disk error")],
                         ids=["permission", "other-os-error"])
@pytest.mark.parametrize("previously_complete", [False, True])
def test_failed_upload_cleanup_preserves_other_os_errors(rejected_upload, monkeypatch, error,
                                                        previously_complete):
    upload_id, (body, _, _) = rejected_upload
    slot = app_module.pending_uploads[upload_id]
    if previously_complete:
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 120.0)
        assert _put_upload(upload_id, b"previous-video").status_code == 200
        monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 0.0)

    def remove(path):
        assert path == slot["path"]
        raise error

    monkeypatch.setattr(app_module.os, "remove", remove)
    with pytest.raises(type(error), match=str(error)):
        _put_upload(upload_id, body)
    assert os.path.exists(slot["path"])
    assert slot["complete"] is False
    assert slot["bytes"] == 0


def test_upload_cannot_be_consumed_while_probe_is_pending(dirs, monkeypatch):
    upload_id = _tool_payload(_mcp("create_upload", {}))["upload_id"]
    slot = app_module.pending_uploads[upload_id]
    assert _put_upload(upload_id, b"previous-video").status_code == 200
    started, release = threading.Event(), threading.Event()

    def probe(path):
        started.set()
        assert release.wait(5), "test did not release the media probe"
        return 120.0

    monkeypatch.setattr(app_module, "_media_duration_seconds", probe)

    async def run():
        async with _client() as client:
            put = asyncio.create_task(client.put(f"/api/uploads/{upload_id}", content=b"replacement"))
            try:
                assert await asyncio.to_thread(started.wait, 5), "media probe did not start"
                with pytest.raises(app_module.HTTPException) as exc:
                    app_module._take_pending_upload(upload_id, None)
                assert exc.value.status_code == 409
                assert slot["complete"] is False and slot["bytes"] == 0
            finally:
                release.set()
                response = await put
            assert response.status_code == 200
            assert response.json()["bytes"] == 11

    asyncio.run(run())
    assert slot["complete"] is True and slot["bytes"] == 11


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_upload_duration_is_unreadable(dirs, monkeypatch, duration):
    upload_id = _tool_payload(_mcp("create_upload", {}))["upload_id"]
    assert _put_upload(upload_id, b"previous-video").status_code == 200
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: duration)
    response = _put_upload(upload_id, b"replacement")
    assert response.status_code == 400
    assert response.json() == {"detail": "The body is not a readable video file"}
    slot = app_module.pending_uploads[upload_id]
    assert slot["complete"] is False and slot["bytes"] == 0
    assert not os.path.exists(slot["path"])


def test_completed_upload_cannot_be_taken_or_deleted_by_another_owner(dirs, monkeypatch):
    slot = _tool_payload(_mcp("create_upload", {}))
    upload_id = slot["upload_id"]
    response = _put_upload(upload_id, b"fixture-media")
    assert response.status_code == 200
    stored = app_module.pending_uploads[upload_id]
    assert stored["complete"] is True and stored["bytes"] == 13
    stored["user_id"] = "fixture-owner"
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    assert app_module._take_pending_upload(upload_id, "fixture-owner") is stored
    with pytest.raises(app_module.HTTPException) as exc:
        app_module._take_pending_upload(upload_id, "different-owner")
    assert exc.value.status_code == 404
    assert exc.value.detail == "Unknown or expired upload_id"

    async def other_owner(request):
        return "different-owner"

    monkeypatch.setattr(app_module, "_owner_id", other_owner)
    # Direct endpoint invocation avoids unrelated cloud authentication middleware;
    # this checks the existing ownership guard, not live authentication.
    with pytest.raises(app_module.HTTPException) as exc:
        asyncio.run(app_module.delete_upload(upload_id, None))
    assert exc.value.status_code == 404
    assert app_module.pending_uploads[upload_id] is stored
    assert os.path.exists(stored["path"])


def test_real_video_upload_completes_after_media_probe(tmp_path, monkeypatch):
    """Real local FFmpeg input, ASGI upload and ffprobe; no pipeline/provider."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for the real upload seam")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(uploads))
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=32x48:r=5:d=1", "-c:v", "mpeg4", str(source)],
                   check=True, capture_output=True)
    upload_id = _tool_payload(_mcp("create_upload", {}))["upload_id"]
    response = _put_upload(upload_id, source.read_bytes())
    assert response.status_code == 200
    assert response.json()["duration_seconds"] == 1.0
    assert response.json()["bytes"] == source.stat().st_size
    slot = app_module.pending_uploads[upload_id]
    assert slot["complete"] is True and slot["bytes"] == source.stat().st_size
    with open(slot["path"], "rb") as uploaded:
        assert uploaded.read() == source.read_bytes()
