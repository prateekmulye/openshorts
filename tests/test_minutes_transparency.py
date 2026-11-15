"""Minutes users can see and steer before a job runs.

- The cost line (/api/process/estimate + ``estimate_outcome``) says what a
  source will cost before Generate.
- A stretch picked in the dashboard (``source_start`` / ``source_end``) is
  billed alone and main.py cuts the source to it.
- Minutes are rounded down (``config.billable_minutes``), and editing an
  existing clip is free, bounded by an hourly cap instead.
- Posting follows the upload to its real outcome and refuses accounts whose
  login expired; YouTube titles are cut to 100 characters.
"""
import asyncio
import json
import os
import shutil
import subprocess

import httpx
import pytest
from fastapi import HTTPException

app_module = pytest.importorskip("app")
from cloud import config  # noqa: E402


class TestBillableMinutes:
    def test_rounds_down_with_a_floor_of_one(self):
        assert config.billable_minutes(4.6) == 4
        assert config.billable_minutes(20.0) == 20
        assert config.billable_minutes(0.3) == 1
        assert config.billable_minutes("lots") == 1

    def test_editor_actions_are_free(self):
        assert config.EDITOR_ACTION_MINUTES == 0


class TestPlanSourceRange:
    def test_no_range_or_the_whole_video_is_an_ordinary_job(self):
        assert app_module.plan_source_range(45, None) is None
        assert app_module.plan_source_range(45, (0, None)) is None
        assert app_module.plan_source_range(45, (None, 45 * 60)) is None
        # Past the end is clamped to it: still the whole video.
        assert app_module.plan_source_range(45, (0, 99999)) is None

    def test_a_stretch_is_kept_in_seconds(self):
        assert app_module.plan_source_range(45, (120, 420)) == {
            "start_seconds": 120.0, "end_seconds": 420.0}
        assert app_module.plan_source_range(45, ("600", None)) == {
            "start_seconds": 600.0, "end_seconds": 2700.0}

    @pytest.mark.parametrize("rng", [(10, 40), (500, 100), ("a", 100),
                                      ("nan", 100), (0, "nan"),
                                      ("-inf", 100), (0, "inf")])
    def test_too_short_backwards_or_garbage_is_a_400(self, rng):
        with pytest.raises(HTTPException) as e:
            app_module.plan_source_range(45, rng)
        assert e.value.status_code == 400


class TestEstimateOutcome:
    def test_covered_by_the_balance(self):
        out = app_module.estimate_outcome("starter", 57.4, 100, True)
        assert out["outcome"] == "whole" and out["cost_minutes"] == 57
        # 20.7 minutes cost 20: fits a 20-minute balance whole.
        assert app_module.estimate_outcome("free", 20.7, 20, True)["outcome"] == "whole"

    def test_free_first_video_is_clipped_whole(self):
        out = app_module.estimate_outcome("free", 45.5, 20, False)
        assert out["outcome"] == "first_video_free" and out["cost_minutes"] == 20

    def test_free_later_video_is_clipped_to_the_balance(self):
        out = app_module.estimate_outcome("free", 50, 20, True)
        assert out["outcome"] == "first_part" and out["processed_minutes"] == 20

    def test_paid_plan_past_the_balance_meets_the_wall(self):
        out = app_module.estimate_outcome("starter", 120, 30, True)
        assert out["outcome"] == "needs_more" and out["partial_minutes"] == 30


class TestApplySourceCut:
    def test_picked_stretch_is_cut_to_its_exact_length(self):
        env = {"SOURCE_START_SECONDS": "stale"}
        app_module._apply_source_cut(env, {
            "processed_minutes": 5, "total_minutes": 45,
            "start_seconds": 130.0, "end_seconds": 460.0, "chosen": True})
        assert env["MAX_SOURCE_MINUTES"] == "5.5000"
        assert env["SOURCE_START_SECONDS"] == "130.000"

    def test_quota_wall_cut_is_unchanged(self):
        env = {"SOURCE_START_SECONDS": "9"}
        app_module._apply_source_cut(env, {"processed_minutes": 20, "total_minutes": 45})
        assert env == {"MAX_SOURCE_MINUTES": "20"}

    def test_no_partial_clears_both(self):
        env = {"MAX_SOURCE_MINUTES": "3", "SOURCE_START_SECONDS": "9"}
        app_module._apply_source_cut(env, None)
        assert env == {}


# --------------------------------------------------------------------------- #
# Endpoint: the picked stretch reaches the reservation and the job
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())

    async def _probe(url):
        return {"max_height": 1080, "duration": 45 * 60}
    monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)
    return out_root, up_root


def test_source_range_reaches_the_reservation_and_the_job(public_youtube_dns, dirs, monkeypatch):
    out_root, _ = dirs
    partial = {"processed_minutes": 5, "total_minutes": 45, "start_seconds": 130.0,
               "end_seconds": 460.0, "chosen": True}
    seen = {}

    async def _reserve(request, url, input_path, job_id, max_minutes=None, source_range=None):
        seen["source_range"] = source_range
        return 7, 1, "res-1", "starter", partial
    monkeypatch.setattr(app_module, "reserve_process_minutes", _reserve)

    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process", json={
                "url": "https://www.youtube.com/watch?v=long", "acknowledged": True,
                "source_start": 130, "source_end": 460}, headers={"X-Gemini-Key": "k"})
    resp = asyncio.run(_do())
    assert resp.status_code == 200, resp.text
    assert seen["source_range"] == (130, 460)
    job = app_module.jobs[resp.json()["job_id"]]
    assert job["env"]["SOURCE_START_SECONDS"] == "130.000"
    assert job["env"]["MAX_SOURCE_MINUTES"] == "5.5000"
    manifest = json.load(open(os.path.join(out_root, resp.json()["job_id"], app_module._RESUME_FILE)))
    assert manifest["partial"] == partial


@pytest.fixture()
def selfhost_source(dirs, monkeypatch):
    """Only external source lookup is faked; reservation and enqueue are real."""
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)

    async def _validate(url):
        pass  # Public-URL DNS lookup is covered by the SSRF tests.
    monkeypatch.setattr(app_module, "_validate_source_url", _validate)
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 2700.0)
    return dirs


def _submit_selfhost_range(fields, upload=False):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            if upload:
                return await client.post("/api/process", data={"acknowledged": "true", **fields},
                                         files={"file": ("source.mp4", b"test fixture")},
                                         headers={"X-Gemini-Key": "test-key"})
            return await client.post("/api/process", json={
                "url": "https://www.youtube.com/watch?v=long", "acknowledged": True,
                **fields}, headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


@pytest.mark.parametrize("upload", [False, True], ids=["url", "upload"])
@pytest.mark.parametrize("fields,start,end,length,minutes", [
    ({"source_start": 900, "source_end": 1800}, "900.000", 1800.0, "15.0000", 15),
    ({"source_start": 900}, "900.000", 2700.0, "30.0000", 30),
    ({"source_start": 130, "source_end": 460}, "130.000", 460.0, "5.5000", 5.5),
    ({"source_start": 900, "source_end": 960}, "900.000", 960.0, "1.0000", 1),
])
def test_selfhost_range_enqueues_and_resumes_exact_cut(
        selfhost_source, monkeypatch, upload, fields, start, end, length, minutes):
    out_root, _ = selfhost_source
    response = _submit_selfhost_range(fields, upload)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    job = app_module.jobs[job_id]
    assert job["env"]["SOURCE_START_SECONDS"] == start
    assert job["env"]["MAX_SOURCE_MINUTES"] == length
    assert job["reservation_id"] is None and job["user_id"] is None
    partial = {"processed_minutes": minutes, "total_minutes": 45.0,
               "start_seconds": float(start), "end_seconds": end, "chosen": True}
    assert job["partial"] == partial
    manifest = json.loads((out_root / job_id / app_module._RESUME_FILE).read_text())
    assert manifest["partial"] == partial
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    app_module._resume_interrupted_jobs()
    resumed = app_module.jobs[job_id]
    assert resumed["env"]["SOURCE_START_SECONDS"] == start
    assert resumed["env"]["MAX_SOURCE_MINUTES"] == length


@pytest.mark.parametrize("fields", [{}, {"source_start": 0}, {"source_end": 2700}])
def test_selfhost_whole_source_has_no_cut(selfhost_source, fields):
    response = _submit_selfhost_range(fields)
    assert response.status_code == 200, response.text
    job = app_module.jobs[response.json()["job_id"]]
    assert job["partial"] is None
    assert "SOURCE_START_SECONDS" not in job["env"]
    assert "MAX_SOURCE_MINUTES" not in job["env"]


def test_selfhost_without_range_does_not_probe_at_reservation(monkeypatch):
    async def _unexpected(url):
        pytest.fail("No-range reservation must not probe the source")
    monkeypatch.setattr(app_module, "_probe_youtube_quality", _unexpected)
    assert asyncio.run(app_module.reserve_process_minutes(None, "unused", None, "job")) == (
        None, 2, None, None, None)


@pytest.mark.parametrize("start,end", [
    (900, 800), (900, 959), (2700, None), ("nan", 1800),
    (900, "nan"), ("-inf", 1800), (900, "inf"), ("garbage", 1800),
    (True, 1800), (False, 1800), (900, True), ([], 1800), ({}, 1800),
])
def test_selfhost_invalid_range_is_rejected_before_enqueue(selfhost_source, start, end):
    response = _submit_selfhost_range({"source_start": start, "source_end": end})
    assert response.status_code == 400, response.text
    assert app_module.jobs == {}
    assert list(selfhost_source[0].iterdir()) == []


@pytest.mark.parametrize("duration", [0, -1, None, True, "bad", float("nan"), float("inf")])
def test_selfhost_unknown_duration_cannot_accept_a_range(selfhost_source, monkeypatch, duration):
    async def _probe(url):
        return {"duration": duration, "max_height": 1080}
    monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)
    response = _submit_selfhost_range({"source_start": 900, "source_end": 1800})
    assert response.status_code == 400, response.text
    assert app_module.jobs == {}


@pytest.mark.parametrize("upload", [False, True], ids=["url", "upload"])
def test_selfhost_probe_error_never_enqueues_whole_source(selfhost_source, monkeypatch, upload):
    def unavailable(*args):
        raise RuntimeError("fixture probe unavailable")

    async def unavailable_url(url):
        unavailable()

    monkeypatch.setattr(app_module, "_probe_youtube_quality", unavailable_url)
    # Upload preflight runs before reservation; exercise the fail-open probe's
    # real return contract (zero) so reservation must fail closed on duration.
    monkeypatch.setattr(app_module, "_media_duration_seconds", lambda path: 0)
    response = _submit_selfhost_range({"source_start": 900, "source_end": 1800}, upload)
    assert response.status_code == 400, response.text
    assert app_module.jobs == {}
    assert app_module.job_queue.empty()
    assert list(selfhost_source[0].iterdir()) == []
    assert list(selfhost_source[1].iterdir()) == []


def test_selfhost_range_preserves_quality_confirmation(selfhost_source, monkeypatch):
    async def low_resolution(url):
        return {"duration": 2700, "max_height": 360}
    monkeypatch.setattr(app_module, "_probe_youtube_quality", low_resolution)
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 720)
    response = _submit_selfhost_range({"source_start": 900, "source_end": 1800})
    assert response.status_code == 200, response.text
    assert response.json()["needs_confirmation"] is True
    assert "job_id" not in response.json()
    assert app_module.jobs == {}
    assert app_module.job_queue.empty()
    assert list(selfhost_source[0].iterdir()) == []


def test_health_advertises_source_ranges():
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health")
    response = asyncio.run(_do())
    assert response.status_code == 200
    assert response.json()["capabilities"]["source_ranges"] is True


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg and ffprobe")
def test_real_upload_range_cuts_selected_frames_and_resumes_once(dirs, tmp_path, monkeypatch):
    """Synthetic media, real upload/probe/cut/resume; no ML or external service.

    Load the two unchanged cut functions from main.py in isolation: importing
    the whole worker requires optional ML packages and loads model weights.
    This proves the cut implementation, not full-worker/model execution.
    """
    import ast
    import hashlib
    import math
    from pathlib import Path
    import cv2

    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    source = tmp_path / "synthetic-colors.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=red:s=64x64:r=2:d=30",
        "-f", "lavfi", "-i", "color=blue:s=64x64:r=2:d=60",
        "-f", "lavfi", "-i", "color=green:s=64x64:r=2:d=30",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-g", "2", str(source)], check=True)
    assert app_module._media_duration_seconds(str(source)) == 120.0

    async def submit():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                     base_url="http://testserver") as client:
            return await client.post("/api/process", headers={"X-Gemini-Key": "fixture-key"},
                data={"acknowledged": "true", "source_start": "45", "source_end": "105"},
                files={"file": ("synthetic-colors.mp4", source.read_bytes(), "video/mp4")})

    response = asyncio.run(submit())
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    job = app_module.jobs[job_id]
    assert job["partial"] == {"processed_minutes": 1.0, "total_minutes": 2.0,
        "start_seconds": 45.0, "end_seconds": 105.0, "chosen": True}
    worker_source = job["cmd"][job["cmd"].index("-i") + 1]
    main_path = Path(app_module.__file__).with_name("main.py")
    tree = ast.parse(main_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_file_stamp", "cap_source_duration")]
    assert len(functions) == 2
    namespace = {"cv2": cv2, "os": os, "json": json, "math": math, "subprocess": subprocess}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(main_path), "exec"), namespace)
    cap = namespace["cap_source_duration"]
    cap(worker_source, job["env"]["MAX_SOURCE_MINUTES"],
        start_seconds=job["env"]["SOURCE_START_SECONDS"])
    duration = app_module._media_duration_seconds(worker_source)
    assert 59.5 <= duration <= 60.5
    video = cv2.VideoCapture(worker_source)
    try:
        ok, first = video.read()
        assert ok
        video.set(cv2.CAP_PROP_POS_MSEC, 59000)
        ok, last = video.read()
        assert ok
    finally:
        video.release()
    first_bgr, last_bgr = first.mean(axis=(0, 1)), last.mean(axis=(0, 1))
    assert first_bgr[0] > 200 and first_bgr[1] < 10 and first_bgr[2] < 10  # blue, not initial red
    assert last_bgr[1] > 100 and last_bgr[0] < 10 and last_bgr[2] < 10  # final 15 s is green
    cut_hash = hashlib.sha256(Path(worker_source).read_bytes()).hexdigest()
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    app_module._resume_interrupted_jobs()
    resumed = app_module.jobs[job_id]
    cap(worker_source, resumed["env"]["MAX_SOURCE_MINUTES"],
        start_seconds=resumed["env"]["SOURCE_START_SECONDS"])
    assert hashlib.sha256(Path(worker_source).read_bytes()).hexdigest() == cut_hash
    print(f"synthetic cut: source=120s selected=45..105s actual={duration}s "
          f"first_bgr={first_bgr.tolist()} last_bgr={last_bgr.tolist()} resume_sha256={cut_hash}")


def test_estimate_without_billing_says_so():
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process/estimate", json={"seconds": 600})
    resp = asyncio.run(_do())
    assert resp.status_code == 200
    assert resp.json() == {"billing": False}


def test_probe_cache_expires(monkeypatch):
    monkeypatch.setattr(app_module, "_probe_cache", {})
    app_module._store_probe("u1", "https://youtu.be/x", 12.5)
    assert app_module._cached_probe("u1", "https://youtu.be/x") == 12.5
    assert app_module._cached_probe("u2", "https://youtu.be/x") is None
    key = ("u1", "https://youtu.be/x")
    ts, m = app_module._probe_cache[key]
    app_module._probe_cache[key] = (ts - app_module._PROBE_CACHE_TTL - 1, m)
    assert app_module._cached_probe("u1", "https://youtu.be/x") is None


def test_ai_edits_are_free_only_up_to_the_daily_allowance(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "_ai_edit_counts", {})
    owner = {"id": "u1"}

    async def _owner(request):
        return owner["id"]
    monkeypatch.setattr(app_module, "_owner_id", _owner)

    def cost():
        return asyncio.run(app_module._ai_edit_minutes(None))
    assert [cost() for _ in range(config.AI_EDITS_FREE_PER_DAY)] == [0] * config.AI_EDITS_FREE_PER_DAY
    assert cost() == config.MANAGED_ANALYSIS_MINUTES  # Gemini spend is metered again
    owner["id"] = "u2"
    assert cost() == 0  # per user


def test_free_editor_actions_are_capped_per_hour(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "_editor_action_times", {})
    for _ in range(config.EDITOR_ACTIONS_PER_HOUR):
        app_module._check_editor_rate("u1")
    with pytest.raises(HTTPException) as e:
        app_module._check_editor_rate("u1")
    assert e.value.status_code == 429
    app_module._check_editor_rate("u2")  # per user


# --------------------------------------------------------------------------- #
# Posting
# --------------------------------------------------------------------------- #
class TestYoutubeTitle:
    def test_short_title_is_untouched(self):
        assert app_module.youtube_title("  How I  doubled my views ") == "How I doubled my views"

    def test_long_title_is_cut_on_a_word(self):
        long = "word " * 40
        out = app_module.youtube_title(long)
        assert len(out) <= app_module.YOUTUBE_TITLE_MAX
        assert out.endswith("word")


def test_post_status_only_answers_the_account_that_posted(monkeypatch):
    monkeypatch.setattr(app_module, "_post_request_owners", {"req-1": "someone-else"})

    async def _owner(request):
        return "me"
    monkeypatch.setattr(app_module, "_owner_id", _owner)

    async def _get(path):
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path)
    assert asyncio.run(_get("/api/social/post-status/req-1")).status_code == 404
    unknown = asyncio.run(_get("/api/social/post-status/never-seen"))
    assert unknown.status_code == 200 and unknown.json() == {"status": "unknown"}


# --------------------------------------------------------------------------- #
# main.cap_source_duration with a start offset
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg")
def test_cut_from_a_start_offset_happens_once(tmp_path):
    main = pytest.importorskip("main")
    path = str(tmp_path / "src.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "150", "-c:v", "libx264", "-preset", "ultrafast", "-g", "25",
        "-c:a", "aac", "-shortest", path], check=True)

    def seconds():
        return float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path]).decode().strip())

    main.cap_source_duration(path, 1, start_seconds=30)
    assert 59 <= seconds() <= 62
    assert os.path.exists(f"{path}.range")
    # A resumed job runs the same cut again on the already-cut file: no-op.
    main.cap_source_duration(path, 1, start_seconds=30)
    assert 59 <= seconds() <= 62


def test_failed_posts_are_logged_once_per_request(monkeypatch, capsys):
    monkeypatch.setattr(app_module, "_logged_post_failures", set())
    results = [{"platform": "tiktok", "success": False, "message": "Account  needs\nreauth"},
               {"platform": "youtube", "success": True, "message": "Published"}]
    app_module._log_post_failures("req-9", "completed", results)
    app_module._log_post_failures("req-9", "completed", results)  # the next poll
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["[social-post-failed] request=req-9 platform=tiktok status=completed "
                   "message='Account needs reauth'"]
    app_module._log_post_failures("req-ok", "completed", results[1:])
    app_module._log_post_failures("req-x", "failed", [])
    out = capsys.readouterr().out
    assert "req-ok" not in out and "platform=all status=failed" in out


def _saturate_editor_budget(monkeypatch, key):
    import time
    from types import SimpleNamespace
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    now = time.monotonic()
    monkeypatch.setattr(app_module, "_editor_action_times",
                        {key: [now] * config.EDITOR_ACTIONS_PER_HOUR})

    async def _noop(*a, **k):
        return True

    async def _owner(request):
        return "u1"

    async def _user(request):
        return SimpleNamespace(id="u1")
    monkeypatch.setattr(app_module, "require_managed_entitlement", _noop)
    monkeypatch.setattr(app_module, "_ensure_job_files", _noop)
    monkeypatch.setattr(app_module, "_owner_id", _owner)
    monkeypatch.setattr(app_module, "_user_from_request", _user)
    monkeypatch.setitem(app_module.jobs, "job-rl", {"user_id": "u1", "status": "completed"})


def test_scene_listing_is_capped_per_hour(monkeypatch):
    """Listing scenes cuts the clip from source and runs scene detection."""
    from fastapi.testclient import TestClient
    _saturate_editor_budget(monkeypatch, "scenes:u1")
    r = TestClient(app_module.app).get("/api/clip/job-rl/0/scenes")
    assert r.status_code == 429


def test_scene_listing_has_its_own_budget(monkeypatch):
    """Opening the framing editor must not eat the budget of the edits."""
    from fastapi.testclient import TestClient
    _saturate_editor_budget(monkeypatch, "u1")
    r = TestClient(app_module.app).get("/api/clip/job-rl/0/scenes")
    assert r.status_code != 429


def test_caption_removal_is_capped_per_hour(monkeypatch):
    from fastapi.testclient import TestClient
    _saturate_editor_budget(monkeypatch, "u1")
    r = TestClient(app_module.app).post("/api/subtitle/remove",
                                        json={"job_id": "job-rl", "clip_index": 0})
    assert r.status_code == 429
