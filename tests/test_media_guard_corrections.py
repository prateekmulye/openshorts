"""Missing guards beyond current-main job-file validation; external services are faked."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import editor
import gpu_decode
import hooks

BAD_FILTERS = [
    "eq=/contrast=fixture", "hue=/h=fixture", "scale=/w=fixture",
    "eq='/contrast'=fixture", "eq='' /contrast=fixture",
    "eq=/''contrast=fixture", "eq=''/contrast=fixture",
    "eq=/contrast='fixture'", "eq=contrast=1:/brightness=fixture",
    "eq=contrast=1:''/brightness=fixture", "eq='contrast=1:/brightness=fixture'",
]


@pytest.fixture
def editor_calls(monkeypatch):
    """Mock external FFmpeg/probe and model responses, never the guard methods."""
    calls = []
    state = {"fail_first": False, "repair": "eq=contrast=1.1"}
    instance = editor.VideoEditor.__new__(editor.VideoEditor)
    instance.model_name = "test-fixture-model"
    instance.client = SimpleNamespace(models=SimpleNamespace(generate_content=
        lambda **kw: SimpleNamespace(text=json.dumps({"filter_string": state["repair"]}))))
    monkeypatch.setattr(editor.subprocess, "check_output", lambda *a, **kw: b"32x48")

    def run(cmd, **kw):
        args = [os.fsdecode(a) for a in cmd]
        calls.append((args, kw))
        failed = state["fail_first"] and len(calls) == 1
        return subprocess.CompletedProcess(cmd, 1 if failed else 0, b"", b"bad syntax" if failed else b"")

    monkeypatch.setattr(editor.subprocess, "run", run)
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    return instance, calls, state


@pytest.mark.parametrize("bad", BAD_FILTERS)
def test_original_filter_is_refused_at_execution_boundary(editor_calls, bad):
    instance, calls, _ = editor_calls
    instance.apply_edits("input.mp4", "output.mp4", {"filter_string": bad})
    assert len(calls) == 1
    args, _ = calls[0]
    assert args[:6] == ["ffmpeg", "-y", "-i", "input.mp4", "-c", "copy"]
    assert "-vf" not in args and "-filter_complex" not in args


@pytest.mark.parametrize("bad", BAD_FILTERS)
def test_repaired_filter_is_refused_before_second_dry_run(editor_calls, bad):
    instance, calls, state = editor_calls
    state.update(fail_first=True, repair=bad)
    with pytest.raises(RuntimeError, match="failed validation even after self-repair"):
        instance.apply_edits("input.mp4", "output.mp4", {"filter_string": "eq=contrast=1"})
    assert len(calls) == 1
    assert calls[0][0][-3:] == ["-f", "null", "-"]
    assert calls[0][0][calls[0][0].index("-vf") + 1] == "eq=contrast=1,setsar=1"


@pytest.mark.parametrize("filter_string", [
    "eq=contrast=1/2", "eq=contrast='1/2'", "scale=w=iw/2:h=ih/2",
])
def test_arithmetic_division_is_still_an_edit(editor_calls, filter_string):
    instance, calls, _ = editor_calls
    instance.apply_edits("input.mp4", "output.mp4", {"filter_string": filter_string})
    assert len(calls) == 2
    assert calls[-1][0][calls[-1][0].index("-vf") + 1] == filter_string + ",setsar=1"


def _recipe(tmp_path, values):
    (tmp_path / "source.mp4").touch()
    recipe, png = hooks.recipe_paths(str(tmp_path), "hooked.mp4")
    Path(png).touch()
    Path(recipe).write_text(json.dumps({"source": "source.mp4", **values}))


@pytest.mark.parametrize("values", [
    {"x": "0:0,drawtext=textfile=sentinel.txt", "y": 0},
    {"x": 0, "y": "ih[v];movie=sentinel.txt"},
    {"x": 0, "y": 0, "duration": "5)',drawtext=text='x"},
    {"x": 20001, "y": 0}, {"x": -20001, "y": 0},
    {"x": 0, "y": 20001}, {"x": 0, "y": -20001},
    {"x": None, "y": 0}, {"y": 0},
    # Upstream int(infinity) raised OverflowError; malformed recipes must return False.
    {"x": float("inf"), "y": 0}, {"x": 0, "y": float("-inf")},
    {"x": float("nan"), "y": 0},
    {"x": 0, "y": 0, "duration": float("nan")},
    {"x": 0, "y": 0, "duration": float("inf")},
    {"x": 0, "y": 0, "duration": -1},
    {"x": 0, "y": 0, "duration": 86400},
    # Falsey values are not absence. Numeric zero is the caller's legacy
    # whole-clip convention, covered separately below; other values must parse.
    {"x": 0, "y": 0, "duration": []},
    {"x": 0, "y": 0, "duration": {}},
    {"x": 0, "y": 0, "duration": ""},
    {"x": 0, "y": 0, "duration": False},
    {"x": 0, "y": 0, "duration": True},
    {"x": 0, "y": 0, "duration": "0"},
])
def test_invalid_hook_recipe_never_encodes(tmp_path, monkeypatch, values):
    _recipe(tmp_path, values)
    monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **kw: pytest.fail("invalid recipe reached a process"))
    assert hooks.materialize(str(tmp_path), "hooked.mp4") is False
    assert not (tmp_path / "hooked.mp4").exists()
    assert not list(tmp_path.glob(".part_*"))


@pytest.mark.parametrize("duration, graph", [
    ({}, "[0:v][1:v]overlay=0:0"),
    ({"duration": None}, "[0:v][1:v]overlay=0:0"),
    # AUTO_HOOK_SECONDS=0 and _overlay_graph already mean the whole clip.
    # Preserve that numeric-zero behavior instead of rejecting old recipes.
    ({"duration": 0}, "[0:v][1:v]overlay=0:0"),
    ({"duration": 0.0}, "[0:v][1:v]overlay=0:0"),
    ({"duration": 0.5}, "[0:v][1:v]overlay=0:0:enable='between(t,0,0.5)'"),
    ({"duration": "0.5"}, "[0:v][1:v]overlay=0:0:enable='between(t,0,0.5)'"),
])
def test_optional_hook_duration_keeps_its_window(tmp_path, monkeypatch, duration, graph):
    """Mock FFmpeg only; inspect the actual command from the stored recipe."""
    _recipe(tmp_path, {"x": 0, "y": 0, **duration})
    monkeypatch.setattr(gpu_decode, "ENABLED", False)
    # Rendering uses RENDER, not ENABLED; avoid counting a GPU probe as an encode.
    monkeypatch.setattr(gpu_decode, "RENDER", False)
    calls = []

    def encode(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"fixture-encoded-output")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(hooks.subprocess, "run", encode)
    assert hooks.materialize(str(tmp_path), "hooked.mp4") is True
    assert len(calls) == 1
    assert calls[0][calls[0].index("-filter_complex") + 1] == graph
    assert (tmp_path / "hooked.mp4").read_bytes() == b"fixture-encoded-output"
    assert not list(tmp_path.glob(".part_*"))


@pytest.mark.parametrize("stage", ["division", "hook"])
def test_real_media_seam(tmp_path, monkeypatch, stage):
    """Synthetic local media through real FFmpeg; no model/provider calls."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe required for real media seams")
    from PIL import Image
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    monkeypatch.setattr(gpu_decode, "ENABLED", False)
    monkeypatch.setattr(gpu_decode, "RENDER", False)
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=32x48:r=5:d=1", "-c:v", "mpeg4", str(source)],
                   check=True, capture_output=True)
    output = tmp_path / ("hooked.mp4" if stage == "hook" else "output.mp4")
    if stage == "division":
        instance = editor.VideoEditor.__new__(editor.VideoEditor)
        instance.client = None
        instance.apply_edits(str(source), str(output), {"filter_string": "eq=contrast=1/2"})
    else:
        _recipe(tmp_path, {"x": "0", "y": "0", "duration": "0.5"})
        _, png = hooks.recipe_paths(str(tmp_path), "hooked.mp4")
        Image.new("RGBA", (8, 8), "red").save(png)
        assert hooks.materialize(str(tmp_path), "hooked.mp4")
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                            check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["nb_frames"]) == (32, 48, "5")
    assert not list(tmp_path.glob(".part_*"))
