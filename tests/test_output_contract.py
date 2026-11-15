"""Contract regression fixtures; provider responses are fixtures, never live evidence.

Load pipeline functions without ML imports so these checks also run on thin CI.
Real media rendering is exercised separately against ffmpeg.
"""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import clip_selection as selection
import output_contract


def pipeline_functions(*names):
    tree = ast.parse(Path('main.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = vars(selection).copy()
    ns.update(os=os, json=json, output_contract=output_contract, llm_backend=SimpleNamespace(active=lambda: True, vision_model_name=lambda: '', model_name=lambda: 'fixture', base_url=lambda: 'fixture'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main.py', 'exec'), ns)
    return ns


def run_selection(monkeypatch, responses):
    monkeypatch.setenv('CLIP_TARGET_MIN', '2')
    monkeypatch.setenv('CLIP_TARGET_MAX', '2')
    monkeypatch.setenv('CLIP_MIN_SECONDS', '8')
    monkeypatch.setenv('CLIP_MAX_SECONDS', '18')
    ns = pipeline_functions('get_viral_clips')
    prompts = []
    def stage(client, model, windows, prompt, schema, key, costs, label):
        if key == 'windows':
            return [{'id': w['id'], 'score': 80} for w in windows]
        prompts.append(prompt(windows))
        return responses.pop(0)
    ns.update(score_batch_size=lambda: 3, _run_stage_split=stage,
              gemini_worker=SimpleNamespace(ScoreResponse=object(), DetailResponse=object(),
                  GeminiBlockedError=type('Blocked', (Exception,), {}),
                  SCORE_PROMPT_TEMPLATE='{windows_json}', DETAIL_PROMPT_TEMPLATE='{windows_json}'))
    out = ns['get_viral_clips']({'language':'en', 'segments':[{'start':0, 'end':40, 'text':'A real transcript fixture', 'words':[]}]}, 40)
    return out, prompts


def clip(start, end):
    return {'start':start, 'end':end, 'source_window_id':'window_001', 'predicted_score':80}


def test_invalid_duration_fallback_is_not_delivered(monkeypatch):
    out, prompts = run_selection(monkeypatch, [[clip(0, 2), clip(5, 28)], []])
    assert out is not None
    assert out['shorts'] == []
    assert out['output_contract']['status'] == 'no_suitable_moments'
    assert len(prompts) == 2


def test_one_window_gets_one_bounded_refill_excluding_accepted_time(monkeypatch):
    out, prompts = run_selection(monkeypatch, [[clip(0,10)], [clip(20,30)]])
    assert len(prompts) == 2
    assert 'EXCLUDED_INTERVALS_JSON' in prompts[1]
    assert len(out['shorts']) == 2
    assert out['output_contract']['selected_count'] == 2


def test_underfill_is_explicit_and_never_padded(monkeypatch):
    out, prompts = run_selection(monkeypatch, [[clip(0,10)], []])
    assert len(prompts) == 2
    assert out['shorts'] == [dict(clip(0,10), proposed=[0,10])]
    assert out['output_contract']['status'] == 'partial'
    assert out['output_contract']['shortfall_count'] == 1


def test_silent_missing_vision_is_capability_error(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    ns = pipeline_functions('get_visual_clips')
    with pytest.raises(RuntimeError, match='CAPABILITY_UNAVAILABLE:visual_clip_selection'):
        ns['get_visual_clips']('not-read.mp4', 40)


def test_original_uses_native_renderer():
    ns = pipeline_functions('_render_clip')
    calls = []
    ns.update(finalize_clip_passthrough=lambda *a: calls.append(a) or True,
              ASPECT_RATIO=9/16, process_video_to_vertical=lambda *a, **k: False)
    assert ns['_render_clip']('in.mp4', 'out.mp4', 'original', None, None, False)
    assert calls == [('in.mp4', 'out.mp4')]


@pytest.mark.parametrize('start,end', [(0,2), (0,23), (-1,10), (35,45), (float('nan'),12), (0,float('inf'))])
def test_bounds_reject_invalid_values(start,end):
    assert not selection.valid_clip_bounds(start,end,40,8,18)


def test_bounds_allow_exact_limits():
    assert selection.valid_clip_bounds(0,8,40,8,18)
    assert selection.valid_clip_bounds(22,40,40,8,18)


def test_recut_snap_validates_no_words():
    import recut
    with pytest.raises(recut.RecutError):
        recut.snap_segments([{'start':39,'end':45}], {}, 40)


def test_probe_display_aspect_uses_sar_and_rotation():
    import output_contract
    assert output_contract.display_aspect({'width':720,'height':480,'sample_aspect_ratio':'4:3'}) == 2.0
    assert output_contract.display_aspect({'width':720,'height':480,'sample_aspect_ratio':'4:3', 'side_data_list':[{'rotation':90}]}) == 0.5


def test_final_gate_rejects_duration_and_aspect(monkeypatch):
    import output_contract
    monkeypatch.setattr(output_contract, 'probe', lambda path: {'duration':2.0,'display_aspect':16/9})
    with pytest.raises(ValueError, match='duration'):
        output_contract.validate_delivery('fixture.mp4',8,18,'original',16/9)
    monkeypatch.setattr(output_contract, 'probe', lambda path: {'duration':10.0,'display_aspect':9/16})
    with pytest.raises(ValueError, match='aspect'):
        output_contract.validate_delivery('fixture.mp4',8,18,'original',16/9)


def test_completion_counts_actual_delivery():
    import output_contract
    result = output_contract.outcome(2,2,2,1)
    assert result['status'] == 'partial'
    assert result['delivered_count'] == 1
    assert result['shortfall_count'] == 1
    assert output_contract.outcome(2,2,2,2)['status'] == 'complete'


def test_deduplication_happens_before_refill(monkeypatch):
    out, prompts = run_selection(monkeypatch, [[clip(0,10), clip(0,10)], [clip(20,30)]])
    assert len(prompts) == 2
    assert [(s['start'],s['end']) for s in out['shorts']] == [(0.0,10.0),(20.0,30.0)]


def test_refill_cannot_repeat_accepted_time(monkeypatch):
    out, prompts = run_selection(monkeypatch, [[clip(0,10)], [clip(0,10)]])
    assert len(prompts) == 2
    assert len(out['shorts']) == 1
    assert out['output_contract']['status'] == 'partial'


@pytest.mark.parametrize('duration', [6.96, 18.01])
def test_measured_duration_has_no_implicit_tolerance(monkeypatch, duration):
    monkeypatch.setattr(output_contract, 'probe', lambda path: {'duration':duration,'display_aspect':16/9})
    with pytest.raises(ValueError, match='duration'):
        output_contract.validate_delivery('fixture.mp4',7,18,'original',16/9)


def test_original_rejects_small_display_aspect_drift(monkeypatch):
    monkeypatch.setattr(output_contract, 'probe', lambda path: {'duration':10,'display_aspect':1.781})
    with pytest.raises(ValueError, match='aspect'):
        output_contract.validate_delivery('fixture.mp4',7,18,'original',16/9)
