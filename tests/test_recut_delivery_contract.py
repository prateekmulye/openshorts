"""Real local FFmpeg + API edit checks; generated media is a test fixture."""
import asyncio
import json
import shutil

import httpx
import pytest

app = pytest.importorskip('app')
from test_frame_timing import fixture_video


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='ffmpeg unavailable')
@pytest.mark.parametrize('ceiling,expected,initial',[(3.8,200,False),(3.7,500,False),(None,409,False),(3.7,500,True)])
def test_rejected_clip_restored_only_after_measured_contract(tmp_path,monkeypatch,ceiling,expected,initial):
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setattr(app,'OUTPUT_DIR',str(tmp_path))
    monkeypatch.setattr(app,'BILLING_ENABLED',False)
    monkeypatch.setattr(app,'jobs',{})
    jid='measured-recut'; directory=tmp_path/jid;directory.mkdir()
    fixture_video(directory/'V_clip_2.mp4',audio=True)
    shutil.copyfile(directory/'V_clip_2.mp4',directory/'V_clip_1.mp4')
    clips=[{'start':0,'end':4.5,'delivery_validated':i==0 or initial,
            'rendered_source_range':{'start':0,'end':4.5,'expected_frames':108}} for i in range(2)]
    data={'shorts':clips,'output_format':'original','transcript':{},
          'output_contract':{'requested_min':2,'requested_max':2,'selected_count':2}}
    if ceiling is not None:
        data['delivery_contract']={'min_seconds':1,'max_seconds':ceiling,
                                   'output_format':'original','source_aspect':16/9}
    metadata=directory/'V_metadata.json';metadata.write_text(json.dumps(data))
    app.jobs[jid]={'status':'completed','output_dir':str(directory),'user_id':None,
                   'logs':[],'result':app._result_from_disk(jid,str(directory))}
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app),base_url='http://test') as client:
            return await client.post('/api/clip/rerender',json={'job_id':jid,'clip_index':1,
                'segments':[{'start':0.13,'end':3.93}],'snap_to_words':False,'reapply_captions':False})
    response=asyncio.run(request())
    assert response.status_code == expected,response.text
    saved=json.loads(metadata.read_text())
    if expected == 200:
        assert saved['shorts'][1]['delivery_validated'] is True
        assert saved['shorts'][1]['rendered_source_range'] is None
        assert saved['shorts'][1]['delivery_probe']['duration'] == pytest.approx(3.75375,abs=0.000001)
        assert saved['shorts'][0].get('video_url') is None
        result=app.jobs[jid]['result']
        assert [c['clip_index'] for c in result['clips']] == [0,1]
        assert saved['output_contract']['delivered_count'] == 2
        assert result['output_contract']['delivered_count'] == 2
        assert result['output_contract']['status'] == 'complete'
    else:
        assert saved['shorts'][1]['delivery_validated'] is initial
        assert len(app.jobs[jid]['result']['clips']) == (2 if initial else 1)
        assert saved['shorts'][1].get('recipe') is None

        app.jobs.clear()
        app._recover_jobs_from_disk()
        restored=app.jobs[jid]['result']['clips']
        assert len(restored) == (2 if initial else 1)
        if initial:
            assert restored[1]['video_url'].endswith('/V_clip_2.mp4')
            assert len(list((directory/'rejected').glob('*.mp4'))) == 1


def test_failed_delivery_retains_all_unique_render_twins_outside_discovery(tmp_path):
    # Real media, including local stand-ins for the names a cloud watermark
    # would produce; no claim that a live cloud watermark/storage ran.
    clean='recut_unique_V_clip_2.mp4'
    fixture_video(tmp_path/clean,audio=True)
    for name in ('subtitled_'+clean,'wm_subtitled_'+clean):
        shutil.copyfile(tmp_path/clean,tmp_path/name)
    data={'delivery_contract':{'min_seconds':1,'max_seconds':3,
                              'output_format':'original','source_aspect':16/9}}
    with pytest.raises(ValueError,match='OUTPUT_CONTRACT:duration'):
        app._validate_recut_delivery(data,str(tmp_path),'wm_subtitled_'+clean,clean)
    assert list(tmp_path.glob('*.mp4')) == []
    assert len(list((tmp_path/'rejected').glob('*.mp4'))) == 3


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='ffmpeg unavailable')
@pytest.mark.parametrize('case',['duration','aspect','valid'])
def test_manual_reframe_measures_delivery_before_replacing_current_clip(tmp_path,monkeypatch,case):
    """Real API -> source cut -> renderer -> ffprobe; no render/result stub.

    A short generated source and the supported deterministic scene detector
    keep this guard regression local. It does not test learned scene quality.
    """
    import subprocess
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setenv('SCENE_ENGINE','pyscenedetect')
    monkeypatch.setattr(app,'OUTPUT_DIR',str(tmp_path))
    monkeypatch.setattr(app,'BILLING_ENABLED',False)
    monkeypatch.setattr(app,'jobs',{})
    jid='measured-manual-reframe'; directory=tmp_path/jid;directory.mkdir()
    fixture_video(directory/'source.mp4',audio=True)
    # The existing file is actual media with a distinct, shorter timeline.
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(directory/'source.mp4'),
        '-t','1.001','-vf','crop=54:54','-c:v','libx264','-c:a','aac',
        str(directory/'V_clip_1.mp4')],check=True)
    old_url=f'/videos/{jid}/V_clip_1.mp4'
    clip={'start':0,'end':1.3,'video_url':old_url,'delivery_validated':case!='valid',
          'recipe':{'v':1,'segments':[{'start':0,'end':1.3}],
                    'canonical_range':{'start':0,'end':1.3},'framing':'full'}}
    data={'shorts':[clip],'source_video':'source.mp4','output_format':'square',
          'transcript':{},'output_contract':{'requested_min':1,'requested_max':1,'selected_count':1},
          'delivery_contract':{'min_seconds':1,'max_seconds':1.1 if case=='duration' else 2,
              # The aspect case intentionally supplies a contradictory render
              # setting: measured pixels, not the chosen setting, must win.
              'output_format':'vertical' if case=='aspect' else 'square','source_aspect':16/9}}
    metadata=directory/'V_metadata.json';metadata.write_text(json.dumps(data))
    before=metadata.read_bytes()
    app.jobs[jid]={'status':'completed','output_dir':str(directory),'user_id':None,
                   'logs':[],'result':{'clips':[dict(clip,clip_index=0)]}}
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app),base_url='http://test') as client:
            return await client.post('/api/clip/reframe',json={'job_id':jid,'clip_index':0,
                'crop_overrides':{'0':0.5},'reapply_captions':False})
    response=asyncio.run(request())
    if case=='valid':
        assert response.status_code == 200,response.text
        saved=json.loads(metadata.read_text())['shorts'][0]
        assert saved['delivery_validated'] is True
        assert saved['delivery_probe']['duration'] == pytest.approx(1.292958,abs=0.000001)
        assert saved['delivery_probe']['display_aspect'] == 1.0
        assert saved['video_url'] != old_url
        assert app.jobs[jid]['result']['output_contract']['delivered_count'] == 1
    else:
        assert response.status_code == 500,response.text
        assert f'OUTPUT_CONTRACT:{case}' in response.json()['detail']
        assert metadata.read_bytes() == before
        assert app.jobs[jid]['result']['clips'][0]['video_url'] == old_url
        rejected=list((directory/'rejected').glob('*.mp4'))
        assert len(rejected) == 1
        assert not list(directory.glob('recut_*.mp4'))
        app.jobs.clear()
        app._recover_jobs_from_disk()
        assert app.jobs[jid]['result']['clips'][0]['video_url'] == old_url
