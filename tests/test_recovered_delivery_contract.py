"""Restart fixtures mirror rejected artifacts remaining on disk.

The status/result functions are real; fixture bytes model storage presence,
not playable video or successful rendering. No cloud service is contacted.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

app = pytest.importorskip('app')


@pytest.mark.parametrize('entry', ['disk', 'cached_cloud_restore', 'resume'])
@pytest.mark.parametrize('accepted', [1, 0])
def test_recovery_status_never_resurrects_rejected_clips(tmp_path, monkeypatch, entry, accepted):
    monkeypatch.setattr(app,'OUTPUT_DIR',str(tmp_path))
    monkeypatch.setattr(app,'jobs',{})
    monkeypatch.setattr(app,'BILLING_ENABLED',False)
    job_id='recovery-contract'
    directory=tmp_path/job_id; directory.mkdir()
    clips=[{'start':0,'end':10,'delivery_validated':i < accepted} for i in range(2)]
    for i in range(2):
        (directory/f'V_clip_{i+1}.mp4').write_bytes(b'storage-presence-fixture')
    (directory/'V_metadata.json').write_text(json.dumps({
        'shorts':clips,'output_contract':{'requested_min':2,'requested_max':2,
                                        'selected_count':2,'delivered_count':2,'status':'complete'}}))
    if entry == 'disk':
        app._recover_jobs_from_disk()
    elif entry == 'cached_cloud_restore':
        # Exercise cloud restore's local-cache registration without mocking a
        # result or bypassing an ownership/URL guard.
        project=SimpleNamespace(metadata_r2_key='V_metadata.json',state={'clips':[]})
        asyncio.run(app._restore_job_files(job_id,project,None))
    else:
        job={'output_dir':str(directory),'user_id':None,'logs':app._TimedLog([])}
        app.jobs[job_id]=job
        app._resume_delivery_record(job_id,job,{'outcome':'completed'})
    status=asyncio.run(app.get_status(job_id,None))
    if accepted:
        assert app.jobs[job_id]['status'] == 'completed'
        assert len(status['result']['clips']) == 1
        assert status['result']['clips'][0]['delivery_validated'] is True
        assert status['result']['output_contract']['delivered_count'] == 1
        assert status['result']['output_contract']['status'] == 'partial'
    else:
        assert app.jobs[job_id]['status'] == 'failed'
        assert status['result'] is None


def test_filtered_results_keep_original_metadata_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(app,'jobs',{})
    clips=[{'delivery_validated':False},{'delivery_validated':True,'start':447.3,'end':477.22}]
    (tmp_path/'V_clip_2.mp4').write_bytes(b'storage-presence-fixture')
    kept,missing=app._clips_actually_rendered('identity',str(tmp_path),'V',clips)
    assert missing == 1
    assert kept[0]['clip_index'] == 1
    job={'result':{'clips':kept}}
    assert app._result_clip(job,1)['start'] == 447.3
    assert app._result_clip(job,0) is None


def test_partial_results_do_not_advertise_rejected_ready_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(app,'jobs',{'identity':{'ready_files':{0:'V_clip_1.mp4',1:'V_clip_2.mp4'}}})
    (tmp_path/'V_metadata.json').write_text(json.dumps({'shorts':[
        {'delivery_validated':False},{'delivery_validated':True}]}))
    for i in (1,2):
        (tmp_path/f'V_clip_{i}.mp4').write_bytes(b'storage-presence-fixture')
    app._refresh_partial_result('identity',str(tmp_path))
    clips=app.jobs['identity']['result']['clips']
    assert len(clips) == 1
    assert clips[0]['clip_index'] == 1


def test_webhook_preserves_original_clip_identity(monkeypatch):
    monkeypatch.setattr(app,'BILLING_ENABLED',False)
    entries=asyncio.run(app._webhook_clip_entries('job',{'result':{'clips':[
        {'clip_index':1,'video_url':'/videos/job/V_clip_2.mp4'}]}}))
    assert entries[0]['index'] == 1
