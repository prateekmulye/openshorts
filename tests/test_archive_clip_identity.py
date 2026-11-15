"""Archive identity with fake external storage/database; no live cloud proof."""
import asyncio
import json
from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest
videos=pytest.importorskip('cloud.videos')


def test_archive_preserves_original_clip_index_and_canonical_twin(tmp_path,monkeypatch):
    uploaded=[]; rows=[]
    monkeypatch.setattr(videos,'settings',SimpleNamespace(r2_configured=True))
    monkeypatch.setattr(videos.storage,'upload_file',lambda path,key,*args: uploaded.append(path))
    class Session:
        def begin(self): return session()
        async def execute(self,statement):
            return SimpleNamespace(scalars=lambda:[],scalar_one_or_none=lambda:None)
        def add(self,row): rows.append(row)
    @asynccontextmanager
    async def session(): yield Session()
    monkeypatch.setattr(videos.database,'session',session)
    (tmp_path/'V_metadata.json').write_text(json.dumps({'shorts':[{},{}]}))
    for name in ('V_clip_1.mp4','V_clip_2.mp4','subtitled_V_clip_2.mp4'):
        (tmp_path/name).write_bytes(b'external-storage-fixture')
    asyncio.run(videos.archive_job('owner','job',[{'clip_index':1,
        'video_url':'/videos/job/subtitled_V_clip_2.mp4'}],str(tmp_path)))
    assert str(tmp_path/'V_clip_2.mp4') in uploaded
    assert str(tmp_path/'V_clip_1.mp4') not in uploaded
    video=next(row for row in rows if isinstance(row,videos.UserVideo))
    project=next(row for row in rows if isinstance(row,videos.Project))
    assert video.clip_index == 1
    assert project.state['clips'][0]['index'] == 1
