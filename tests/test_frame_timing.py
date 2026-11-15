"""Real ffmpeg regression checks on generated video/audio test fixtures.

These fixtures test frame/timestamp handling, never model/editorial quality.
"""
import json
import shutil
import subprocess

import pytest
import ffmpeg_utils
import reframe_v2

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
                                reason='ffmpeg/ffprobe unavailable')


def probe(path):
    data = json.loads(subprocess.check_output([
        'ffprobe','-v','error','-show_streams','-of','json',str(path)]))
    return {s['codec_type']:s for s in data['streams']}


def fixture_video(path, *, shifted=False, audio=False):
    cmd = ['ffmpeg','-v','error','-y','-f','lavfi','-i',"nullsrc=size=96x54:rate=24000/1001,geq=lum='16+N*2':cb=128:cr=128"]
    if audio:
        cmd += ['-f','lavfi','-i','aevalsrc=sin(2*PI*(440*t+80*t*t)):s=48000']
    cmd += ['-t','4.5','-c:v','libx264','-preset','ultrafast','-qp','0','-fps_mode','passthrough']
    if shifted:
        cmd += ['-vf','setpts=PTS+1']
    cmd += ['-c:a','aac'] if audio else ['-an']
    subprocess.run(cmd+[str(path)],check=True)


def test_cut_aligns_inward_and_normalized_audio_cannot_extend_it(tmp_path, monkeypatch):
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setenv('AUDIO_NORMALIZE','1')
    monkeypatch.setenv('GPU_DECODE','0')
    source=tmp_path/'source.mp4'; output=tmp_path/'cut.mp4'
    fixture_video(source,audio=True)
    actual_range = ffmpeg_utils.cut_clip(str(source),str(output),0.13,3.93,1)
    assert actual_range == {'start': 0.16683333333333333, 'end': 3.9205833333333335,
                            'expected_frames': 90}
    streams=probe(output); video=streams['video']; audio=streams['audio']
    # Complete source frames [4,94) at 24000/1001: 90 frames, 3.75375s.
    assert int(video['nb_frames']) == 90
    assert float(video['start_time']) == 0.0
    assert float(video['duration']) == pytest.approx(3.75375,abs=0.000001)
    assert abs(float(audio['duration'])-3.75375) <= 0.001
    assert abs(float(audio['start_time'])) <= 0.001

    # The fixture chirp identifies source time independently of mux metadata.
    # loudnorm may change gain, but must not shift the selected audio content.
    import numpy as np
    def pcm(path):
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),
            '-vn','-ac','1','-ar','48000','-f','f32le','-'])
        return np.frombuffer(raw,dtype=np.float32)[::8]
    original=pcm(source); cut=pcm(output)[:2048]
    offsets=np.arange(960,1043)
    scores=[np.dot(original[n:n+len(cut)],cut) /
            np.linalg.norm(original[n:n+len(cut)]) for n in offsets]
    assert abs(int(offsets[np.argmax(scores)])-1001) <= 2


def test_scene_encoding_preserves_fractional_rate_frame_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setattr(reframe_v2.gpu_render,'ENABLED',False)
    monkeypatch.setattr(reframe_v2.gpu_decode,'ENABLED',False)
    source=tmp_path/'source.mp4'; output=tmp_path/'rendered.mp4'
    fixture_video(source,shifted=True)
    count=int(probe(source)['video']['nb_frames']); fps=24000/1001
    boundaries=[0,7,22,39,67,90,count]
    specs=[(a/fps,(b-a)/fps,'[0:v]null[v]',None) for a,b in zip(boundaries,boundaries[1:])]
    reframe_v2.encode_scenes(str(source),specs,96,54,96,54,fps,str(tmp_path),None,str(output))
    video=probe(output)['video']
    assert int(video['nb_frames']) == count
    assert float(video['duration']) == pytest.approx(count/fps,abs=0.000001)
    assert float(video['start_time']) == 0.0

    # Uniform per-frame brightness encodes the frame number. Count alone
    # would miss replacing a dropped scene-boundary frame with a duplicate.
    import numpy as np
    def frame_values(path):
        raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(path),
                                       '-fps_mode','passthrough','-f','rawvideo',
                                       '-pix_fmt','gray','-'])
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1,96*54).mean(axis=1)
    assert np.max(np.abs(frame_values(source)-frame_values(output))) <= 1.0


def test_recut_uses_same_source_frame_and_audio_budget(tmp_path, monkeypatch):
    import recut
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setenv('AUDIO_NORMALIZE','1')
    source=tmp_path/'source.mp4'; output=tmp_path/'recut.mp4'
    fixture_video(source,audio=True)
    recut.run_cut_concat(str(source),[{'start':0.13,'end':3.93}],str(output),str(tmp_path))
    streams=probe(output)
    assert int(streams['video']['nb_frames']) == 90
    assert float(streams['video']['duration']) == pytest.approx(3.75375,abs=0.000001)
    assert abs(float(streams['audio']['duration'])-3.75375) <= 0.001
