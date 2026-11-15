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


def fixture_video(path, *, shifted=False, audio=False, rate="24000/1001"):
    cmd = ['ffmpeg','-v','error','-y','-f','lavfi','-i',f"nullsrc=size=96x54:rate={rate},geq=lum='16+N*2':cb=128:cr=128"]
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


@pytest.mark.parametrize('cutter',['pipeline','recut'])
@pytest.mark.parametrize('start,end,frames,duration',[(0.13,3.93,90,3.75),(0.46,4.5,96,4.0)])
def test_integer_cfr_cut_has_aligned_audio_and_video_budget(tmp_path,monkeypatch,cutter,start,end,frames,duration):
    """24fps reproduces the v3 replay's skipped integer-rate branch.

    Non-grid cuts expose late video start; a cut reaching source EOF exposes
    loudnorm's audio overrun. Both ends must match the inward frame budget.
    """
    import recut
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setenv('AUDIO_NORMALIZE','1')
    source=tmp_path/'source.mp4';output=tmp_path/'cut.mp4'
    fixture_video(source,audio=True,rate='24')
    if cutter=='pipeline':
        ffmpeg_utils.cut_clip(str(source),str(output),start,end,1)
    else:
        recut.run_cut_concat(str(source),[{'start':start,'end':end}],str(output),str(tmp_path))
    streams=probe(output);v=streams['video'];a=streams['audio']
    assert float(v['start_time']) == 0
    assert float(a['start_time']) == 0
    assert int(v['nb_frames']) == frames
    assert float(v['duration']) == pytest.approx(duration,abs=0.000001)
    assert abs(float(a['duration'])-duration) <= 0.001


def test_chapter_bearing_scene_join_has_contiguous_frame_timestamps(tmp_path,monkeypatch):
    """A source chapter must not determine the concat offset of a shorter scene.

    The v4 failure retained all frames but inherited a chapter/data track
    ending at the full cut duration, extending the first scene's container.
    """
    monkeypatch.setenv('FFMPEG_ENCODER','x264')
    monkeypatch.setattr(reframe_v2.gpu_render,'ENABLED',False)
    monkeypatch.setattr(reframe_v2.gpu_decode,'ENABLED',False)
    raw=tmp_path/'raw.mp4';source=tmp_path/'source.mp4';output=tmp_path/'rendered.mp4'
    fixture_video(raw,audio=True,rate='24')
    chapters=tmp_path/'chapters.txt'
    chapters.write_text(';FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=4500\ntitle=Fixture chapter\n')
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-i',str(chapters),
        '-map','0','-map_metadata','1','-map_chapters','1','-c','copy',str(source)],check=True)
    graph=reframe_v2.general_filtergraph(96,160,orig_w=96,orig_h=54)
    specs=[(0,86/24,graph,None),(86/24,22/24,graph,None)]
    reframe_v2.encode_scenes(str(source),specs,96,54,96,160,24,str(tmp_path),None,str(output))
    frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(output)]))['frames']
    times=[float(f['best_effort_timestamp_time']) for f in frames]
    assert len(times) == 108
    assert max(abs((b-a)-1/24) for a,b in zip(times,times[1:])) < 0.000002
    assert times[-1] == pytest.approx(107/24,abs=0.000001)
    streams=probe(output)
    assert float(streams['video']['duration']) == 4.5
    assert float(streams['audio']['duration']) == 4.5
    segment=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_chapters',
        '-show_format','-show_streams','-of','json',str(tmp_path/'seg_000.mp4')]))
    assert segment['chapters'] == []
    assert [s['codec_type'] for s in segment['streams']] == ['video']
    assert float(segment['format']['duration']) == pytest.approx(86/24,abs=0.000001)
