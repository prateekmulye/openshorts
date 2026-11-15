"""Synthetic video and external-model fixtures; never evidence of model quality."""
import base64
import io
import json

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image

import gemini_worker
import llm_backend


def jpeg_url():
    data = io.BytesIO()
    Image.new('RGB', (32, 24), 'red').save(data, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(data.getvalue()).decode()


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / 'synthetic-fixture.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for i in range(100):
        writer.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    writer.release()
    return path


def test_vision_model_requires_explicit_optin(monkeypatch):
    monkeypatch.delenv('LLM_VISION_MODEL', raising=False)
    assert llm_backend.vision_model_name() is None
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    assert llm_backend.vision_model_name() == 'qwen2.5vl:3b'


def test_generate_json_embeds_jpeg_without_fetch(monkeypatch):
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    calls = []
    image = jpeg_url()

    def external_model(req):
        calls.append(req)
        body = json.loads(req.content)
        assert body['model'] == 'qwen2.5vl:3b'
        assert body['messages'][1]['content'] == [
            {'type': 'text', 'text': 'fixture prompt'},
            {'type': 'image_url', 'image_url': {'url': image}}]
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"shorts": []}'}}]})

    monkeypatch.setattr(llm_backend, '_client', lambda: httpx.Client(transport=httpx.MockTransport(external_model)))
    result, _ = llm_backend.generate_json('fixture prompt', gemini_worker.VisualResponse,
                                         model='qwen2.5vl:3b', images=[image])
    assert result == {'shorts': []}
    assert len(calls) == 1


@pytest.mark.parametrize('images', [['https://example.org/x.jpg'], ['data:image/jpeg;base64,!!!!'],
                                   [jpeg_url()] * 13, ['data:image/jpeg;base64,' + 'A' * 70000]])
def test_client_rejects_bad_or_excess_images_before_network(images, monkeypatch):
    monkeypatch.setattr(llm_backend, '_client', lambda: pytest.fail('must reject before network'))
    with pytest.raises(ValueError):
        llm_backend.generate_json('fixture', gemini_worker.VisualResponse, images=images)


def test_sampler_bounds_timestamps_and_local_only(video):
    import local_vision
    images, sampling = local_vision.sample_frames(video, 10)
    assert 1 <= len(images) <= 12
    assert len(sampling['timestamps_seconds']) == len(images)
    assert sampling['timestamps_seconds'][0] == 0
    assert all(0 <= t < 10 for t in sampling['timestamps_seconds'])
    assert sampling['whole_video_reviewed'] is False
    assert sampling['audio_reviewed'] is False
    for url in images:
        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
        assert len(raw) <= 49152
        with Image.open(io.BytesIO(raw)) as image:
            assert image.format == 'JPEG' and max(image.size) <= 512
    with pytest.raises(ValueError, match='local'):
        local_vision.sample_frames('https://example.org/video.mp4', 10)
    for duration in [float('nan'), float('inf'), -1, 0]:
        with pytest.raises(ValueError, match='duration'):
            local_vision.sample_frames(video, duration)


def test_selector_preserves_empty_selection_and_sampling_limits(video, monkeypatch):
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    calls = []

    def external_model(prompt, schema, model=None, images=None):
        calls.append((prompt, images))
        assert 'sampled frames' in prompt and 'audio' in prompt
        assert model == 'qwen2.5vl:3b'
        assert schema is gemini_worker.VisualResponse
        return {'shorts': []}, {'local': True}

    monkeypatch.setattr(llm_backend, 'generate_json', external_model)
    result = local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)
    assert result['shorts'] == []
    assert result['vision_sampling']['semantic_review'] == 'not_performed'
    # Twelve retained samples now require two calls under the fixed context cap.
    assert len(calls) == 2
    monkeypatch.delenv('LLM_VISION_MODEL')
    with pytest.raises(RuntimeError, match='LLM_VISION_MODEL'):
        local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)


def test_model_unavailable_remains_error(video, monkeypatch):
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError('fixture model unavailable')
    monkeypatch.setattr(llm_backend, 'generate_json', unavailable)
    with pytest.raises(RuntimeError, match='Local vision unavailable'):
        local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)


@pytest.mark.parametrize('size', [(512, 24), (32, 512), (513, 24), (32, 513)])
def test_client_exact_jpeg_dimension_boundary(size):
    data = io.BytesIO()
    Image.new('RGB', size, 'red').save(data, format='JPEG')
    url = 'data:image/jpeg;base64,' + base64.b64encode(data.getvalue()).decode()
    if 513 in size:
        with pytest.raises(ValueError, match='Invalid bounded JPEG'):
            llm_backend._image_parts([url])
    else:
        assert llm_backend._image_parts([url])[0]['image_url']['url'] == url


def test_context_limited_http_fixture_batches_all_samples(video, monkeypatch):
    """External HTTP fixture reproduces the observed 12903 > 8192 error, not model quality."""
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    original_images, original_sampling = local_vision.sample_frames(video, 10)
    requests = []
    def external_model(req):
        body = json.loads(req.content)
        content = body['messages'][1]['content']
        images = [part['image_url']['url'] for part in content if part['type'] == 'image_url']
        requests.append((images, content[0]['text']))
        if len(images) > 6:
            return httpx.Response(400, json={'error': {'message': 'request (12903 tokens) exceeds the available context size (8192 tokens), try increasing it', 'type': 'invalid_request_error'}})
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"shorts": []}'}}],
                                        'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
    monkeypatch.setattr(llm_backend, '_client', lambda: httpx.Client(transport=httpx.MockTransport(external_model)))
    result = local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)
    assert [len(images) for images, _ in requests] == [6, 6]
    # Interleaving preserves every sample while allowing candidates across the full source.
    assert requests[0][0] == original_images[::2]
    assert requests[1][0] == original_images[1::2]
    sampling = result['vision_sampling']
    assert sampling['timestamps_seconds'] == original_sampling['timestamps_seconds']
    assert sampling['attempted_batches'] == sampling['succeeded_batches'] == 2
    assert sampling['analysis_status'] == 'sampled_frames_analyzed'
    assert sampling['analyzed_timestamps_seconds'] == original_sampling['timestamps_seconds']
    assert result['cost_analysis']['input_tokens'] == 200
    assert result['cost_analysis']['output_tokens'] == 20
    assert sampling['whole_video_reviewed'] is False
    # Old half-source ranges could be shorter than the requested clip minimum.
    assert all('0 <= start < end <= 10' in prompt for _, prompt in requests)
    assert json.dumps(original_sampling['timestamps_seconds'][::2]) in requests[0][1]
    assert json.dumps(original_sampling['timestamps_seconds'][1::2]) in requests[1][1]
    assert sampling['batch_grouping'] == 'interleaved_sparse_frames'


def test_second_batch_failure_reports_partial_analysis_and_returns_no_clips(video, monkeypatch):
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    calls = []
    def external_model(*args, images=None, **kwargs):
        calls.append(images)
        if len(calls) == 2:
            raise httpx.ReadTimeout('fixture second batch unavailable')
        return {'shorts': [{'start': 0, 'end': 4}]}, {'local': True}
    monkeypatch.setattr(llm_backend, 'generate_json', external_model)
    with pytest.raises(RuntimeError, match='Local vision unavailable') as error:
        local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)
    metadata = json.loads(str(error.value).split('vision_sampling=', 1)[1])
    assert metadata['analysis_status'] == 'partial_analysis_unavailable'
    assert metadata['attempted_batches'] == 2
    assert metadata['succeeded_batches'] == 1
    assert len(metadata['analyzed_timestamps_seconds']) == 6
    assert [batch['status'] for batch in metadata['batches']] == ['succeeded', 'failed']


def test_first_batch_failure_marks_remaining_samples_unattempted(video, monkeypatch):
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    def external_model(*args, **kwargs):
        raise httpx.ReadTimeout('fixture first batch unavailable')
    monkeypatch.setattr(llm_backend, 'generate_json', external_model)
    with pytest.raises(RuntimeError, match='Local vision unavailable') as error:
        local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)
    metadata = json.loads(str(error.value).split('vision_sampling=', 1)[1])
    assert metadata['analysis_status'] == 'unavailable'
    assert metadata['attempted_batches'] == 1 and metadata['succeeded_batches'] == 0
    assert metadata['analyzed_timestamps_seconds'] == []
    assert [batch['status'] for batch in metadata['batches']] == ['failed', 'not_attempted']


def test_batch_candidates_merge_without_shifting_and_reject_outside_source(video, monkeypatch):
    # Full-source interleaved batches reject source violations, not valid cross-half clips.
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    calls = []
    def external_model(*args, **kwargs):
        calls.append(1)
        return {'shorts': ([{'start': 0, 'end': 4}, {'start': 6, 'end': 11}] if len(calls) == 1
                           else [{'start': 6, 'end': 9}, {'start': -1, 'end': 4}])}, None
    monkeypatch.setattr(llm_backend, 'generate_json', external_model)
    result = local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)
    assert result['shorts'] == [{'start': 0, 'end': 4}, {'start': 6, 'end': 9}]
    assert [batch['discarded_outside_batch'] for batch in result['vision_sampling']['batches']] == [1, 1]


def test_short_source_keeps_valid_candidate_longer_than_half_source(video, monkeypatch):
    """External-model fixture reproduces the verifier's 10s/7..12s regression."""
    import local_vision
    monkeypatch.setenv('LLM_BASE_URL', 'http://local.test/v1')
    monkeypatch.setenv('LLM_VISION_MODEL', 'qwen2.5vl:3b')
    monkeypatch.setenv('CLIP_MIN_SECONDS', '7')
    monkeypatch.setenv('CLIP_MAX_SECONDS', '12')
    prompts = []
    def external_model(prompt, *args, images=None, **kwargs):
        prompts.append(prompt)
        assert len(images) <= 6
        return {'shorts': [{'start': 0, 'end': 8}]}, None
    monkeypatch.setattr(llm_backend, 'generate_json', external_model)
    result = local_vision.get_visual_clips(video, 10, gemini_worker.VisualResponse)
    assert result['shorts'] == [{'start': 0, 'end': 8}, {'start': 0, 'end': 8}]
    assert len(prompts) == 2
    assert all('0 <= start < end <= 10' in prompt for prompt in prompts)
    # Duplicate proposals remain for the existing central engine deduper.
    from clip_selection import dedupe_overlapping
    assert dedupe_overlapping(result['shorts']) == [{'start': 0, 'end': 8}]
