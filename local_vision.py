"""Local silent-clip candidates from sparse JPEG samples; never full-video review."""
import base64
import json
import math
import os
from pathlib import Path

import cv2

import llm_backend
from clip_selection import clip_duration_bounds
from frame_sampler import read_at


def sample_frames(video_path, video_duration):
    """Return bounded data JPEGs and the nominal timestamps actually sampled."""
    if not isinstance(video_duration, (int, float)) or not math.isfinite(video_duration) or video_duration <= 0:
        raise ValueError('Invalid source duration')
    path = Path(video_path)
    if '://' in str(video_path) or not path.is_file():
        raise ValueError('Vision needs an existing local video file')
    cap = cv2.VideoCapture(str(path.resolve()))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise RuntimeError('Local vision unavailable: source frames cannot be read')
        count = min(int(count), max(1, math.ceil(video_duration * fps)))
        # shortcut: at most 12 sparse frames; use scene-aware batches for denser review.
        indices = sorted({round(i * (count - 1) / 11) for i in range(12)})
        images, timestamps, missing = [], [], []
        for index, frame in zip(indices, read_at(cap, indices)):
            timestamp = round(index / fps, 3)
            if frame is None:
                missing.append(timestamp)
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 512 / max(width, height))
            resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            encoded = None
            for quality in (65, 45, 25):
                ok, data = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok and len(data) <= 49152:
                    encoded = data.tobytes()
                    break
            if encoded is None:
                missing.append(timestamp)
                continue
            images.append('data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii'))
            timestamps.append(timestamp)
        if not images:
            raise RuntimeError('Local vision unavailable: no usable sampled frames')
        return images, {
            'method': 'uniform_sparse_frames', 'timestamps_seconds': timestamps,
            'timestamp_basis': 'frame_index / nominal_fps; variable-frame-rate timing may differ',
            'unavailable_timestamps_seconds': missing, 'source_duration_seconds': video_duration,
            'sample_count': len(images), 'whole_video_reviewed': False,
            'audio_reviewed': False, 'semantic_review': 'not_performed',
            'limitations': 'Only sampled frames were supplied. Motion, sound and story between samples are unreviewed.',
        }
    finally:
        cap.release()


def get_visual_clips(video_path, video_duration, schema, language='en', viewer_clips=None):
    """Return candidates; the engine enforces source/duration bounds before rendering."""
    model = llm_backend.vision_model_name()
    if not llm_backend.active() or not model:
        raise RuntimeError('Local vision unavailable: set LLM_BASE_URL and explicit LLM_VISION_MODEL')
    images, sampling = sample_frames(video_path, video_duration)
    low, high = clip_duration_bounds()
    try:
        maximum = max(1, min(15, int(os.environ.get('CLIP_TARGET_MAX', '3'))))
    except ValueError:
        maximum = 3
    timestamps = sampling['timestamps_seconds']
    # shortcut: six images per call for the fixed local context; no sample is dropped.
    batch_size = 6
    batch_count = math.ceil(len(images) / batch_size)
    sampling.update(attempted_batches=0, succeeded_batches=0,
                    analyzed_timestamps_seconds=[], batches=[], batch_grouping='interleaved_sparse_frames')
    # Interleave across the source so batching cannot impose shorter clip limits.
    for index in range(batch_count):
        sampling['batches'].append({'timestamps_seconds': timestamps[index::batch_count],
                                    'start_seconds': 0, 'end_seconds': video_duration,
                                    'status': 'not_attempted'})
    shorts, costs = [], []
    for index, batch in enumerate(sampling['batches']):
        prompt = (
            'Propose visual clip candidates from these sampled frames of a silent video. '
            'You have not watched the whole video and have no audio. Do not infer dialogue, '
            'sound, complete motion or events between frames. Return an empty shorts list '
            'if the images do not support a useful candidate. Candidates need human semantic review. '
            f'The images follow this timestamp order in seconds: {json.dumps(batch["timestamps_seconds"])}. '
            f'Source duration: {video_duration:g} seconds. Use absolute seconds without resetting to zero. '
            'These sparse samples are interleaved across the source, not continuous coverage. '
            f'Only propose candidates within source bounds: '
            f'{batch["start_seconds"]:g} <= start < end <= {batch["end_seconds"]:g}; '
            f'each candidate must be {low:g} to {high:g} seconds. '
            f'Return at most {maximum} candidates. Write restrained image-grounded descriptions and hooks in {language}. '
            'Do not claim virality is measured. Output a single JSON object matching this schema: '
            + json.dumps(schema.model_json_schema())
        )
        sampling['attempted_batches'] += 1
        try:
            result, cost = llm_backend.generate_json(
                prompt, schema, model=model, images=images[index::batch_count])
        except Exception as exc:
            batch['status'] = 'failed'
            sampling['analysis_status'] = ('partial_analysis_unavailable' if sampling['succeeded_batches']
                                           else 'unavailable')
            # Fail closed: partial coverage cannot silently become a deliverable.
            raise RuntimeError(f'Local vision unavailable: {type(exc).__name__}: {exc}; '
                               f'vision_sampling={json.dumps(sampling)}') from exc
        batch['status'] = 'succeeded'
        sampling['succeeded_batches'] += 1
        sampling['analyzed_timestamps_seconds'].extend(batch['timestamps_seconds'])
        bounded = [clip for clip in result['shorts']
                   if batch['start_seconds'] <= clip['start'] < clip['end'] <= batch['end_seconds']]
        batch['discarded_outside_batch'] = len(result['shorts']) - len(bounded)
        shorts.extend(bounded)
        if cost:
            costs.append(cost)
    sampling['analysis_status'] = 'sampled_frames_analyzed'
    sampling['analyzed_timestamps_seconds'].sort()
    cost = dict(costs[0]) if costs else None
    if cost is not None:
        for key in ('input_tokens', 'output_tokens', 'thinking_tokens', 'input_cost', 'output_cost', 'total_cost'):
            cost[key] = sum(item.get(key, 0) for item in costs)
        cost['price_estimated'] = any(item.get('price_estimated', False) for item in costs)
    return {'shorts': shorts, 'cost_analysis': cost, 'vision_sampling': sampling}
