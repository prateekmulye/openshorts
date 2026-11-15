"""Measured delivery contract; model suggestions are never delivery evidence."""
import json
import math
import subprocess
from fractions import Fraction


def outcome(requested_min, requested_max, selected_count, delivered_count=None):
    count = selected_count if delivered_count is None else delivered_count
    status = ('no_suitable_moments' if selected_count == 0 else
              'partial' if count < requested_min else
              'pending' if delivered_count is None else 'complete')
    return dict(requested_min=requested_min, requested_max=requested_max,
                selected_count=selected_count, delivered_count=delivered_count,
                status=status, shortfall_count=max(0, requested_min - count))


def display_aspect(stream):
    sar = stream.get('sample_aspect_ratio') or '1:1'
    sar = Fraction(1) if sar in ('N/A', '0:1') else Fraction(sar.replace(':', '/'))
    aspect = Fraction(stream['width']) * sar / Fraction(stream['height'])
    rotation = float(stream.get('tags', {}).get('rotate', 0))
    for data in stream.get('side_data_list', []):
        if 'rotation' in data:
            rotation = float(data['rotation'])
    if round(rotation) % 180 == 90:
        aspect = 1 / aspect
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError('invalid display aspect')
    return float(aspect)


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_streams', '-show_format', '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    duration = float(stream.get('duration') or data['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid measured duration')
    return {'duration': duration, 'display_aspect': display_aspect(stream)}


def validate_delivery(path, min_seconds, max_seconds, output_format, source_aspect):
    measured = probe(path)
    if not min_seconds <= measured['duration'] <= max_seconds:
        raise ValueError(f"OUTPUT_CONTRACT:duration {measured['duration']:.3f}s outside {min_seconds:g}..{max_seconds:g}s")
    target = source_aspect if output_format in ('original', 'horizontal') else (1.0 if output_format == 'square' else 9/16)
    if measured['display_aspect'] != target:
        raise ValueError(f"OUTPUT_CONTRACT:aspect {measured['display_aspect']:.6f} expected {target:.6f}")
    return measured
