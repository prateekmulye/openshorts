"""TransNet transition frames must survive scene concatenation.

Detector probabilities and decoded frames below are hermetic fixtures, not
runtime evidence. The real detector adapter and FrameTimecode conversion run.
"""
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    import scene_detection as scenes
    from transnetv2_pytorch import TransNetV2
except ImportError:
    scenes = None


@unittest.skipIf(scenes is None, 'requires scene-detection runtime dependencies')
class SceneCoverageTests(unittest.TestCase):
    def detect(self, probabilities, reported_count=None, min_sec='0'):
        class Model:
            device = 'cpu'
            predictions_to_scenes = staticmethod(TransNetV2.predictions_to_scenes)

            def predict_frames(self, frames, quiet=True):
                return torch.tensor(probabilities), None

        def capture_property(prop):
            if prop == scenes.cv2.CAP_PROP_FPS:
                return 24.0
            return len(probabilities) if reported_count is None else reported_count

        with patch.object(scenes.cv2, 'VideoCapture') as capture, \
             patch.object(scenes, '_extract_frames_small', return_value=np.zeros(
                 (len(probabilities), 27, 48, 3), dtype=np.uint8)), \
             patch.object(scenes, '_get_tn2_model', return_value=Model()), \
             patch.dict('os.environ', {'SCENE_MIN_SEC': min_sec}):
            capture.return_value.get.side_effect = capture_property
            result, fps = scenes._detect_transnetv2('fixture.mp4')
        self.assertEqual(fps, 24.0)
        return [(start.get_frames(), end.get_frames()) for start, end in result]

    def test_one_and_two_frame_transition_gaps_are_not_deleted(self):
        # Inclusive detector ranges: [0,28], [29,67], [68,97],
        # [99,171], [174,191], [192,248]. Missing frames 98,172,173.
        probabilities = [0.0] * 249
        for start, end in [(28,29), (67,68), (97,99), (171,174), (191,192)]:
            probabilities[start:end] = [1.0] * (end-start)
        self.assertEqual(self.detect(probabilities),
                         [(0,29), (29,68), (68,99), (99,174), (174,192), (192,249)])

    def test_leading_and_trailing_transitions_cover_decoded_frames(self):
        self.assertEqual(self.detect([1,1,0,0,1,1,1], reported_count=11), [(0,7)])

    def test_decoded_length_wins_over_underreported_container_count(self):
        self.assertEqual(self.detect([0,0,1,1,1], reported_count=3), [(0,5)])

    def test_short_scene_merge_keeps_complete_partition(self):
        self.assertEqual(self.detect([0,1,1,0,0,0,0,0], min_sec='0.2'), [(0,8)])

    def test_single_scene_unchanged(self):
        self.assertEqual(self.detect([0,0,0,0]), [(0,4)])

    def test_all_transition_frames_are_still_one_complete_scene(self):
        self.assertEqual(self.detect([1,1,1,1]), [(0,4)])


if __name__ == '__main__':
    unittest.main()
