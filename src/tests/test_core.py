import unittest
import numpy as np

from eventvad_repro.evaluation import frame_scores, metrics, paint_labels
from eventvad_repro.graph import GraphConfig, build_blockwise_graph, propagate
from eventvad_repro.manifests import canonical_segments, validate_segments
from eventvad_repro.scoring import parse_anomaly_score, uniformly_sample_indices


class CoreTests(unittest.TestCase):
    def test_segments_cover_first_and_last_frames(self):
        video = {"video_id": "v", "frame_count": 10, "sha256": "a"}
        segments = canonical_segments(video, [3, 8], "c")
        validate_segments(segments, [{**video, "schema_version": 1, "dataset": "x", "split": "test", "source_path": "x", "fps_num": 30, "fps_den": 1, "width": 1, "height": 1, "timebase_policy": "native"}])
        self.assertEqual([(x["start_frame"], x["end_frame_exclusive"]) for x in segments], [(0, 3), (3, 8), (8, 10)])

    def test_graph_handles_tiny_video_and_is_finite(self):
        features = np.arange(5 * 640, dtype=np.float32).reshape(5, 640)
        output = propagate(features, build_blockwise_graph(features, GraphConfig(block_size=200)))
        self.assertEqual(output.shape, (5, 640)); self.assertTrue(np.isfinite(output).all())

    def test_strict_score_and_sampling(self):
        self.assertEqual(parse_anomaly_score("description\nAnomaly score: 0.25"), .25)
        self.assertEqual(parse_anomaly_score("0.25"), .25)
        with self.assertRaises(ValueError): parse_anomaly_score("score 0.25")
        self.assertEqual(uniformly_sample_indices(4, 5), [4] * 16)

    def test_exact_coverage_and_metrics(self):
        videos = [{"video_id": "v", "frame_count": 4}]
        segments = [{"video_id": "v", "segment_id": "a", "start_frame": 0, "end_frame_exclusive": 2}, {"video_id": "v", "segment_id": "b", "start_frame": 2, "end_frame_exclusive": 4}]
        predictions = [{"segment_id": "a", "status": "success", "score": .1}, {"segment_id": "b", "status": "success", "score": .9}]
        result = metrics(frame_scores(videos, segments, predictions), {"v": paint_labels(4, [(2, 4)])}, "average_precision")
        self.assertEqual(result["roc_auc"], 1.0); self.assertEqual(result["average_precision"], 1.0)

if __name__ == "__main__": unittest.main()
