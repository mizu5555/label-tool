import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from label_tool_v6 import (
    MainWindow,
    bbox_to_ball,
    compatible_output_fps,
    draw_pickleball_marker,
    empty_row,
    ensure_subdir,
    export_validation_video,
    frames_per_segment,
    group_labeled_frames,
)


class BboxToBallTests(unittest.TestCase):
    def test_box_converts_to_exact_center_and_radius(self):
        self.assertEqual(
            bbox_to_ball(10, 20, 21, 29, 1920, 1080),
            (15.5, 24.5, 5.5),
        )

    def test_reverse_drag_has_same_result(self):
        forward = bbox_to_ball(10, 20, 30, 40, 100, 100)
        reverse = bbox_to_ball(30, 40, 10, 20, 100, 100)
        self.assertEqual(forward, reverse)

    def test_small_box_uses_its_center(self):
        self.assertEqual(
            bbox_to_ball(10, 20, 13, 24, 100, 100),
            (11.5, 22.0, 2.0),
        )

    def test_click_without_drag_is_treated_as_point(self):
        self.assertEqual(
            bbox_to_ball(13, 24, 13, 24, 100, 100),
            (13.0, 24.0, 0.0),
        )

    def test_coordinates_are_clamped_to_frame(self):
        self.assertEqual(
            bbox_to_ball(-10, -20, 110, 120, 100, 80),
            (49.5, 39.5, 49.5),
        )

    def test_invalid_frame_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            bbox_to_ball(0, 0, 1, 1, 0, 100)


class LabelCsvRoundTripTests(unittest.TestCase):
    def test_center_and_radius_survive_csv_round_trip(self):
        row = empty_row(7, 30.0)
        row.update(
            Visibility=1,
            X=15.5,
            Y=24.5,
            Z=5.5,
            Event=3,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labels.csv"
            writer = SimpleNamespace(
                csv_path=str(csv_path),
                df={7: row},
                total_frames=8,
                fps=30.0,
            )
            MainWindow.save_csv(writer)

            reader = SimpleNamespace(csv_path=str(csv_path), df={}, fps=30.0)
            MainWindow.load_csv_if_any(reader)
            loaded = reader.df[7]

        self.assertEqual(loaded['Frame'], 7)
        self.assertEqual(loaded['Visibility'], 1)
        self.assertEqual(loaded['X'], 15.5)
        self.assertEqual(loaded['Y'], 24.5)
        self.assertEqual(loaded['Z'], 5.5)
        self.assertEqual(loaded['Event'], 3)
        self.assertAlmostEqual(loaded['Timestamp'], 7 / 30.0)


class SafeOutputPathTests(unittest.TestCase):
    def test_creates_nested_directory_below_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            created = Path(ensure_subdir(tmp_dir, 'clips', 'video'))
            self.assertTrue(created.is_dir())
            self.assertEqual(created, Path(tmp_dir).resolve() / 'clips' / 'video')

    def test_rejects_directory_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                ensure_subdir(tmp_dir, '..', 'outside')


class ClipGroupingTests(unittest.TestCase):
    def test_groups_frames_within_gap_tolerance(self):
        self.assertEqual(
            group_labeled_frames([1, 5, 35, 66], gap_tolerance=30),
            [(1, 35), (66, 66)],
        )

    def test_empty_labels_produce_no_clips(self):
        self.assertEqual(group_labeled_frames([]), [])


class VideoSplitMathTests(unittest.TestCase):
    def test_150_fps_two_minutes_is_18000_frames(self):
        self.assertEqual(frames_per_segment(150.0, 2.0), 18000)

    def test_fractional_fps_is_rounded_to_nearest_frame(self):
        self.assertEqual(frames_per_segment(29.97, 0.5), 899)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            frames_per_segment(0, 2)
        with self.assertRaises(ValueError):
            frames_per_segment(150, 0)

    def test_camera_fps_is_normalized_for_mpeg4_writer(self):
        self.assertEqual(compatible_output_fps(150.00181140089785), 150.0)

    def test_ntsc_fps_keeps_three_decimal_precision(self):
        self.assertEqual(compatible_output_fps(29.970029), 29.97)


class ValidationVideoTests(unittest.TestCase):
    def test_marker_changes_pixels_around_csv_point(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        row = empty_row(0, 30.0)
        row.update(Visibility=1, X=30, Y=20, Z=5, Event=3)

        draw_pickleball_marker(image, row)

        self.assertGreater(int(image[14:27, 24:37].sum()), 0)

    def test_exports_every_frame_with_labels_burned_in(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.mp4"
            output = Path(tmp_dir) / "validation" / "source_validation.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*'mp4v'),
                10.0,
                (64, 48),
                isColor=True,
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()

            row = empty_row(1, 10.0)
            row.update(Visibility=1, X=20, Y=20, Z=5, Event=1)
            written, cancelled = export_validation_video(
                str(source), str(output), {1: row}
            )

            self.assertEqual(written, 3)
            self.assertFalse(cancelled)
            self.assertTrue(output.is_file())
            cap = cv2.VideoCapture(str(output))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            self.assertEqual(len(frames), 3)
            self.assertGreater(int(frames[1][14:27, 14:27].sum()), 0)


if __name__ == '__main__':
    unittest.main()
