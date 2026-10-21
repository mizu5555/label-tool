import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'prelabel'))

import check_labels  # noqa: E402
import prelabel  # noqa: E402
from label_tool_v6 import MainWindow  # noqa: E402

HAS_FFMPEG = shutil.which('ffmpeg') is not None


def _labels(xs, ys, src, index=None):
    n = len(xs)
    return pd.DataFrame({'V': [1 if np.isfinite(x) else -1 for x in xs], 'X': xs, 'Y': ys, 'Z': 0.0, 'src': src},
                        index=np.arange(n) if index is None else index)


def _write_clip(folder, name, n, fps=150, rows=None):
    (folder / 'video').mkdir(parents=True, exist_ok=True)
    (folder / 'csv').mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(folder / 'video' / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
    for _ in range(n):
        w.write(np.zeros((480, 640, 3), np.uint8))
    w.release()
    df = rows if rows is not None else pd.DataFrame({'Frame': range(n), 'Visibility': 0, 'X': 0.0, 'Y': 0.0})
    df.to_csv(folder / 'csv' / f'{name}_ball.csv', index=False)


class LosslessWriterTests(unittest.TestCase):
    @unittest.skipUnless(HAS_FFMPEG, 'ffmpeg not installed')
    def test_round_trip_is_pixel_exact_at_150_fps(self):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (480, 640), dtype=np.uint8) for _ in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'clip.mp4')
            w = prelabel.LosslessWriter(path)
            for f in frames:
                w.write(f)
            w.close()
            cap = cv2.VideoCapture(path)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FPS)), 150)
            for f in frames:
                ok, im = cap.read()
                self.assertTrue(ok)
                np.testing.assert_array_equal(im[..., 0], f)


class SanitizeTests(unittest.TestCase):
    def setUp(self):
        self.mt = np.arange(20) / 150.0

    def test_edge_label_becomes_unknown(self):
        xs = np.r_[np.linspace(100, 110, 10), [1.0]]
        L = prelabel.sanitize(_labels(xs, np.full(11, 200.0), ['det3d'] * 11), self.mt)
        self.assertEqual(L.src.iat[-1], 'edge')
        self.assertEqual(L.V.iat[-1], -1)

    def test_sudden_jump_becomes_unknown(self):
        xs = np.r_[np.linspace(100, 110, 10), [300.0]]
        L = prelabel.sanitize(_labels(xs, np.full(11, 200.0), ['det3d'] * 11), self.mt)
        self.assertEqual(L.src.iat[-1], 'jump')

    def test_fast_ball_consistent_with_neighbors_is_kept(self):
        xs = 100 + 32.0 * np.arange(11)  # 32 px/frame from the first labeled frame
        L = prelabel.sanitize(_labels(xs, np.full(11, 200.0), ['det3d'] * 11), self.mt)
        self.assertTrue((L.V == 1).all())

    def test_jump_across_dropped_frame_is_allowed(self):
        mt = np.r_[np.arange(10), [16]] / 150.0  # 6 dropped frames
        xs = np.r_[np.linspace(100, 145, 10), [175.0]]
        L = prelabel.sanitize(_labels(xs, np.full(11, 200.0), ['det3d'] * 11), mt)
        self.assertTrue((L.V == 1).all())

    def test_track_run_without_3d_anchor_becomes_unknown(self):
        xs = np.linspace(100, 120, 10)
        L = prelabel.sanitize(_labels(xs, np.full(10, 200.0), ['track'] * 8 + ['det3d'] * 2), self.mt)
        self.assertTrue((L.src == 'no_anchor').all())


class FinalJumpTests(unittest.TestCase):
    def test_inconsistent_step_is_listed_with_padding(self):
        xs = np.r_[np.linspace(100, 120, 10), [200.0], np.linspace(121, 131, 9)]
        L = _labels(xs, np.full(20, 200.0), ['det3d'] * 20)
        self.assertEqual(prelabel.final_jumps(L), list(range(7, 15)))

    def test_smooth_fast_ball_is_not_listed(self):
        L = _labels(100 + 30.0 * np.arange(15), np.full(15, 200.0), ['track'] * 15)
        self.assertEqual(prelabel.final_jumps(L), [])


class ReviewRangeTests(unittest.TestCase):
    def test_nearby_frames_merge_and_must_check_comes_first(self):
        src = ['det3d'] * 5 + ['track'] * 3 + ['det3d'] * 2 + ['track'] * 2 + ['unknown'] * 4
        rv = pd.DataFrame({'Frame': range(len(src)), 'source': src})
        rr = prelabel.review_ranges(rv)
        self.assertEqual(rr[0], (12, 15, 'must', 'unknown'))
        self.assertEqual(rr[1], (5, 11, 'spot', 'track'))

    def test_hit_note_is_listed_as_must_check(self):
        rv = pd.DataFrame({'Frame': range(5), 'source': ['det3d'] * 5})
        self.assertEqual(prelabel.review_ranges(rv, [(2, 'hit_unlabeled')]), [(2, 2, 'must', 'hit_unlabeled')])


class LabelToolCompatibilityTests(unittest.TestCase):
    def test_part_csv_loads_in_label_tool(self):
        labels = {1: (1, 320.5, 240.25, 2.0, 'det3d', 1), 2: (1, 321.0, 241.0, 2.0, 'track', 0), 3: (0, 0, 0, 0, 'out', 0)}
        parts = prelabel.part_tables(6, labels, np.arange(6) / 150.0, part_frames=4)
        self.assertEqual([len(c) for _, _, c, _ in parts], [4, 2])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'video').mkdir()
            (Path(tmp) / 'csv').mkdir()
            parts[0][2].to_csv(Path(tmp) / 'csv' / 'clip_ball.csv', index=False)
            from label_tool_v6 import csv_path_for_video
            reader = SimpleNamespace(csv_path=csv_path_for_video(str(Path(tmp) / 'video' / 'clip.mp4'), create_dir=False),
                                     df={}, fps=150.0)
            MainWindow.load_csv_if_any(reader)
            self.assertEqual(reader.df[1]['Visibility'], 1)
            self.assertAlmostEqual(reader.df[1]['X'], 320.5)
            self.assertEqual(reader.df[1]['Event'], 1)
            self.assertEqual(reader.df[0]['Visibility'], 0)  # idle frames default to no ball
        self.assertEqual(list(parts[1][3].source), ['idle', 'idle'])


class AutoPredictStopTests(unittest.TestCase):
    def window(self, no_ball):
        calls = []
        df = {0: {'Visibility': 1, 'Event': 0, 'X': 10.0, 'Y': 10.0}}
        w = SimpleNamespace(df=df, current_frame=0, total_frames=10, auto_predict_enabled=True, no_ball_frames=no_ball,
                            has_video=lambda: True, update_frame=lambda: None,
                            auto_predict_next=lambda idx: calls.append(idx))
        return w, calls

    def test_known_empty_frame_is_not_auto_predicted(self):
        w, calls = self.window({1})
        MainWindow.next_frame(w)
        self.assertEqual(calls, [])

    def test_undecided_frame_is_still_auto_predicted(self):
        w, calls = self.window(set())
        MainWindow.next_frame(w)
        self.assertEqual(calls, [1])

    def test_review_file_is_found_next_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'review').mkdir()
            pd.DataFrame({'Frame': range(5), 'source': ['det3d', 'dead', 'unknown', 'out', 'idle']}).to_csv(
                root / 'review' / 'clip.csv', index=False)
            from label_tool_v6 import load_no_ball_frames, review_path_for_video
            path = review_path_for_video(str(root / 'video' / 'clip.mp4'))
            self.assertEqual(load_no_ball_frames(path), {1, 3, 4})
            self.assertEqual(load_no_ball_frames(str(root / 'review' / 'missing.csv')), set())


def _shot_publish(events, shots=()):
    rows = [{'Type': 'Event', 'event': e, 'timestamp': ts, 'shot_id': sid,
             'payload_role': 'machine_feed' if e == 2 else 'player_return'} for ts, e, sid in events]
    rows += [{'Type': 'Segment', 'phase_label': 'primary_shot', 'shot_id': sid, 'shot_result': res} for sid, res in shots]
    return pd.DataFrame(rows, columns=['Type', 'event', 'timestamp', 'shot_id', 'payload_role', 'phase_label', 'shot_result'])


def _flight(t0, t1, y0, y1, z0=1.0, vz=3.0):
    t = np.arange(t0, t1, 1 / 150)
    s = t - t0
    return pd.DataFrame({'Timestamp': t, 'X': 0.0, 'Y': y0 + (y1 - y0) * s / (t1 - t0), 'Z': z0 + vz * s - 4.905 * s ** 2})


class FindRalliesTests(unittest.TestCase):
    def test_rally_ends_at_second_landing_after_last_hit(self):
        pub = _shot_publish([(10.0, 2, None), (11.0, 1, 1), (12.0, 1, 2), (12.6, 3, 2), (13.2, 4, 2)], [(2, 'in')])
        pts = pd.concat([_flight(10.0, 11.0, 6, -5), _flight(11.0, 13.4, -5, 5)])
        r, = prelabel.find_rallies(pub, pts)
        self.assertEqual((r['start'], r['end'], r['end_kind']), (10.0, 13.2, 'land2'))
        self.assertIn((12.6, prelabel.EV_BOUNCE), r['events'])
        self.assertIn((11.0, prelabel.EV_HIT), r['events'])
        self.assertIn((12.0, prelabel.EV_HIT), r['events'])

    def test_second_landing_is_extrapolated_from_the_rebound(self):
        pub = _shot_publish([(10.0, 1, 1), (10.8, 3, 1)], [(1, 'in')])
        rebound = _flight(10.8, 11.1, 2, 3, z0=0.04, vz=2.0)  # 3D lost while the ball comes down again
        pts = pd.concat([_flight(10.0, 10.8, -4, 2), rebound])
        r, = prelabel.find_rallies(pub, pts)
        self.assertEqual(r['end_kind'], 'land1_extrap')
        self.assertAlmostEqual(r['end'], 10.8 + (2.0 + np.sqrt(4 + 2 * 9.81 * 0.003)) / 9.81, delta=0.02)

    def test_same_time_bounce_events_count_once(self):
        pub = _shot_publish([(10.0, 1, 1), (10.8, 3, 1), (10.8, 4, 1)], [(1, 'in')])
        r, = prelabel.find_rallies(pub, _flight(10.0, 10.8, -4, 2))
        self.assertEqual((r['end'], r['end_kind']), (10.8, 'land1'))

    def test_far_side_3d_gap_does_not_split_the_rally(self):
        # serve, then no 3D for 1.8 s while the ball bounces on the far side, then the return
        pub = _shot_publish([(10.0, 2, None), (12.6, 1, 1), (13.4, 3, 1), (14.1, 4, 1)], [(1, 'in')])
        pts = pd.concat([_flight(10.0, 10.8, 6, -2), _flight(12.6, 14.2, -5, 4)])
        r, = prelabel.find_rallies(pub, pts)
        self.assertEqual((r['start'], r['end']), (10.0, 14.1))

    def test_every_serve_and_a_long_pause_start_new_rallies(self):
        pub = _shot_publish([(10.0, 2, None), (11.0, 1, 1), (12.0, 2, None), (13.0, 1, 2), (20.0, 1, 3)])
        rallies = prelabel.find_rallies(pub, pd.DataFrame(columns=['Timestamp', 'X', 'Y', 'Z']))
        self.assertEqual([r['start'] for r in rallies], [10.0, 12.0, 20.0])
        self.assertEqual([r['serve'] for r in rallies], [10.0, 12.0, None])
        self.assertLess(rallies[0]['end'], 12.0)

    def test_net_shot_ends_at_net_contact(self):
        pub = _shot_publish([(10.0, 2, None), (11.0, 1, 1)], [(1, 'net')])
        pts = pd.concat([_flight(10.0, 11.0, 6, -5), _flight(11.0, 12.0, -5, 1)])
        r, = prelabel.find_rallies(pub, pts)
        self.assertEqual(r['end_kind'], 'net')
        self.assertAlmostEqual(r['end'], 11.0 + 5 / 6, delta=0.02)  # y crosses 0 at 5/6 of the flight

    def test_net_time_ignores_the_next_serve(self):
        # net ball loses 3D before the net; the next serve's points must not be used
        pub = _shot_publish([(10.0, 2, None), (11.0, 1, 1), (11.7, 4, 1), (12.4, 2, None)], [(1, 'net')])
        pts = pd.concat([_flight(10.0, 11.0, 6, -5), _flight(11.0, 11.3, -5, -3), _flight(12.4, 13.0, 6.9, 0)])
        r = prelabel.find_rallies(pub, pts)[0]
        self.assertLessEqual(r['end'], 11.7)

    def test_hit_at_the_net_is_not_its_own_net_contact(self):
        pub = _shot_publish([(10.0, 1, 1), (10.6, 4, 1)], [(1, 'net')])
        pts = pd.concat([_flight(10.0, 10.3, -0.2, -1.5), _flight(10.3, 10.5, -1.5, -0.05)])  # hit near the net, ball goes back to it
        r, = prelabel.find_rallies(pub, pts)
        self.assertGreater(r['end'], 10.3)

    def test_landing_is_extrapolated_without_bounce_event(self):
        pub = _shot_publish([(10.0, 2, None)])
        pts = _flight(10.0, 10.5, 6, 1, z0=1.0, vz=1.0)  # 3D lost before the ball lands
        r, = prelabel.find_rallies(pub, pts)
        self.assertEqual(r['end_kind'], 'extrap')
        self.assertAlmostEqual(r['end'], 10.0 + (1.0 + np.sqrt(1 + 2 * 9.81 * 0.963)) / 9.81, delta=0.02)

    def test_without_machine_the_first_hit_starts_the_rally(self):
        pub = _shot_publish([(20.5, 1, 1), (21.6, 1, 2), (22.1, 3, 2), (22.7, 4, 2)])
        r, = prelabel.find_rallies(pub, _flight(20.0, 22.8, -5, 5))  # ball tossed before the serve hit
        self.assertEqual((r['start'], r['end'], r['serve']), (20.5, 22.7, None))

    def test_no_serve_or_hit_means_no_rally(self):
        self.assertEqual(prelabel.find_rallies(_shot_publish([]), _flight(5.0, 6.0, 0, 1)), [])


class CheckLabelsTests(unittest.TestCase):
    def test_clean_clip_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 20
            rows = pd.DataFrame({'Frame': range(n), 'Visibility': 1, 'X': np.linspace(100, 200, n), 'Y': 200.0})
            _write_clip(Path(tmp), 'ok', n, rows=rows)
            errors, warns, _ = check_labels.check_clip(str(Path(tmp) / 'video' / 'ok.mp4'), str(Path(tmp) / 'csv' / 'ok_ball.csv'), 20)
            self.assertEqual(errors, [])
            self.assertEqual(warns, [])

    def test_row_count_mismatch_and_zero_coordinates_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = pd.DataFrame({'Frame': range(5), 'Visibility': [1, 1, 0, 0, 0], 'X': [0.0, 100, 0, 0, 0], 'Y': [0.0, 100, 0, 0, 0]})
            _write_clip(Path(tmp), 'bad', 8, rows=rows)
            errors, _, _ = check_labels.check_clip(str(Path(tmp) / 'video' / 'bad.mp4'), str(Path(tmp) / 'csv' / 'bad_ball.csv'), 20)
            self.assertEqual(len(errors), 2)

    def test_jump_is_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 12
            xs = np.r_[np.linspace(100, 110, 11), [400.0]]
            rows = pd.DataFrame({'Frame': range(n), 'Visibility': 1, 'X': xs, 'Y': 200.0})
            _write_clip(Path(tmp), 'jump', n, rows=rows)
            _, warns, _ = check_labels.check_clip(str(Path(tmp) / 'video' / 'jump.mp4'), str(Path(tmp) / 'csv' / 'jump_ball.csv'), 20)
            self.assertTrue(any('jump' in w for w in warns))

    def test_fast_ball_from_first_frame_is_not_a_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 12
            xs = np.r_[[0.0] * 4, 150 + 32.0 * np.arange(8)]
            rows = pd.DataFrame({'Frame': range(n), 'Visibility': [0] * 4 + [1] * 8, 'X': xs, 'Y': 200.0})
            _write_clip(Path(tmp), 'fast', n, rows=rows)
            _, warns, _ = check_labels.check_clip(str(Path(tmp) / 'video' / 'fast.mp4'), str(Path(tmp) / 'csv' / 'fast_ball.csv'), 20)
            self.assertEqual(warns, [])

    def test_frozen_auto_predict_run_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 40
            xs = np.r_[np.linspace(100, 200, 20), np.full(20, 200.0)]  # ball lands, marker stays put
            rows = pd.DataFrame({'Frame': range(n), 'Visibility': 1, 'X': xs, 'Y': 200.0})
            _write_clip(Path(tmp), 'frozen', n, rows=rows)
            errors, _, _ = check_labels.check_clip(str(Path(tmp) / 'video' / 'frozen.mp4'), str(Path(tmp) / 'csv' / 'frozen_ball.csv'), 20)
            self.assertTrue(any('auto-predict' in e for e in errors))

    def test_static_ball_is_flagged_like_training(self):
        df = pd.DataFrame({'Frame': range(100), 'Visibility': 1, 'X': 300.0, 'Y': 200.0})
        self.assertTrue(check_labels.static_ball_mask(df, 150, 20).all())
        df['X'] = np.linspace(100, 400, 100)
        self.assertFalse(check_labels.static_ball_mask(df, 150, 20).any())


if __name__ == '__main__':
    unittest.main()
