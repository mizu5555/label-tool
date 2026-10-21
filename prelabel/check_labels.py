#!/usr/bin/env python3
"""Pre-upload checks for labeled clips: video/CSV consistency, label coordinates, jumps and the
effect of the training-side static-ball filter.

Works on prelabel.py output and on any manually labeled folder containing video/ and csv/.
Usage:
  python3 prelabel/check_labels.py /path/to/labeling/20261013_143316 [more folders ...]
Exit code is 1 when any ERROR is found.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

REQUIRED = ['Frame', 'Visibility', 'X', 'Y']


def static_ball_mask(df, fps, head_width_px, threshold_cm=13, duration_s=1 / 3):
    """Static-ball rule of the training preprocess_csvV4: max displacement within duration_s,
    converted to cm via head width, below threshold_cm. Keep in sync if training moves to V5.
    (V4 additionally merges short static runs, so this flags slightly more frames.)"""
    window = int(duration_s * fps)
    half = window // 2
    vis = df.Visibility.values == 1
    X, Y = df.X.values, df.Y.values
    mask = np.zeros(len(df), bool)
    for i in range(len(df)):
        if not vis[i]:
            continue
        lo, hi = max(i - half, 0), min(i + half, len(df))
        seg = np.arange(lo, hi)
        if len(seg) < window:  # pad with the edge frame at clip boundaries
            pad = window - len(seg)
            seg = np.r_[[seg[0]] * pad, seg] if i < half else np.r_[seg, [seg[-1]] * pad]
        seg = seg[vis[seg]]
        if len(seg) < half:
            continue
        xy = np.c_[X[seg], Y[seg]]
        if np.max(np.linalg.norm(xy - xy[0], axis=1)) * 20 / head_width_px < threshold_cm:
            mask[i] = True
    return mask


def check_clip(video, csv_path, head_width):
    errors, warns = [], []
    cap = cv2.VideoCapture(video)
    nv, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not os.path.isfile(csv_path):
        return ['CSV missing'], [], {}
    df = pd.read_csv(csv_path)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        return [f'CSV lacks columns {miss}'], [], {}
    if len(df) != nv:
        errors.append(f'CSV has {len(df)} rows, video has {nv} frames')
    if (df.Frame.values != np.arange(len(df))).any():
        errors.append('Frame is not numbered 0..N-1')
    if fps <= 0:
        errors.append('cannot read fps')
    v = df[df.Visibility == 1]
    bad = v[(v.X <= 0) | (v.Y <= 0) | (v.X >= w) | (v.Y >= h)]
    if len(bad):
        errors.append(f'{len(bad)} frames have Visibility=1 with coordinates off-screen or 0 (e.g. Frame {bad.Frame.iloc[0]})')
    edge = v[(v.X < 3) | (v.Y < 3) | (v.X > w - 4) | (v.Y > h - 4)]
    if len(edge) > len(bad):
        warns.append(f'{len(edge) - len(bad)} frames sit on the image border')
    vis = df.Visibility.values == 1
    step = np.hypot(np.diff(df.X.values), np.diff(df.Y.values))
    both = vis[1:] & vis[:-1]
    for i in np.where(both)[0]:
        nb = [step[k] for k in range(max(0, i - 5), min(len(step), i + 6)) if k != i and both[k]]
        lim = max(20.0, 2.5 * float(np.median(nb))) if nb else 20.0  # compare with neighbors on both sides
        if step[i] > lim:
            warns.append(f'jump of {step[i]:.0f} px at Frame {i}->{i + 1}')
    runs, frozen, i = [], [], 0
    X, Y = df.X.values, df.Y.values
    while i < len(vis):
        if vis[i]:
            j = i
            while j < len(vis) and vis[j]:
                j += 1
            runs.append(j - i)
            for k in range(i, j - 14):  # a ball does not stay on one pixel: label-tool auto-predict left over
                if np.hypot(X[k:k + 15] - X[k], Y[k:k + 15] - Y[k]).max() < 1.5:
                    frozen.append(k)
                    break
            i = j
        else:
            i += 1
    if frozen:
        errors.append(f'{len(frozen)} runs of >=15 labeled frames that do not move (e.g. Frame {frozen[0]}); '
                      'label-tool auto-predict leftovers?')
    short = sum(r < 3 for r in runs)
    if short:
        warns.append(f'{short} isolated runs of 1-2 labeled frames')
    st = static_ball_mask(df, fps if fps > 0 else 150, head_width)
    if vis.sum() and st.sum() / vis.sum() > 0.1:
        warns.append(f'training would drop {st.sum()}/{vis.sum()} positives as static balls')
    info = dict(frames=len(df), fps=fps, visible=int(vis.sum()), hits=int((df['Event'] == 1).sum()) if 'Event' in df else 0,
                static_removed=int(st.sum()))
    return errors, warns, info


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check labeled clips before uploading to the training folder')
    ap.add_argument('folders', nargs='+', help='folders containing video/ and csv/')
    ap.add_argument('--head-width', type=float, default=None, help='override near_camera_head_width_px from metadata.json')
    ap.add_argument('--quiet', action='store_true', help='only print clips with problems')
    a = ap.parse_args(argv)
    total_err, rows = 0, []
    for folder in a.folders:
        meta = os.path.join(folder, 'metadata.json')
        hw = a.head_width
        if hw is None:
            if os.path.isfile(meta):
                hw = json.load(open(meta, encoding='utf-8'))['calibration']['near_camera_head_width_px']
            else:
                print(f'WARNING {folder}: no metadata.json (training will fail); using head width 20 for the static check')
                hw = 20.0
        videos = sorted(glob.glob(os.path.join(folder, 'video', '*.mp4')) + glob.glob(os.path.join(folder, 'video', '*.avi')))
        if not videos:
            print(f'ERROR {folder}: no videos under video/')
            total_err += 1
        for vid in videos:
            name = os.path.splitext(os.path.basename(vid))[0]
            errors, warns, info = check_clip(vid, os.path.join(folder, 'csv', name + '_ball.csv'), hw)
            total_err += len(errors)
            rows.append(dict(clip=name, **info, errors=len(errors), warnings=len(warns)))
            if errors or warns or not a.quiet:
                tag = 'ERROR' if errors else ('WARN ' if warns else 'OK   ')
                print(f'{tag} {name}  ' + '; '.join(errors + warns))
    if rows:
        r = pd.DataFrame(rows)
        print(f'\n{len(r)} clips, {int(r.frames.sum())} frames, {int(r.visible.sum())} visible, '
              f'{int(r.hits.sum())} hit events, {int(r.static_removed.sum())} frames dropped by the static filter; '
              f'{int(r.errors.sum())} errors, {int((r.warnings > 0).sum())} clips with warnings')
    sys.exit(1 if total_err else 0)


if __name__ == '__main__':
    main()
