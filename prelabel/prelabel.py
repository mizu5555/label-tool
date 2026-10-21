#!/usr/bin/env python3
"""Pre-label a CameraSensor replay session into clips label-tool can open directly
(video/, csv/, review/, metadata.json).

Each camera becomes lossless videos of up to --part-minutes. Only the ball in play is labeled:
from a serve (machine launch, or the first hit without a machine) to the second landing after the
last hit, or to the net contact for a net ball; everything else is Visibility=0. Label sources
(recorded in review/):
  det3d      live TrackNet detection within DET_TOL of the projected 3D point (lens distortion
             included); position snapped to the background-subtracted blob center
  proj_blob  3D point exists but this camera missed it; blob found near the projection
  track      no 3D point (mostly the player side); blob tracked frame by frame from labeled neighbors
  interp     linear fill of gaps up to MAX_INTERP frames
  pre / dead / idle      Visibility=0: before the serve, after the ball is dead, between rallies
  out / out_est          Visibility=0: projection off-screen, or estimated position outside this
             camera's view
  unknown    undecidable; written as Visibility=0 and listed in review/ for a human to check

Usage:
  python3 prelabel/prelabel.py --session /path/to/replay/2026-10-13_14-33-16 --out /path/to/labeling
"""
import argparse
import ast
import configparser
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

FPS = 150
W, H = 640, 480
DET_TOL = 6.0       # max detection-to-projection distance (px)
BLOB_TOL = 6.0      # blob search radius (px)
TRACK_GATE = 10.0   # max per-frame deviation from the predicted track (px)
DIFF_THR = 20       # blob threshold on the background-subtracted image (0-255)
MAX_INTERP = 3
OUT_MARGIN = 30     # estimated position must be this far off-screen to count as unseen
EV_HIT, EV_SERVE, EV_BOUNCE = 1, 3, 5  # label-tool event ids
G = 9.81
CONFIDENT = ('det3d', 'proj_blob', 'track', 'interp')


# ---------------- camera & projection ----------------

def load_camera(session_dir, c):
    cf = configparser.ConfigParser()
    cf.read(os.path.join(session_dir, f'CameraReader_{c}.cfg'))
    o = cf['Other']
    E = np.array(ast.literal_eval(o['extrinsic_mat']))
    K = np.array(ast.literal_eval(o['ks']))
    dist = np.array(ast.literal_eval(o['dist']))
    rv, _ = cv2.Rodrigues(E[:3, :3])
    return dict(E=E, K=K, dist=dist, rv=rv, t=E[:3, 3], f=K[0, 0], r_max=_monotonic_radius(dist))


def _monotonic_radius(dist):
    """Largest normalized radius where the radial distortion polynomial is still increasing;
    beyond it projections fold back and are meaningless."""
    k1, k2, _, _, k3 = np.ravel(dist)[:5]
    r = np.linspace(0, 2, 4001)
    d = np.diff(r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6))
    return float(r[np.argmax(d <= 0)]) if (d <= 0).any() else 2.0


def project(cam, X):
    """3D -> raw (distorted) image coordinates. Returns (uv, depth); depth is -1 behind the camera
    or outside the valid range of the distortion model."""
    X = np.asarray(X, float).reshape(-1, 3)
    Xc = X @ cam['E'][:3, :3].T + cam['t']
    depth = Xc[:, 2].copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.hypot(Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2])
    depth[~(r < 0.98 * cam['r_max'])] = -1
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), cam['rv'], cam['t'], cam['K'], cam['dist'])
    return uv.reshape(-1, 2), depth


def in_image(u, v, margin=0):
    return -margin <= u < W + margin and -margin <= v < H + margin


# ---------------- rally segmentation ----------------

def find_rallies(publish, pts, max_gap=3.5):
    """Rallies from serve and hit events: every machine serve starts a rally, and so does a hit
    more than max_gap seconds after the previous serve/hit (play without a machine). 3D gaps are
    ignored, since the far side often has none for more than a second while the ball is in play.

    Each rally runs from its serve (or first hit) to the second landing after its last serve/hit
    (the model cannot tell the last shot from the others, so the rebound stays a positive), or to
    the net contact when that shot went into the net. end_kind tells how the end was found:
    'land2' (Model3D second-bounce event), 'land1_extrap' (first bounce event, second landing
    extrapolated from the rebound), 'land1_activity' / 'land1' (rebound not extrapolable: last 3D
    point / first bounce), 'net' (closest 3D point to the net plane), 'net_bounce' (net ball without
    3D near the net: its bounce, late by the fall time), 'extrap' (no bounce event: first landing
    extrapolated) or 'activity' (last 3D point)."""
    ev = publish[publish.Type == 'Event'].dropna(subset=['timestamp']).sort_values('timestamp')
    props = [(float(r.timestamp), 'serve' if r.event == 2 else 'hit', r.shot_id) for r in ev.itertuples()
             if r.event == 1 or (r.event == 2 and r.payload_role == 'machine_feed')]
    groups = []
    for t, kind, sid in props:
        if not groups or kind == 'serve' or t - groups[-1][-1][0] > max_gap:
            groups.append([])
        groups[-1].append((t, kind, sid))
    rallies = []
    for i, g in enumerate(groups):
        nxt = groups[i + 1][0][0] if i + 1 < len(groups) else np.inf
        rallies.append(_rally_end(g, nxt, ev, publish, pts))
    return rallies


def _rally_end(g, nxt, ev, publish, pts):
    start, last_prop = g[0][0], g[-1][0]
    serve = start if g[0][1] == 'serve' else None
    result = None
    if g[-1][1] == 'hit' and 'shot_result' in publish:
        seg = publish[(publish.Type == 'Segment') & (publish.phase_label == 'primary_shot') & (publish.shot_id == g[-1][2])]
        result = seg.shot_result.iloc[0] if len(seg) else None
    after = pts[(pts.Timestamp > last_prop + 0.02) & (pts.Timestamp < min(nxt, last_prop + 3.0))]
    bt = np.sort(ev[ev.event.isin([3, 4]) & (ev.timestamp > last_prop) & (ev.timestamp < nxt)].timestamp.unique())
    bounces = [float(t) for i, t in enumerate(bt) if i == 0 or t - bt[i - 1] > 0.05]  # event 3 = 1st, 4 = 2nd landing
    end, kind = None, None
    if result == 'net':  # a net ball drops after the contact, so the contact is before its bounce
        b1 = bounces[0] if bounces else None
        end = _net_time(after if b1 is None else after[after.Timestamp <= b1])
        if end is not None and (b1 is None or end <= b1):
            kind = 'net'
        else:
            end, kind = b1, 'net_bounce'
    elif len(bounces) >= 2:
        end, kind = bounces[1], 'land2'
    elif bounces:  # second landing from the rebound after the first one
        rebound = after[after.Timestamp > bounces[0] + 0.03]
        end, kind = _landing_time(rebound), 'land1_extrap'
        if end is None and len(rebound):
            end, kind = float(rebound.Timestamp.iloc[-1]), 'land1_activity'
        if end is None:
            end, kind = bounces[0], 'land1'
    if end is None:
        end, kind = _landing_time(after), 'extrap'
    if end is None:
        end, kind = (float(after.Timestamp.iloc[-1]) if len(after) else last_prop + 0.3), 'activity'
    end = min(end, nxt - 1 / FPS)
    events = [(t, EV_HIT) for t, k, _ in g if k == 'hit'] + ([(serve, EV_SERVE)] if serve is not None else [])
    events += [(float(t), EV_BOUNCE) for t in ev[ev.event.isin([3, 4]) & (ev.timestamp > start)
                                                  & (ev.timestamp < end - 0.02)].timestamp.unique()]
    return dict(start=start, end=end, end_kind=kind, serve=serve, events=sorted(events))


def _landing_time(q, ball_r=0.037, horizon=1.0):
    """Ballistic extrapolation (gravity only) of the last 3D points to the ground."""
    if len(q) < 8:
        return None
    q = q[q.Timestamp >= q.Timestamp.iloc[-1] - 0.3]
    t0 = q.Timestamp.iloc[-1]
    if q.Z.iloc[-1] <= 0.1:
        return float(t0)
    t = q.Timestamp.values - t0
    vz, z0 = np.polyfit(t, q.Z.values + 0.5 * G * t ** 2, 1)
    disc = vz ** 2 + 2 * G * (z0 - ball_r)
    if disc < 0:
        return None
    dt = (vz + np.sqrt(disc)) / G
    return float(t0 + dt) if 0 <= dt <= horizon else None


def _net_time(q):
    """Net contact of a shot that went into the net: the 3D point closest to the net plane
    (y = 0); without points, a linear extrapolation of y to 0 within 0.5 s."""
    if len(q) >= 3:
        if q.Y.abs().min() <= 0.5:
            return float(q.Timestamp.iloc[int(np.argmin(q.Y.abs().values))])
        tail = q.iloc[-8:]
        vy, y0 = np.polyfit(tail.Timestamp.values - tail.Timestamp.iloc[-1], tail.Y.values, 1)
        if vy and 0 <= -y0 / vy <= 0.5:
            return float(tail.Timestamp.iloc[-1] - y0 / vy)
    return None


# ---------------- frames & background subtraction ----------------

def read_ranges(path, ranges):
    """Decode the needed frames sequentially (cap.set seeks are off by several frames on these
    mp4 files). Returns {frame: gray}."""
    need = set()
    for a, b in ranges:
        need.update(range(max(a, 0), b))
    if not need:
        return {}
    cap = cv2.VideoCapture(path)
    out, i, last = {}, 0, max(need)
    while i <= last:
        if i in need:
            ok, im = cap.read()
            if not ok:
                break
            out[i] = im[..., 0].copy() if im.ndim == 3 else im
        elif not cap.grab():
            break
        i += 1
    return out


class BlobFinder:
    """10-frame median background subtraction centered on the frame (same as TrackNet's
    preprocessing), evaluated only in a small window around the candidate."""

    def __init__(self, frames):
        self.F = frames

    def find(self, f, x, y, tol, r=24):
        fr = [self.F.get(f + k) for k in range(-5, 5)]
        if any(v is None for v in fr) or not np.isfinite(x + y):
            return None
        x0 = int(np.clip(x - r, 0, W - 2 * r))
        y0 = int(np.clip(y - r, 0, H - 2 * r))
        st = np.stack([v[y0:y0 + 2 * r, x0:x0 + 2 * r] for v in fr]).astype(np.int16)
        d = np.clip(st[5] - np.median(st, 0), 0, 255).astype(np.uint8)
        n, lab, stats, cen = cv2.connectedComponentsWithStats((d > DIFF_THR).astype(np.uint8))
        best = None
        for i in range(1, n):
            if not 1 <= stats[i, 4] <= 80:
                continue
            c = cen[i] + [x0, y0]
            dd = np.hypot(c[0] - x, c[1] - y)
            if dd <= tol and (best is None or dd < best[0]):
                best = (dd, c, stats[i, 4])
        return None if best is None else (float(best[1][0]), float(best[1][1]), int(best[2]))


# ---------------- pre-labeling ----------------

def label_clip(cam, clip, pts, det, mt, blobs):
    """Per-frame labels for one clip on one camera, indexed by source frame number."""
    fa = int(np.searchsorted(mt, clip['a']))
    fb = min(int(np.searchsorted(mt, clip['b'])), len(mt))
    fs = int(np.searchsorted(mt, clip['serve'])) if clip['serve'] is not None else fa  # ball is inside the machine before a serve
    idx = np.arange(fa, fb)
    L = pd.DataFrame({'V': -1, 'X': np.nan, 'Y': np.nan, 'Z': 0.0, 'src': 'unknown'}, index=idx)

    q = pts[(pts.Timestamp >= mt[fa] - 0.01) & (pts.Timestamp <= mt[fb - 1] + 0.01)]
    proj = {}
    if len(q):
        uv, depth = project(cam, q[['X', 'Y', 'Z']].values)
        ts = q.Timestamp.values
        fr = np.clip(np.searchsorted(mt, ts), 1, len(mt) - 1)
        fr = np.where(abs(mt[fr] - ts) < abs(mt[fr - 1] - ts), fr, fr - 1)
        for f, (u, v), z, t in zip(fr, uv, depth, ts):
            if abs(mt[f] - t) < 1 / (2 * FPS) and fa <= f < fb:
                proj[int(f)] = (u, v, z)
    radius = lambda z: round(float(0.037 * cam['f'] / max(z, 0.5)), 2)

    for f in idx:
        if f < fs and f not in proj:
            L.loc[f, ['V', 'src']] = [0, 'pre']
            continue
        if f not in proj:
            continue
        u, v, z = proj[f]
        if z <= 0 or not in_image(u, v):
            L.loc[f, ['V', 'src']] = [0, 'out']
            continue
        if f in det.index and np.hypot(det.at[f, 'X'] - u, det.at[f, 'Y'] - v) <= DET_TOL:
            b = blobs.find(f, det.at[f, 'X'], det.at[f, 'Y'], BLOB_TOL)
            if b:  # no blob means a static object (e.g. a lamp): reject
                L.loc[f, ['V', 'X', 'Y', 'Z', 'src']] = [1, b[0], b[1], radius(z), 'det3d']
            continue
        b = blobs.find(f, u, v, BLOB_TOL)
        if b:
            L.loc[f, ['V', 'X', 'Y', 'Z', 'src']] = [1, b[0], b[1], radius(z), 'proj_blob']

    fw, bw = _track(L, idx, blobs), _track(L, idx[::-1], blobs)
    for f in set(fw) | set(bw):
        a, b = fw.get(f), bw.get(f)
        if a and b and np.hypot(a[0] - b[0], a[1] - b[1]) > 4:
            continue  # forward and backward tracks disagree
        x, y, area = a if a else b
        L.loc[f, ['V', 'X', 'Y', 'Z', 'src']] = [1, x, y, round(float(np.sqrt(area / np.pi)), 2), 'track']
    _interpolate(L)
    return L


def _track(L, order, blobs):
    res, prev, vel, miss = {}, None, np.zeros(2), 0
    for f in order:
        if L.at[f, 'V'] == 1:
            p = np.array([L.at[f, 'X'], L.at[f, 'Y']])
            if prev is not None and miss == 0:
                vel = p - prev
            prev, miss = p, 0
            continue
        if L.at[f, 'V'] == 0 or prev is None:
            prev = None
            continue
        pred = prev + vel * (miss + 1)
        b = blobs.find(f, pred[0], pred[1], TRACK_GATE)
        if b:
            res[f] = b
            vel = (np.array(b[:2]) - prev) / (miss + 1)
            prev, miss = np.array(b[:2]), 0
        else:
            miss += 1
            if miss > 3:
                prev = None
    return res


def _interpolate(L):
    V = L.V.values
    i = 0
    while i < len(V):
        if V[i] != -1:
            i += 1
            continue
        j = i
        while j < len(V) and V[j] == -1:
            j += 1
        if 0 < i and j < len(V) and V[i - 1] == 1 and V[j] == 1 and j - i <= MAX_INTERP:
            x0, y0, x1, y1 = L.X.iat[i - 1], L.Y.iat[i - 1], L.X.iat[j], L.Y.iat[j]
            for k in range(i, j):
                w = (k - i + 1) / (j - i + 1)
                L.iloc[k, L.columns.get_indexer(['V', 'X', 'Y', 'src'])] = [1, x0 * (1 - w) + x1 * w, y0 * (1 - w) + y1 * w, 'interp']
        i = j


def jump_limit(step, valid, i, k=5, floor=20.0):
    """Largest plausible per-frame step at i: 2.5x the median of up to k valid steps on each
    side (a fast ball is consistent with its neighbors; a wrong label is not)."""
    nb = [step[j] for j in range(max(1, i - k), min(len(step), i + k + 1)) if j != i and valid[j - 1]]
    return max(floor, 2.5 * float(np.median(nb))) if nb else floor


def final_jumps(L, pad=3):
    """Positions in the written labels that check_labels.py would flag as jumps (± pad frames),
    so they are reviewed even when every frame came from a confident source."""
    V = (L.V == 1).values
    step = np.r_[np.nan, np.hypot(np.diff(L.X.values), np.diff(L.Y.values))]
    valid = V[1:] & V[:-1]
    bad = [i for i in range(1, len(L)) if valid[i - 1] and step[i] > jump_limit(step, valid, i)]
    return sorted({j for i in bad for j in range(max(0, i - pad), min(len(L), i + pad + 1)) if V[j]})


def sanitize(L, mt):
    """Edge-clamped labels, discontinuous jumps and track runs without 3D anchors -> unknown."""
    edge = (L.V == 1) & ((L.X < 3) | (L.X > W - 4) | (L.Y < 3) | (L.Y > H - 4))
    L.loc[edge, ['V', 'src']] = [-1, 'edge']
    f, V, X, Y = L.index.values, L.V.values, L.X.values, L.Y.values
    nf = np.r_[1, np.maximum(1, np.round(np.diff(mt[f]) * FPS))]  # dropped camera frames allow larger steps
    step = np.r_[np.nan, np.hypot(np.diff(X), np.diff(Y))] / nf
    jump = [i for i in range(1, len(f)) if V[i] == 1 and V[i - 1] == 1
            and step[i] > jump_limit(step, (V[1:] == 1) & (V[:-1] == 1), i)]
    for i in jump:
        L.iloc[i, L.columns.get_indexer(['V', 'src'])] = [-1, 'jump']
    V, src, i = L.V.values, L.src.values, 0
    while i < len(V):
        if V[i] != 1:
            i += 1
            continue
        j = i
        while j < len(V) and V[j] == 1:
            j += 1
        if sum(s in ('det3d', 'proj_blob') for s in src[i:j]) < 3:
            L.iloc[i:j, L.columns.get_indexer(['V', 'src'])] = [-1, 'no_anchor']
        i = j
    return L


def mark_unseen(L, cam, pts, mt, max_gap=2.5):
    """For frames still unknown, estimate the position by interpolating neighboring 3D points;
    if it projects well off-screen the camera cannot see the ball."""
    unk = L.index[L.V == -1]
    if not len(unk) or not len(pts):
        return L
    ts, P = pts.Timestamp.values, pts[['X', 'Y', 'Z']].values
    for f in unk:
        t = mt[f]
        i = np.searchsorted(ts, t)
        if i == 0 or i >= len(ts) or ts[i] - ts[i - 1] > max_gap:
            continue
        w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        (u, v), z = (lambda r: (r[0][0], r[1][0]))(project(cam, P[i - 1] * (1 - w) + P[i] * w))
        if z <= 0 or not in_image(u, v, OUT_MARGIN):
            L.loc[f, ['V', 'src']] = [0, 'out_est']
    return L


def place_events(L, clip, mt, mode):
    L['Event'] = 0
    notes = []
    if mode == 'none':
        return notes
    for t, e in clip["events"]:
        if mode == 'hit' and e != EV_HIT:
            continue
        f = int(np.argmin(abs(mt - t)))
        cand = [g for g in range(f - 2, f + 3) if g in L.index and L.at[g, 'V'] == 1]
        if cand:
            L.loc[min(cand, key=lambda g: abs(g - f)), 'Event'] = e
        elif f in L.index:
            L.loc[f, 'Event'] = e
            if e == EV_HIT:
                notes.append((f, 'hit_unlabeled'))
    return notes


# ---------------- output ----------------

class LosslessWriter:
    """Lossless grayscale H.264 (full range, so OpenCV reads back the exact pixels)."""

    def __init__(self, path, fps=FPS):
        self.p = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(fps),
             '-i', '-', '-c:v', 'libx264', '-qp', '0', '-preset', 'veryfast', '-pix_fmt', 'gray', '-color_range', 'pc', path],
            stdin=subprocess.PIPE)

    def write(self, gray):
        self.p.stdin.write(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')


def part_tables(n_frames, labels, mt, part_frames):
    """Split per-frame labels of a whole recording into label-tool CSV tables and review tables.
    labels: {frame: (V, X, Y, Z, source, event)}; frames without an entry are idle (V=0)."""
    out = []
    for p, a in enumerate(range(0, n_frames, part_frames)):
        idx = np.arange(a, min(n_frames, a + part_frames))
        rows = [labels.get(int(f), (0, 0.0, 0.0, 0.0, 'idle', 0)) for f in idx]
        V = np.array([r[0] for r in rows], int)
        csv = pd.DataFrame({
            'Frame': np.arange(len(idx)), 'Visibility': V,
            'X': np.where(V == 1, np.round([r[1] for r in rows], 2), 0.0),
            'Y': np.where(V == 1, np.round([r[2] for r in rows], 2), 0.0),
            'Z': np.where(V == 1, [r[3] for r in rows], 0.0),
            'Event': [int(r[5]) for r in rows],
            'Timestamp': np.round(np.arange(len(idx)) / FPS, 6)})
        rv = pd.DataFrame({'Frame': np.arange(len(idx)), 'source': [r[4] for r in rows], 'src_frame': idx,
                           'cam_timestamp': mt[np.minimum(idx, len(mt) - 1)]})
        out.append((p, a, csv, rv))
    return out


MUST_CHECK = ('unknown', 'edge', 'jump', 'no_anchor', 'hit_unlabeled', 'jump_check', 'end_check')
SPOT_CHECK = ('track', 'interp', 'proj_blob', 'idle_motion')


def review_ranges(rv, extra=(), merge_gap=5):
    """Group frames needing a human into ranges: 'must' (undecided) and 'spot' (auto-filled labels);
    ranges closer than merge_gap frames are merged."""
    s = rv.set_index('Frame').source.copy()
    for f, reason in extra:
        s.loc[f] = reason
    out = []
    for level, reasons in (('must', MUST_CHECK), ('spot', SPOT_CHECK)):
        sel = s[s.isin(reasons)]
        start = prev = None
        kinds = set()
        for f, r in sel.items():
            if start is not None and f - prev > merge_gap + 1:
                out.append((start, prev, level, '+'.join(sorted(kinds))))
                start, kinds = None, set()
            if start is None:
                start = f
            kinds.add(r)
            prev = f
        if start is not None:
            out.append((start, prev, level, '+'.join(sorted(kinds))))
    return sorted(out, key=lambda x: (x[2] != 'must', x[0]))


def session_tag(session_dir):
    return os.path.basename(os.path.normpath(session_dir)).replace('-', '')


def idle_motion(det, labels, n_frames, min_run=10, min_travel=20.0):
    """Idle frames where live TrackNet followed something moving: possibly a missed rally."""
    out, run = [], []
    for f in range(n_frames):
        ok = f in det.index and labels.get(f, (0, 0, 0, 0, 'idle'))[4] == 'idle'
        if ok and run and f == run[-1] + 1:
            run.append(f)
            continue
        if len(run) >= min_run:
            xy = det.loc[run, ['X', 'Y']].values
            if np.linalg.norm(xy - xy[0], axis=1).max() >= min_travel:
                out += run
        run = [f] if ok else []
    return out


def label_camera(session_dir, c, rallies, pts, events_mode, part_frames, root, tag):
    """Stream one camera: write lossless 2-minute parts and pre-label every rally on the way."""
    cam = load_camera(session_dir, c)
    mt = pd.read_csv(os.path.join(session_dir, f'CameraReader_{c}_meta.csv')).timestamp.values
    det = pd.read_csv(os.path.join(session_dir, f'TrackNet_data_{c}.csv'))
    det = det[det.Visibility == 1].drop_duplicates('Frame').set_index('Frame')
    frame_of = lambda t: int(np.clip(np.searchsorted(mt, t), 0, len(mt) - 1))
    wins = []
    for r in rallies:
        fs, fe = frame_of(r['start']), frame_of(r['end'])
        wins.append(dict(r, fs=fs, fe=fe, a=r['start'] - 0.1, b=r['end'] + 0.15,
                         lo=frame_of(r['start'] - 0.1) - 6, hi=frame_of(r['end'] + 0.15) + 6))
    labels, notes, buf, wi = {}, [], {}, 0
    cap = cv2.VideoCapture(os.path.join(session_dir, f'CameraReader_{c}.mp4'))
    writer, f = None, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = im[..., 0] if im.ndim == 3 else im
        if f % part_frames == 0:
            if writer:
                writer.close()
            writer = LosslessWriter(os.path.join(root, 'video', f'{tag}_cam{c}_part{f // part_frames:02d}.mp4'))
        writer.write(g)
        if any(w['lo'] <= f < w['hi'] for w in wins[wi:wi + 3]):
            buf[f] = g.copy()
        while wi < len(wins) and f >= wins[wi]['hi'] - 1:
            _label_rally(cam, wins[wi], pts, det, mt, buf, labels, notes, events_mode)
            keep = wins[wi + 1]['lo'] if wi + 1 < len(wins) else f + 1
            buf = {k: v for k, v in buf.items() if k >= keep}
            wi += 1
        f += 1
    if writer:
        writer.close()
    for w in wins[wi:]:  # rallies running past the end of the recording
        _label_rally(cam, w, pts, det, mt, buf, labels, notes, events_mode)
    notes += [(i, 'idle_motion') for i in idle_motion(det, labels, f)]
    summary = []
    for p, a, csv, rv in part_tables(f, labels, mt, part_frames):
        name = f'{tag}_cam{c}_part{p:02d}'
        csv.to_csv(os.path.join(root, 'csv', name + '_ball.csv'), index=False)
        rv.to_csv(os.path.join(root, 'review', name + '.csv'), index=False)
        L = pd.DataFrame({'V': csv.Visibility.values, 'X': csv.X.values, 'Y': csv.Y.values})
        part_notes = [(n - a, r) for n, r in notes if a <= n < a + len(csv)]
        part_notes += [(i, 'jump_check') for i in final_jumps(L)]
        rr = review_ranges(rv, part_notes)
        pd.DataFrame(rr, columns=['start_frame', 'end_frame', 'level', 'reason']).to_csv(
            os.path.join(root, 'review', name + '_ranges.csv'), index=False)
        src = rv.source.value_counts().to_dict()
        summary.append(dict(video=name, camera=c, frames=len(csv), rally_frames=len(csv) - src.get('idle', 0),
                            visible=int(csv.Visibility.sum()),
                            unknown=sum(src.get(x, 0) for x in ('unknown', 'edge', 'jump', 'no_anchor')),
                            must_check=sum(r[2] == 'must' for r in rr), spot_check=sum(r[2] == 'spot' for r in rr)))
    return summary


def _label_rally(cam, w, pts, det, mt, buf, labels, notes, events_mode):
    """Pre-label one rally from buffered frames and merge it into the per-frame labels."""
    L = label_clip(cam, w, pts, det, mt, BlobFinder(buf))
    L = L[[f in buf for f in L.index]]
    if not len(L):
        return
    L = mark_unseen(sanitize(L, mt), cam, pts, mt)
    dead = (L.index < w['fs']) | (L.index > w['fe'])  # before the serve or after the ball is dead
    L.loc[dead, ['V', 'src']] = [0, 'pre']
    L.loc[L.index > w['fe'], 'src'] = 'dead'
    notes += place_events(L, w, mt, events_mode)
    notes += [(f, 'end_check') for f in range(w['fe'] - 5, w['fe'] + 6)]  # the cut point is an estimate
    for f, r in L.iterrows():
        V = 0 if r.V == -1 else int(r.V)
        if r.src == 'pre' and labels.get(int(f), (0,))[0] == 1:
            continue  # overlapping windows: keep the previous rally's ball
        labels[int(f)] = (V, float(r.X) if V else 0.0, float(r.Y) if V else 0.0, float(r.Z) if V else 0.0, r.src, int(r.Event))


def run(session_dir, out_root, cams, head_width, events, part_minutes):
    tag = session_tag(session_dir)
    root = os.path.join(out_root, tag)
    if os.path.isdir(os.path.join(root, 'video')) and os.listdir(os.path.join(root, 'video')):
        # mixing runs with different settings would leave stale files behind
        sys.exit(f'{root} already has videos; choose another --out or delete it first')
    for sub in ('video', 'csv', 'review'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    publish = pd.read_csv(os.path.join(session_dir, 'Model3D_publish.csv'))
    pts = pd.read_csv(os.path.join(session_dir, 'Model3D_points.csv')).sort_values('Timestamp')
    rallies = find_rallies(publish, pts)
    kinds = pd.Series([r['end_kind'] for r in rallies]).value_counts().to_dict()
    print(f'{tag}: {len(rallies)} rallies (end found by {kinds}), cameras {cams}', flush=True)
    part_frames = int(round(part_minutes * 60 * FPS))
    summary = []
    for c in cams:
        summary += label_camera(session_dir, c, rallies, pts, events, part_frames, root, tag)
        print(f'  cam{c} done', flush=True)
    pd.DataFrame([dict(start=r['start'], end=r['end'], end_kind=r['end_kind'], serve=r['serve'] is not None,
                       hits=sum(e == EV_HIT for _, e in r['events'])) for r in rallies]).to_csv(
        os.path.join(root, 'rallies.csv'), index=False)
    meta = {'calibration': {'near_camera_head_width_px': head_width},
            'source': {'session': os.path.basename(os.path.normpath(session_dir)), 'fps': FPS, 'resolution': [W, H], 'cameras': cams},
            'labeling': 'prelabel.py semi-automatic labels; review/ lists frames to check'}
    with open(os.path.join(root, 'metadata.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(root, 'review_summary.csv'), index=False)
    print(f'Output: {root} ({len(sm)} videos; {sm.must_check.sum()} must-check ranges, {sm.spot_check.sum()} spot-check ranges)')
    return root


def main(argv=None):
    ap = argparse.ArgumentParser(description='Pre-label a CameraSensor replay session for label-tool')
    ap.add_argument('--session', required=True, help='replay session dir (CameraReader_N.mp4/.cfg/_meta.csv, TrackNet_data_N.csv, Model3D_*.csv)')
    ap.add_argument('--out', required=True, help='output root; creates <out>/<session>/')
    ap.add_argument('--cams', default='0,1,2,3')
    ap.add_argument('--head-width', type=float, default=20.0, help='near_camera_head_width_px written to metadata.json')
    ap.add_argument('--part-minutes', type=float, default=2.0, help='split each camera into videos of this length')
    ap.add_argument('--events', choices=['hit', 'all', 'none'], default='hit',
                    help='hit: Hit only; all: also Serve and in-play Bounce; none: no events')
    a = ap.parse_args(argv)
    for f in ['Model3D_publish.csv', 'Model3D_points.csv']:
        if not os.path.isfile(os.path.join(a.session, f)):
            sys.exit(f'{f} not found: this session has no live 3D output, cannot pre-label')
    run(a.session, a.out, [int(x) for x in a.cams.split(',')], a.head_width, a.events, a.part_minutes)


if __name__ == '__main__':
    main()
