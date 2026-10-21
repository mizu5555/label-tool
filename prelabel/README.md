# Pre-labeling CameraSensor recordings

`prelabel.py` turns a CameraSensor replay session into videos that label-tool opens
with labels already filled in. A person then reviews and fixes them instead of
labeling from scratch. The rest of the training pipeline is unchanged.

```text
replay/<session>/                       CameraSensor recording (machine or rally mode)
  │
  ├─ 1. prelabel.py                      automatic pre-labels, a few minutes per session
  │       <out>/<session>/video/*.mp4     lossless, 150 fps, whole recording per camera in 2-minute parts
  │       <out>/<session>/csv/*_ball.csv  Frame,Visibility,X,Y,Z,Event,Timestamp
  │       <out>/<session>/review/         what a person must check
  │       <out>/<session>/metadata.json   calibration.near_camera_head_width_px
  │
  ├─ 2. label-tool review                open each video; check the "must" ranges
  │
  ├─ 3. check_labels.py                  automatic checks before upload
  │
  └─ 4. upload video/ + csv/ (+ metadata.json) → frame generator → preprocess → training
```

## Requirements

- Python packages from `requirements.txt` (numpy, pandas, opencv)
- `ffmpeg` with libx264 on PATH (used to write lossless video)
- A session directory that contains `CameraReader_N.mp4`, `CameraReader_N.cfg`,
  `CameraReader_N_meta.csv`, `TrackNet_data_N.csv`, `Model3D_points.csv` and
  `Model3D_publish.csv`. Sessions recorded on the field machine have all of them.

## 1. Pre-label

```bash
python3 prelabel/prelabel.py \
  --session /path/to/replay/2026-10-13_15-00-27 \
  --out /path/to/labeling
```

| option | default | meaning |
|---|---|---|
| `--cams` | `0,1,2,3` | cameras to export |
| `--part-minutes` | `2` | length of each output video |
| `--head-width` | `20` | `near_camera_head_width_px` written to `metadata.json` |
| `--events` | `hit` | `hit`: only Hit (training oversamples these); `all`: also Serve and in-play Bounce; `none` |

Output videos are named `<session>_cam<N>_part<k>`, unique across sessions. The
script stops if `<out>/<session>/` already has videos, so results from different
runs never mix; pick a new `--out` or delete the old folder first.

### What gets labeled

Only the ball **in play** is labeled. A rally starts at a machine serve, or at the
first hit when there is no machine (a hit more than 3.5 s after the previous one
starts a new rally). It ends at the second landing after the last hit (the model
cannot know which shot is the last one, so the rebound after the first landing is
labeled like any other flight), or at the net contact when that shot went into the
net. In-play bounces (a serve landing before the return) are labeled. Everything else is Visibility 0: the ball inside
the machine or in a hand, rolling or lying after the rally, and the time between
rallies.

`rallies.csv` lists every rally with its start, end and how the end was found
(`end_kind`): `land2` (Model3D second-bounce event), `land1_extrap` (first bounce
event, second landing extrapolated from the rebound), `land1_activity` / `land1`
(rebound could not be extrapolated), `net` (closest 3D point to the net plane),
`net_bounce` (net ball with no 3D near the net; its bounce is used, so the end is
late by the fall time), `extrap` (no bounce event; first landing extrapolated) or
`activity`. Kinds other than `land2` and `net` deserve a closer look at `end_check`.

### How labels are made

Each frame gets a source, stored in `review/<video>.csv`:

| source | Visibility | meaning |
|---|---|---|
| `det3d` | 1 | live TrackNet detection confirmed by the 3D point projected into this camera |
| `proj_blob` | 1 | 3D point exists, TrackNet missed; ball found next to the projection |
| `track` | 1 | no 3D point (mostly the far/player side); ball tracked from labeled neighbors |
| `interp` | 1 | gap of up to 3 frames filled linearly |
| `pre`, `dead`, `idle` | 0 | before the serve, after the rally ended, between rallies |
| `out`, `out_est` | 0 | the ball is outside this camera's view |
| `unknown` | 0 | **not decided** — written as 0 so the CSV is valid, but must be reviewed |

All positions are the center of the bright blob in the background-subtracted image
(the same 10-frame median subtraction TrackNet uses). Live TrackNet output is
typically 2-3 px off the ball center, so detections are snapped to the blob.
Detections without a blob (static lights, objects) are rejected.

## 2. Review in label-tool

Open `<out>/<session>/video/<video>.mp4`; the matching CSV loads automatically.
label-tool also reads `review/<video>.csv` and does not auto-predict on frames the
pre-label decided have no ball in play (`idle`, `dead`, `pre`, `out`, `out_est`), so
the marker stops at the end of each rally and when the ball leaves the frame.
Undecided frames are still auto-predicted, which helps filling them. Any stray
prediction is saved with `s`/`q`; `check_labels.py` reports runs of labels that stay
on one pixel, which is what a stray auto-predict usually leaves.
`review_summary.csv` lists every video with its counts.

`review/<video>_ranges.csv` lists frame ranges:

- **`must`** — undecided frames (`unknown`, `edge`, `jump`, `no_anchor`), labels that
  still jump in the final result (`jump_check`, usually around a hit where the ball
  and racket meet), a hit that could not be placed on a labeled frame
  (`hit_unlabeled`), and ±5 frames around each rally end (`end_check`), which is an
  estimate. Label the ball if it is in play and visible; leave Visibility 0 if not.
- **`spot`** — automatically filled labels (`track`, `interp`, `proj_blob`), and
  idle stretches where live TrackNet followed something moving (`idle_motion`, a
  possible missed rally). Play through and make sure nothing is wrong.

## 3. Check before upload

```bash
python3 prelabel/check_labels.py /path/to/labeling/20261013_150027
```

Reports errors (CSV/video frame count mismatch, Frame numbering, visible labels at
0 or off-screen, 15+ labeled frames that do not move) and warnings (labels on the image border, sudden jumps, isolated
1-2 frame labels, videos where the training static-ball filter would drop more than
10% of positives). Exit code 1 if there is any error. It also works on folders
labeled by hand.

## 4. Upload and train

Copy `video/` and `csv/` into the training match folder (and `metadata.json` if the
match does not have one yet), then run the frame generator and training as usual.
`review/`, `review_summary.csv` and `rallies.csv` are not used by training.

Whole recordings contain many frames without a ball in play (typically 70-90%);
use the per-match sample limits in `dataset_split.json` if the balance matters.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests
```
