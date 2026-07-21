# Label Tool

PyQt5 desktop tool for labeling ball positions in video frames and saving the
annotations as CSV.

## Run From Source

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe label_tool_v6.py path\to\video.mp4
```

If no video path is provided, the app opens first without a video. Use the
`Open Video` button on the left panel to choose or switch videos.

## Split Video

Use the `Split Video` button to divide the current video (or choose another
video) by a user-selected number of minutes. At 150 fps, the recommended
default is 2 minutes (18,000 frames) and segments of 5 minutes (45,000 frames)
or less are recommended for labeling. Split files are written beside the
source video in a new folder and contain video only (no audio).

## CSV Output

Labels are saved as `<video_name>_ball.csv`.

- If the video is inside a folder named `video`, CSV files are saved to the
  sibling `csv` folder.
- Otherwise, CSV files are saved to a `csv` folder inside the video's parent
  folder.

The training CSV schema is:

```text
Frame,Visibility,X,Y,Z,Event,Timestamp
```

On the pickleball branch, events are:

```text
0 Normal, 1 Hit, 2 Land, 3 Serve, 4 Net, 5 Bounce
```

Useful pickleball shortcuts:

```text
z/x      previous / next frame
d/f      previous / next 50 frames
v        next frame + HoughCircles aimbot
a        optical-flow track 15 frames
b        toggle auto predict
c        clear current frame
0-5      set event
e        export labeled clips
s        save CSV
q        save and quit
```

## Build

Windows:

```powershell
.\scripts\build.ps1
```

macOS / Linux:

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

See [PACKAGING.md](PACKAGING.md) for GitHub Actions builds for Windows, macOS,
and Linux.
