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

## CSV Output

Labels are saved as `<video_name>_ball.csv`.

- If the video is inside a folder named `video`, CSV files are saved to the
  sibling `csv` folder.
- Otherwise, CSV files are saved to a `csv` folder inside the video's parent
  folder.

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
