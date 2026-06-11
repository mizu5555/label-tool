# Label Tool

PyQt5 desktop tool for labeling ball positions in video frames and saving the
annotations as CSV.

## Run From Source

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe label_tool_v6.py path\to\video.mp4
```

If no video path is provided, the app looks for videos in a `video` folder next
to the script. In packaged builds, it opens a file picker when no console is
available.

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
