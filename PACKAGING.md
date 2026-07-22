# Packaging

This project uses PyInstaller to build a desktop executable for the current
operating system.

PyInstaller does not cross-compile. Build Windows on Windows, macOS on macOS,
and Linux on Linux, or use the included GitHub Actions workflow.

## Local build

Windows PowerShell:

```powershell
.\scripts\build.ps1
```

The output is:

```text
dist\label_tool.exe
```

macOS / Linux:

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

On Linux, the output is:

```text
dist/label_tool
```

On macOS, build with Python 3.12 for the PyQt5/PyInstaller frozen app:

```bash
brew install python@3.12
PYTHON=/opt/homebrew/bin/python3.12 ./scripts/build.sh
```

The output is:

```text
dist/Label Tool.app
```

Launch it with:

```bash
open "dist/Label Tool.app"
```

For a debug build that prints frozen-app tracebacks in Terminal:

```bash
DEBUG=1 PYTHON=/opt/homebrew/bin/python3.12 ./scripts/build.sh
./dist/label_tool
```

If a copied, downloaded, or unzipped app is blocked by Gatekeeper quarantine on
another Mac, remove the quarantine attribute before launching:

```bash
xattr -dr com.apple.quarantine "dist/Label Tool.app"
open "dist/Label Tool.app"
```

## GitHub Actions build

Push this repository to GitHub, then run the `Build executables` workflow from
the Actions tab. It uploads three artifacts:

- `label-tool-Windows`
- `label-tool-macOS`
- `label-tool-Linux`

## Using the packaged app

You can pass a video path directly:

```bash
label_tool path/to/video.mp4
```

You can also put videos in a `video` folder next to the executable and launch
the app. If no console is available, the app opens a file picker.
