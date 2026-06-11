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

The output is:

```text
dist/label_tool
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
