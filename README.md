# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

Three effects are available, each triggered by a hand gesture: **Normal Vanish**, **Naruto Vanish**, and **Goku**.

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe]( https://nicoaauriemma.vercel.app/shop ) and double-click it to run the app. You do not need to install Python.

The published EXE may predate the Goku mode. To use it now, run the local scripts or rebuild the EXE from these sources.

## Requirements

- Windows with a webcam.
- [Python 3](https://www.python.org/downloads/) (tested with Python 3.14). The `py` launcher or `python` must be on the `PATH`.
- Internet access on the first run only.

On the first run, `START_LOCAL.bat` creates a private Python environment in `.venv-local`, installs the packages from `requirements.txt`, and downloads the MediaPipe models into `models/`. Later runs start immediately and work offline.

## Getting started

1. Double-click `START_LOCAL.bat`.
2. In the start menu, choose **Normal Vanish**, **Naruto Vanish**, or **Goku** with the mouse or the `1`, `2`, and `3` keys.
3. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Goku:** Hold your index and middle fingers straight and together against your forehead, with the ring and little fingers folded, for 1 second. Bright vertical comic-style speed lines run across your silhouette, a flash follows, and you teleport away; the lines linger briefly afterward. To reappear, lower your hand, then repeat the gesture for another 1 second: the return plays the effect in reverse. Short tracking dropouts are tolerated, but if you break the pose the countdown starts over. The terminal reports when the pose is recognized and when the teleport fires. The `V` and `R` keys are disabled in this mode: the gesture is the only way to disappear and reappear. The effect is silent by default; to add a sound, put an MP3 file at `assets/teleport.MP3` and it plays on both disappearance and reappearance.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

To start a mode without the menu:

```powershell
.\START_LOCAL.bat --mode normal
.\START_LOCAL.bat --mode naruto
.\START_LOCAL.bat --mode goku
```

The launchers switch to their own folder, so you can also run them from anywhere with their full path.

## Using it in Teams, Zoom, or Meet

1. Install [OBS Studio](https://obsproject.com/) once (administrator rights are required). It provides the **OBS Virtual Camera** driver. Open OBS once after installing, then close it: OBS does not need to be running afterward.
2. Run `START_VIRTUAL.bat` (optionally with `--mode goku`, `--mode naruto`, or `--mode normal`). The terminal confirms `Virtual output active: OBS Virtual Camera`.
3. In your meeting app, choose **OBS Virtual Camera** as the camera. In Teams: **Settings → Devices → Camera**. If it is not listed, restart the meeting app.

Keep VanishCam running for the whole call. If you cannot install software, share the **Webcam - Preview** window in the meeting instead: participants see the effect as a screen share rather than in your camera tile.

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `START_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `START_OBS.bat`, add a **Window Capture** source, select **Webcam - Preview**, and then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
.\START_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear.
- `V`: manually start the selected mode's effect.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

The `V` and `R` keys are disabled in Goku mode.

You can change the camera, resolution, and maximum virtual-output FPS:

```powershell
.\START_LOCAL.bat --camera 1 --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Troubleshooting

- **"Webcam 0 not available"**: close other apps using the camera (Teams, browser, OBS), or try `--camera 1`.
- **"Virtual webcam not available"**: install OBS Studio, and make sure **Start Virtual Camera** is not active inside OBS.
- **A gesture is not recognized**: run with `--gesture-debug` and check the terminal. For Goku, keep your face visible and your fingers straight and touching.
- **Video looks low-resolution**: your webcam may only support 640x480. Try `--width 1280 --height 720`.
- **Reset the environment**: delete the `.venv-local` folder; the next run reinstalls everything.

## Building the EXE

Run `BUILD_WINDOWS.bat`. It installs PyInstaller and the dependencies, downloads the models, builds `dist\VanishCam.exe` from `VanishCam.spec`, and runs a self-test.

## Running the tests

After the first run of `START_LOCAL.bat` has created the environment:

```powershell
.\.venv-local\Scripts\python.exe -m unittest discover -p "test_*.py"
```

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

Goku mode uses the [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (`face_landmarker.task`) to locate the forehead and `hand_landmarker.task` to track the fingers. The teleport's speed lines and flash are drawn in code (`goku_effect.py`), without anime frames.
