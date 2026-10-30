"""VanishCam: gesture-triggered vanish effects for the local webcam and a virtual camera."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np

from mediapipe_tasks import (
    FACE_MODEL, GESTURE_MODEL, HAND_MODEL, SEGMENTER_MODEL, create_segmenter, download_models,
    person_mask,
)
from modes import MODES, Mode, create_mode
from naruto_effect import load_log
from vanish_sound import SOUND_PATH
from video_io import open_camera, open_virtual_camera, send_virtual_frame


WINDOW_NAME = "Webcam - Preview"
CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55


class Phase(str, Enum):
    CALIBRATING = "calibrating"
    READY = "ready"
    VANISHING = "vanishing"
    APPEARING = "appearing"
    VANISHED = "vanished"


@dataclass
class AppState:
    phase: Phase = Phase.CALIBRATING
    calibration_started: float = 0.0
    effect_started: float | None = None
    last_gesture_check: float = float("-inf")
    fullscreen: bool = False
    mask: np.ndarray | None = None
    mask_at: float = float("-inf")
    vanished_frame: np.ndarray | None = None


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gesture-triggered vanish effects for the webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=960, help="Video width")
    parser.add_argument("--height", type=int, default=540, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in fullscreen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum FPS of the virtual output (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual webcam; by default it is enabled if available. --no-virtual-camera disables it",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Virtual webcam driver to use",
    )
    parser.add_argument("--virtual-device", help="Name of the virtual device to use")
    parser.add_argument(
        "--mode", choices=tuple(MODES),
        help="Skip the start menu and launch the chosen mode directly",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print recognition status in the terminal (no text on the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_mode() -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    title = "Choose effect"
    selected = [None]
    buttons = [(mode.name, f"{number}  {mode.title}", mode.hint, 140 + 120 * (number - 1))
               for number, mode in enumerate(MODES.values(), start=1)]
    window = np.zeros((180 + 120 * len(buttons), 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "CHOOSE YOUR VANISH", (68, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    for name, heading, detail, top in buttons:
        cv2.rectangle(window, (65, top), (635, top + 94), (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (65, top), (635, top + 94), (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (90, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (90, top + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for name, _, _, top in buttons:
                if top <= y <= top + 94:
                    selected[0] = name

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if ord("1") <= key < ord("1") + len(buttons):
                selected[0] = buttons[key - ord("1")][0]
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def listens_for_gesture(mode: Mode, phase: Phase) -> bool:
    return phase is Phase.READY or (mode.toggles and phase is Phase.VANISHED)


def start_effect(state: AppState, mode: Mode, now: float, appearing: bool) -> None:
    state.phase = Phase.APPEARING if appearing else Phase.VANISHING
    state.effect_started = now
    state.mask = None
    state.mask_at = float("-inf")
    state.vanished_frame = None
    mode.begin(appearing)


def clear_effect(state: AppState, mode: Mode) -> None:
    state.effect_started = None
    state.mask = None
    state.mask_at = float("-inf")
    state.vanished_frame = None
    mode.reset()


def to_mp_image(frame: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    create_segmenter().close()
    for name in MODES:
        create_mode(name, with_sound=False).close()
    print("[Webcam] Self-test complete: models, assets and audio found.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    mode = args.mode or choose_mode()
    if mode is None:
        return 0
    with ExitStack() as resources:
        return run_camera(args, resources, mode)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode_name: str) -> int:
    download_models(mode_name)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter = create_segmenter()
    resources.callback(segmenter.close)
    mode = create_mode(mode_name)
    resources.callback(mode.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    started = time.perf_counter()
    state = AppState(calibration_started=started, fullscreen=args.fullscreen)
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = float("-inf")
    print("[Webcam] Step out of the frame for 4 seconds: capturing the background.")
    if mode.instructions:
        print(mode.instructions)
    controls = "SPACE new background, R reappear, V vanish, F fullscreen, Q quit"
    if not mode.manual_keys:
        controls = "SPACE new background, F fullscreen, Q quit"
    print(f"[Webcam] Window controls: {controls}.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning images.")
            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            now = time.perf_counter()
            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if state.phase is Phase.CALIBRATING:
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturing background: {remaining} s.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = Phase.READY
                    print("[Webcam] Background ready. You can step back into the frame.")
                display = camera_frame
            else:
                if (
                    listens_for_gesture(mode, state.phase)
                    and now - state.last_gesture_check >= mode.detect_interval
                ):
                    state.last_gesture_check = now
                    mp_image = to_mp_image(camera_frame)
                    triggered = mode.detect(mp_image, elapsed_ms, now)
                    if args.gesture_debug and now - last_gesture_report >= 1.0:
                        print(mode.debug_status())
                        last_gesture_report = now
                    if triggered:
                        start_effect(state, mode, now, appearing=state.phase is Phase.VANISHED)

                if state.phase in (Phase.VANISHING, Phase.APPEARING):
                    age = now - state.effect_started
                    if mode.needs_mask(age, now - state.mask_at):
                        if mp_image is None:
                            mp_image = to_mp_image(camera_frame)
                        state.mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        state.mask_at = now
                    appearing = state.phase is Phase.APPEARING
                    display = mode.render(camera_frame, background, state.mask, age, appearing)
                    if age >= mode.effect_seconds:
                        if appearing:
                            state.phase = Phase.READY
                            clear_effect(state, mode)
                        else:
                            state.phase = Phase.VANISHED
                            state.vanished_frame = mode.finished_frame(background, display)
                elif state.phase is Phase.VANISHED:
                    display = state.vanished_frame
                else:
                    display = camera_frame

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = Phase.CALIBRATING
                state.calibration_started = now
                clear_effect(state, mode)
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Step out of the frame for 4 seconds: capturing the new background.")
            elif key == ord("r") and mode.manual_keys and background is not None:
                state.phase = Phase.READY
                clear_effect(state, mode)
            elif key == ord("v") and mode.manual_keys and background is not None:
                start_effect(state, mode, now, appearing=False)
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[VanishCam] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
