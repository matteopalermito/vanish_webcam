"""Model files, their download, and construction of the MediaPipe tasks."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gestures import GokuPoseDetector, MultiViewGestureRecognizer, MultiViewHandLandmarker


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode != "normal" and destination == GESTURE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[VanishCam] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "VanishCam/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Unable to download the MediaPipe models. "
                "Check your internet connection and try again."
            ) from error


def _require(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Model not found: {path}")


def _video_and_image(task_class, options):
    """Create the VIDEO tracker plus an IMAGE twin used for multi-view retries."""
    with ExitStack() as cleanup:
        primary = task_class.create_from_options(options)
        cleanup.callback(primary.close)
        alternate = task_class.create_from_options(
            replace(options, running_mode=vision.RunningMode.IMAGE)
        )
        cleanup.pop_all()
    return primary, alternate


def create_segmenter() -> vision.ImageSegmenter:
    _require(SEGMENTER_MODEL)
    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    ))


def create_victory_detector() -> MultiViewGestureRecognizer:
    _require(GESTURE_MODEL)
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        ),
    )
    return MultiViewGestureRecognizer(*_video_and_image(vision.GestureRecognizer, options))


def create_joined_hands_detector() -> MultiViewHandLandmarker:
    _require(HAND_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    )
    return MultiViewHandLandmarker(*_video_and_image(vision.HandLandmarker, options))


def create_forehead_detector() -> GokuPoseDetector:
    _require(HAND_MODEL, FACE_MODEL)
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.45,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    with ExitStack() as cleanup:
        hands = vision.HandLandmarker.create_from_options(hand_options)
        cleanup.callback(hands.close)
        face = vision.FaceLandmarker.create_from_options(face_options)
        cleanup.pop_all()
    return GokuPoseDetector(hands, face)


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)
