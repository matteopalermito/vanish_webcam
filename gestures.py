"""Gesture detectors built on MediaPipe hand and face landmarks."""

from __future__ import annotations

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from goku_effect import FACE_EDGE_LANDMARKS, FOREHEAD_LANDMARK, forehead_salute
from naruto_effect import joined_hands_from_landmarks


GESTURE_INTERVAL_SECONDS = 0.14
FACE_MEMORY_MS = 3000


def victory_landmark_score(points: np.ndarray) -> float:
    """Recognize extended index and middle fingers using 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring, pinky.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Seen from the side, folded fingers can look straight but very short:
    # do not require clearly visible joints for ring and pinky.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be apart; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the extended fingers, not a classifier probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # Landmarks are returned even when the classifier does not recognize
    # "Victory" (e.g. with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: estimated depth can be
    # inaccurate when the palm and folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Retry detection on rotated frames without disturbing tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forces a fresh detection; the VIDEO model stays on the original
        # frame. This avoids inconsistent timestamps/tracking across rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # In the side pose the palm is often too small in the full frame.
        # The two lower crops overlap and cover both hands, even when the
        # image is mirrored.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class GokuPoseDetector:
    """Detect index and middle fingers held together against the forehead."""

    def __init__(self, hands, face):
        self.hands = hands
        self.face = face
        self.forehead: np.ndarray | None = None
        self.face_width = 0.0
        self.face_seen_ms = -FACE_MEMORY_MS
        self.face_visible = False
        self.last_hand_count = 0

    def detect(self, image: mp.Image, timestamp_ms: int) -> bool:
        width, height = image.width, image.height
        faces = self.face.detect_for_video(image, timestamp_ms).face_landmarks
        self.face_visible = bool(faces)
        if faces:
            landmarks = faces[0]
            point = landmarks[FOREHEAD_LANDMARK]
            left, right = (landmarks[i] for i in FACE_EDGE_LANDMARKS)
            self.forehead = np.array((point.x * width, point.y * height))
            self.face_width = float(np.hypot((left.x - right.x) * width, (left.y - right.y) * height))
            self.face_seen_ms = timestamp_ms
        result = self.hands.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        # The hand often hides the face, so remember the last known forehead.
        if self.forehead is None or timestamp_ms - self.face_seen_ms > FACE_MEMORY_MS:
            return False
        for image_landmarks, world_landmarks in zip(result.hand_landmarks, result.hand_world_landmarks):
            image_points = np.array([(p.x * width, p.y * height) for p in image_landmarks])
            world_points = np.array([(p.x, p.y, p.z) for p in world_landmarks])
            if forehead_salute(image_points, world_points, self.forehead, self.face_width):
                return True
        return False

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            self.face.close()


def recognize_gesture(recognizer, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)
