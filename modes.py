"""Vanish modes: each one pairs a gesture trigger with its effect.

To add a mode, subclass Mode, implement detect() and render(), and list the
class in MODES. The main loop, the start menu and --mode pick it up from there.
"""

from __future__ import annotations

import mediapipe as mp
import numpy as np

from gestures import GESTURE_INTERVAL_SECONDS, recognize_gesture
from goku_effect import GOKU_HOLD_SECONDS, GOKU_SECONDS, render_teleport
from mediapipe_tasks import (
    APP_DIR, create_forehead_detector, create_joined_hands_detector, create_victory_detector,
)
from naruto_effect import EFFECT_SECONDS, load_log, render_substitution, substitution_anchor
from normal_effect import VANISH_SECONDS, apply_dissolve, make_dissolve_fields
from vanish_sound import VanishSound


TELEPORT_SOUND = APP_DIR / "assets" / "teleport.MP3"
MASK_INTERVAL_SECONDS = 0.05


class Mode:
    """A gesture that triggers an effect drawn between the live frame and the background."""

    name = ""
    title = ""
    hint = ""
    instructions = ""
    effect_seconds = 1.0
    detect_interval = GESTURE_INTERVAL_SECONDS
    # When False, the V/R keys are disabled and only the gesture drives the mode.
    manual_keys = True
    # When True, the same gesture also makes the person reappear.
    toggles = False

    def __init__(self, detector, sound: VanishSound | None = None):
        self.detector = detector
        self.sound = sound

    @classmethod
    def create(cls, with_sound: bool = True) -> Mode:
        raise NotImplementedError

    def detect(self, image: mp.Image, timestamp_ms: int, now: float) -> bool:
        """Return True when the trigger gesture fires."""
        raise NotImplementedError

    def debug_status(self) -> str:
        return ""

    def begin(self, appearing: bool) -> None:
        if self.sound is not None:
            self.sound.play()

    def reset(self) -> None:
        """Forget per-effect state (new background or manual reappear)."""

    def needs_mask(self, age: float, mask_age: float) -> bool:
        return mask_age >= MASK_INTERVAL_SECONDS

    def render(self, frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
               age: float, appearing: bool) -> np.ndarray:
        raise NotImplementedError

    def finished_frame(self, background: np.ndarray, last_display: np.ndarray) -> np.ndarray:
        """Frame shown while the person is gone."""
        return last_display

    def close(self) -> None:
        try:
            self.detector.close()
        finally:
            if self.sound is not None:
                self.sound.close()


class NormalMode(Mode):
    name = "normal"
    title = "NORMAL VANISH"
    hint = "V sign  /  dissolve"
    effect_seconds = VANISH_SECONDS

    def __init__(self, detector, sound: VanishSound | None = None):
        super().__init__(detector, sound)
        self.status = ("No_Hand", 0.0)
        self.fields: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def create(cls, with_sound: bool = True) -> NormalMode:
        return cls(create_victory_detector())

    def detect(self, image: mp.Image, timestamp_ms: int, now: float) -> bool:
        self.status = recognize_gesture(self.detector, image, timestamp_ms)
        name, score = self.status
        return name == "Victory" and score >= 0.62

    def debug_status(self) -> str:
        name, score = self.status
        return (f"[Gesture] {name}, score={score:.2f}, "
                f"crop={getattr(self.detector, 'preferred_crop', None) or 'full'}, "
                f"rotation={getattr(self.detector, 'preferred_rotation', None) or 0}")

    def render(self, frame, background, mask, age, appearing):
        height, width = frame.shape[:2]
        if self.fields is None or self.fields[0].shape != (height, width):
            self.fields = make_dissolve_fields(height, width)
        progress = min(1.0, age / self.effect_seconds)
        return apply_dissolve(frame, background, mask, progress, *self.fields)


class NarutoMode(Mode):
    name = "naruto"
    title = "NARUTO VANISH"
    hint = "Palms together  /  log and smoke"
    effect_seconds = EFFECT_SECONDS
    detect_interval = 0.08

    def __init__(self, detector, sound: VanishSound | None = None,
                 log_sprite: np.ndarray | None = None):
        super().__init__(detector, sound)
        self.log_sprite = load_log() if log_sprite is None else log_sprite
        self.log_cache: dict[int, np.ndarray] = {}
        self.anchor: tuple[int, int, int] | None = None
        self.joined = False

    @classmethod
    def create(cls, with_sound: bool = True) -> NarutoMode:
        detector = create_joined_hands_detector()
        try:
            return cls(detector, VanishSound() if with_sound else None)
        except Exception:
            detector.close()
            raise

    def detect(self, image: mp.Image, timestamp_ms: int, now: float) -> bool:
        self.joined = self.detector.recognize_joined_hands(image, timestamp_ms)
        return self.joined

    def debug_status(self) -> str:
        return (f"[Hands] detected={self.detector.last_hand_count}, joined={self.joined}, "
                f"view={self.detector.preferred_view or 'full'}")

    def begin(self, appearing: bool) -> None:
        self.anchor = None
        super().begin(appearing)

    def reset(self) -> None:
        self.anchor = None

    def needs_mask(self, age: float, mask_age: float) -> bool:
        # The silhouette only matters while the person is still visible.
        return self.anchor is None or age < 0.13

    def render(self, frame, background, mask, age, appearing):
        if self.anchor is None:
            self.anchor = substitution_anchor(mask)
        return render_substitution(frame, background, mask, self.log_sprite, self.anchor,
                                   max(0.04, age), self.log_cache)


class GokuMode(Mode):
    name = "goku"
    title = "GOKU"
    hint = "Two fingers on forehead  /  teleport"
    instructions = ("[Goku] Index and middle fingers together on your forehead for 1 second "
                    "to vanish; repeat to reappear.")
    effect_seconds = GOKU_SECONDS
    detect_interval = 0.07
    manual_keys = False
    toggles = True
    grace_seconds = 0.3

    def __init__(self, detector, sound: VanishSound | None = None):
        super().__init__(detector, sound)
        self.posed = False
        self.reset()

    @classmethod
    def create(cls, with_sound: bool = True) -> GokuMode:
        detector = create_forehead_detector()
        try:
            sound = VanishSound(TELEPORT_SOUND) if with_sound and TELEPORT_SOUND.is_file() else None
            return cls(detector, sound)
        except Exception:
            detector.close()
            raise

    def reset(self) -> None:
        self.hold_started: float | None = None
        self.last_seen = float("-inf")
        self.needs_release = False

    def detect(self, image: mp.Image, timestamp_ms: int, now: float) -> bool:
        self.posed = self.detector.detect(image, timestamp_ms)
        return self.update_hold(self.posed, now)

    def update_hold(self, posed: bool, now: float) -> bool:
        """Fire once the pose has been held long enough; short detection gaps are tolerated."""
        if posed:
            self.last_seen = now
        elif now - self.last_seen > self.grace_seconds:
            if self.hold_started is not None:
                print("[Goku] Pose interrupted: starting over.")
            self.hold_started = None
            self.needs_release = False
            return False
        if self.needs_release:
            return False
        if self.hold_started is None:
            if posed:
                self.hold_started = now
                print(f"[Goku] Pose recognized: hold it for {GOKU_HOLD_SECONDS:g} s...")
            return False
        if now - self.hold_started >= GOKU_HOLD_SECONDS:
            self.hold_started = None
            self.needs_release = True
            print("[Goku] Teleport!")
            return True
        return False

    def debug_status(self) -> str:
        return (f"[Goku] hands={self.detector.last_hand_count}, "
                f"face={self.detector.face_visible}, pose={self.posed}")

    def render(self, frame, background, mask, age, appearing):
        return render_teleport(frame, background, mask, age, appearing)

    def finished_frame(self, background, last_display):
        return background.copy()


MODES: dict[str, type[Mode]] = {mode.name: mode for mode in (NormalMode, NarutoMode, GokuMode)}


def create_mode(name: str, with_sound: bool = True) -> Mode:
    return MODES[name].create(with_sound)
